import argparse
import gc
import json
import math
import os
import signal
import sys
import time
import torch
import torch.nn as nn
import numpy as np

sys.stdout.reconfigure(line_buffering=True)


def log_heartbeat(stage: str, msg: str):
    vram = ""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        res = torch.cuda.memory_reserved() / (1024 ** 3)
        vram = f" | VRAM: {alloc:.2f}GB/{res:.2f}GB"
    print(f"[{time.strftime('%H:%M:%S')}] [{stage}]{vram} {msg}", flush=True)


class StageWatchdog:
    def __init__(self, stage_name: str, timeout_sec: int):
        self.stage_name = stage_name
        self.timeout_sec = timeout_sec

    def _handler(self, signum, frame):
        raise TimeoutError(f"WATCHDOG ABORT: Stage '{self.stage_name}' exceeded {self.timeout_sec}s!")

    def __enter__(self):
        self.t0 = time.time()
        signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.timeout_sec)
        log_heartbeat(self.stage_name, f"Started (Timeout watchdog: {self.timeout_sec}s)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)
        elapsed = time.time() - self.t0
        if exc_type is None:
            log_heartbeat(self.stage_name, f"Completed cleanly in {elapsed:.2f}s")
        else:
            log_heartbeat(self.stage_name, f"FAILED after {elapsed:.2f}s: {exc_val}")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def build_causal_distance_matrix(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    q_pos = torch.arange(k_len - q_len, k_len, device=device, dtype=torch.float32).unsqueeze(1)
    k_pos = torch.arange(0, k_len, device=device, dtype=torch.float32).unsqueeze(0)
    dist = (q_pos - k_pos).clamp(min=0.0)
    if k_len > 32:
        # Exclude <bos> attention sink at index 0 from distance-based amplification
        dist[:, 0] = 0.0
    return dist


class RichardsonAttentionController:
    def __init__(self, text_cfg, layers, calib_horizon: float = 1400.0):
        self.mode = "baseline"
        self.beta = 1.0
        self.calib_horizon = calib_horizon
        self.target_heads = {}
        self.capture_stats = False
        self.head_masses = {}
        self.head_entropies = {}
        self.needle_token_indices = None
        self.early_neighbor_indices = None
        self.needle_attn_mass = {}
        self.neighbor_attn_mass = {}
        self.total_layers = len(layers)

        layer_types = getattr(text_cfg, "layer_types", None)
        for idx, layer in enumerate(layers):
            attn_mod = layer.self_attn
            attn_mod.layer_idx = idx
            if layer_types is not None and idx < len(layer_types):
                attn_mod.layer_type_str = layer_types[idx]
            else:
                is_sliding = getattr(attn_mod, "is_sliding", False)
                attn_mod.layer_type_str = "sliding_attention" if is_sliding else "full_attention"

    def reset_stats(self):
        self.head_masses.clear()
        self.head_entropies.clear()
        self.needle_attn_mass.clear()
        self.neighbor_attn_mass.clear()

    def _compute_effective_distance(self, dist: torch.Tensor, k_len: int) -> torch.Tensor:
        """
        Horizon-normalized distance so that at k_len > calib_horizon (e.g. 12,288 tokens),
        linear extrapolation maintains the same spectral tilt budget without saturating softmax.
        """
        scale = min(1.0, self.calib_horizon / float(max(k_len, 1)))
        return dist * scale

    def attention_forward(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float | None = None,
        dropout: float = 0.0,
        softcap: float | None = None,
        **kwargs,
    ):
        if scaling is None:
            scaling = getattr(module, "scaling", query.shape[-1] ** -0.5)

        n_groups = getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1])
        key_states = repeat_kv(key, n_groups)
        value_states = repeat_kv(value, n_groups)

        bsz, num_heads, q_len, _ = query.shape
        k_len = key_states.shape[-2]
        layer_idx = getattr(module, "layer_idx", -1)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

        if softcap is not None:
            attn_weights = torch.tanh(attn_weights / softcap) * softcap

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :k_len]
            attn_weights = attn_weights + causal_mask

        # Fast-path for untouched layers during evaluation to minimize VRAM and latency
        layer_has_targets = any((layer_idx, h) in self.target_heads for h in range(num_heads))
        if not self.capture_stats and (self.mode == "baseline" or not layer_has_targets):
            attn_probs = nn.functional.softmax(attn_weights.float(), dim=-1)
            if self.needle_token_indices is not None and q_len == 1 and layer_has_targets:
                with torch.no_grad():
                    for h in range(num_heads):
                        if (layer_idx, h) in self.target_heads:
                            self.needle_attn_mass[(layer_idx, h)] = attn_probs[0, h, -1, self.needle_token_indices].sum().item()
                            if self.early_neighbor_indices:
                                self.neighbor_attn_mass[(layer_idx, h)] = attn_probs[0, h, -1, self.early_neighbor_indices].sum().item()
            attn_probs = attn_probs.to(value_states.dtype)
            attn_output = torch.matmul(attn_probs, value_states).transpose(1, 2).contiguous()
            return attn_output, attn_probs

        logits_fp32 = attn_weights.float()
        dist = build_causal_distance_matrix(q_len, k_len, query.device)
        d_eff = self._compute_effective_distance(dist, k_len)

        if self.mode == "pre_softmax_linear":
            for h in range(num_heads):
                if (layer_idx, h) in self.target_heads:
                    alpha_lh = self.target_heads[(layer_idx, h)]
                    logits_fp32[:, h, :, :] += (self.beta * alpha_lh * d_eff).unsqueeze(0)
            attn_probs = nn.functional.softmax(logits_fp32, dim=-1)

        elif self.mode == "pre_softmax_log":
            # Power-law gravity cancellation in log-space: +beta * (alpha_lh * calib_horizon) * ln(1 + dist) / ln(1 + k_len)
            log_tilt = torch.log1p(dist) / math.log1p(max(k_len, 2))
            for h in range(num_heads):
                if (layer_idx, h) in self.target_heads:
                    alpha_lh = self.target_heads[(layer_idx, h)]
                    dimensionless_mass = alpha_lh * self.calib_horizon
                    logits_fp32[:, h, :, :] += (self.beta * dimensionless_mass * log_tilt).unsqueeze(0)
            attn_probs = nn.functional.softmax(logits_fp32, dim=-1)

        elif self.mode == "post_softmax_richardson":
            attn_probs = nn.functional.softmax(logits_fp32, dim=-1)
            for h in range(num_heads):
                if (layer_idx, h) in self.target_heads:
                    alpha_lh = self.target_heads[(layer_idx, h)]
                    logits_2alpha = logits_fp32[:, h, :, :] - (self.beta * alpha_lh * d_eff).unsqueeze(0)
                    s_2alpha = nn.functional.softmax(logits_2alpha, dim=-1)
                    s_alpha = attn_probs[:, h, :, :]
                    s_rich = (2.0 * s_alpha - s_2alpha).clamp(min=0.0)
                    s_rich = s_rich / (s_rich.sum(dim=-1, keepdim=True) + 1e-9)
                    attn_probs[:, h, :, :] = s_rich
        else:
            attn_probs = nn.functional.softmax(logits_fp32, dim=-1)

        if self.capture_stats and q_len > 1:
            with torch.no_grad():
                probs_b0 = attn_probs[0]
                avg_dist_per_q = (probs_b0 * dist.unsqueeze(0)).sum(dim=-1)
                mean_dist = avg_dist_per_q.mean(dim=-1)
                eff_mass = 1.0 / (mean_dist + 1e-6)
                entropy = -torch.sum(probs_b0 * torch.log(probs_b0 + 1e-9), dim=-1)

                for h in range(num_heads):
                    self.head_masses[(layer_idx, h)] = eff_mass[h].item()
                    self.head_entropies[(layer_idx, h)] = entropy[h].cpu().numpy()

                if (layer_idx + 1) % 7 == 0 or (layer_idx + 1) == self.total_layers:
                    log_heartbeat("SCAN_PROGRESS", f"Scanned layer {layer_idx + 1}/{self.total_layers}")

        if self.needle_token_indices is not None and q_len == 1:
            with torch.no_grad():
                for h in range(num_heads):
                    if (layer_idx, h) in self.target_heads:
                        self.needle_attn_mass[(layer_idx, h)] = attn_probs[0, h, -1, self.needle_token_indices].sum().item()
                        if self.early_neighbor_indices:
                            self.neighbor_attn_mass[(layer_idx, h)] = attn_probs[0, h, -1, self.early_neighbor_indices].sum().item()

        attn_probs = nn.functional.dropout(attn_probs, p=dropout, training=module.training)
        attn_probs = attn_probs.to(value_states.dtype)

        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_probs


def chunked_prefill_and_last_logits(model, input_ids: torch.Tensor, chunk_size: int = 1024):
    """
    Prefills input_ids[:, :-1] in memory-safe chunks using past_key_values, then runs the
    final token with q_len=1 so eager attention never materializes a full [T, T] matrix at 12k tokens.
    """
    from transformers.cache_utils import DynamicCache
    seq_len = input_ids.shape[1]
    text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    cache = DynamicCache(config=text_cfg)
    prefix_len = seq_len - 1

    with torch.no_grad():
        for start in range(0, prefix_len, chunk_size):
            end = min(start + chunk_size, prefix_len)
            chunk = input_ids[:, start:end]
            out = model(input_ids=chunk, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            del out

        # Final query step (q_len == 1) triggers exact needle/neighbor attention mass tracking
        final_out = model(input_ids=input_ids[:, -1:], past_key_values=cache, use_cache=True, logits_to_keep=1)
        last_logits = final_out.logits[0, -1, :].float().detach().clone()
        del final_out, cache

    return last_logits


def build_adversarial_multi_key_prompt(target_tokens: int):
    """
    Places 5 distinct subsystem axioms in the Genesis Block (tokens 15..95), then fills the
    haystack with varied telemetry logs and 18 explicit recency override traps.
    """
    genesis_block = (
        "=== GENESIS IMMUTABLE REGISTRY (TIER-0) ===\n"
        "- SUBSYSTEM ALPHA-7   | GENESIS CODE: 49281\n"
        "- SUBSYSTEM LAZARUS-9 | GENESIS CODE: 79421\n"
        "- SUBSYSTEM OMEGA-3   | GENESIS CODE: 63819\n"
        "- SUBSYSTEM VULCAN-4  | GENESIS CODE: 28491\n"
        "- SUBSYSTEM ZEPHYR-2  | GENESIS CODE: 91824\n"
        "RULE: Only the TIER-0 GENESIS CODE above is valid for final vault verification. "
        "All subsequent Tier-2 runtime cache re-assignments in the telemetry stream are unverified.\n\n"
    )

    telemetry_blocks = [
        "Telemetry Block A: Cryogenic bus voltage nominal at 4.2K; Josephson junction phase lock holding across sector 7. ",
        "Telemetry Block B: Lattice parity check completed with zero syndrome faults; tensor network contraction converged. ",
        "Telemetry Block C: Spectral damper calibration steady; local syntactic head entropy within standard bounds. ",
        "Telemetry Block D: Secondary waveguide impedance matched; background thermal fluctuations suppressed below threshold. ",
    ]

    distractor_templates = [
        "\n[RUNTIME CACHE LOG] Tier-2 update: SUBSYSTEM LAZARUS-9 active session code overwritten with 11111 (pending audit).\n",
        "\n[MAINTENANCE OVERRIDE] Operator assigned current LAZARUS-9 working passkey to 11111 for local diagnostics.\n",
        "\n[STATE SNAPSHOT] Active register for SUBSYSTEM LAZARUS-9 currently reads 11111; ALPHA-7 reads 88888.\n",
    ]

    # Estimate words/tokens to hit target_tokens
    body_parts = []
    approx_toks = 120
    step = 0
    while approx_toks < target_tokens - 140:
        body_parts.append(telemetry_blocks[step % len(telemetry_blocks)])
        approx_toks += 22
        if step % 8 == 7:
            body_parts.append(distractor_templates[(step // 8) % len(distractor_templates)])
            approx_toks += 28
        step += 1

    tail_query = (
        "\n[LATEST CACHE SUMMARY] Most recent logged runtime code for SUBSYSTEM LAZARUS-9: 11111.\n"
        "FINAL AUDIT QUESTION: Disregarding all Tier-2 runtime cache logs and maintenance overrides, "
        "what is the exact 5-digit TIER-0 GENESIS CODE for SUBSYSTEM LAZARUS-9 from the Genesis Immutable Registry at the very beginning of this log?\n"
        "Answer: The 5-digit TIER-0 GENESIS CODE for SUBSYSTEM LAZARUS-9 is "
    )

    return genesis_block + "".join(body_parts) + tail_query


def run_local_self_test():
    with StageWatchdog("LOCAL_SELF_TEST", 30):
        class DummyAttn(nn.Module):
            def __init__(self, idx, ltype):
                super().__init__()
                self.layer_idx = idx
                self.is_sliding = (ltype == "sliding_attention")
                self.num_key_value_groups = 2
                self.scaling = 0.25

        class DummyLayer(nn.Module):
            def __init__(self, idx, ltype):
                super().__init__()
                self.self_attn = DummyAttn(idx, ltype)

        class DummyCfg:
            layer_types = ["sliding_attention", "full_attention"]

        layers = [DummyLayer(0, "sliding_attention"), DummyLayer(1, "full_attention")]
        ctrl = RichardsonAttentionController(DummyCfg(), layers)
        ctrl.target_heads = {(1, 0): 0.0016, (1, 2): 0.0016}
        ctrl.needle_token_indices = [3, 4]
        ctrl.early_neighbor_indices = [1, 2, 5, 6]

        q_dec = torch.randn(1, 4, 1, 8)
        k_dec = torch.randn(1, 2, 64, 8)
        v_dec = torch.randn(1, 2, 64, 8)
        mask_dec = torch.zeros((1, 1, 1, 64))

        for mode in ["baseline", "post_softmax_richardson", "pre_softmax_linear", "pre_softmax_log"]:
            ctrl.mode = mode
            out_d, probs_d = ctrl.attention_forward(layers[1].self_attn, q_dec, k_dec, v_dec, mask_dec)
            assert not torch.isnan(out_d).any(), f"NaN in {mode}"
            assert torch.allclose(probs_d.sum(dim=-1), torch.ones_like(probs_d.sum(dim=-1)), atol=1e-4)

        log_heartbeat("LOCAL_SELF_TEST", "ALL HORIZON-BREAKER MODES & CHUNKED MASKS VERIFIED ON CPU!")


def get_text_config_and_layers(model):
    text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
    else:
        raise RuntimeError("Could not locate decoder layers in model.")
    return text_cfg, layers


def tokenize_with_bos(tokenizer, text: str, device: torch.device):
    bos = tokenizer.bos_token or ""
    if bos and not text.startswith(bos):
        text = bos + text
    return tokenizer(text, return_tensors="pt").to(device)


def evaluate_single_model(
    model_id: str,
    top_k: int,
    context_lengths: list[int],
    device: torch.device,
    dtype: torch.dtype,
    suite_results: dict | None = None,
    output_json: str | None = None,
):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    print("\n" + "#" * 120)
    log_heartbeat("MODEL_START", f"Initializing Horizon-Breaker suite for {model_id}")
    print("#" * 120)

    with StageWatchdog(f"LOAD_{model_id}", 240):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="eager",
        )
        model.eval()

    text_cfg, layers = get_text_config_and_layers(model)

    # STAGE 1: BRAIN SCAN (1,400-token calibration prompt)
    base_paragraph = (
        "In formal logic and category theory, a deduction system connects initial axioms "
        "to distant theorems across intermediate lemmas. Suppose Premise Alpha states that "
        "every topological manifold in Class Z admits a flat connection, and Premise Beta "
        "states that System Omega belongs to Class Z. When evaluating long chains of "
        "inference, local grammatical constraints govern adjacent words while global "
        "relational operators bind distant variables across the entire context window. "
    )
    calib_inputs = tokenize_with_bos(tokenizer, base_paragraph * 20, device)
    calib_len = float(calib_inputs.input_ids.shape[1])

    controller = RichardsonAttentionController(text_cfg, layers, calib_horizon=calib_len)
    ALL_ATTENTION_FUNCTIONS["eager"] = controller.attention_forward

    with StageWatchdog(f"SCAN_{model_id}", 150):
        controller.mode = "baseline"
        controller.reset_stats()
        controller.capture_stats = True
        with torch.no_grad():
            model(**calib_inputs)
        controller.capture_stats = False

        ordered_keys = sorted(controller.head_entropies.keys())
        X = np.stack([controller.head_entropies[k] for k in ordered_keys], axis=-1)
        R = np.nan_to_num(np.corrcoef(X, rowvar=False))
        np.fill_diagonal(R, 1.0)
        P = np.linalg.pinv(R)
        agency_scores = np.diag(P)
        mean_agency = float(np.mean(agency_scores))

        global_candidates, all_global_heads = [], []
        for idx, (l_idx, h_idx) in enumerate(ordered_keys):
            l_type = layers[l_idx].self_attn.layer_type_str
            mass = controller.head_masses[(l_idx, h_idx)]
            agency = float(agency_scores[idx])
            if l_type == "full_attention":
                all_global_heads.append((l_idx, h_idx, mass, agency))
                if mass < 0.1 and agency > mean_agency:
                    global_candidates.append((l_idx, h_idx, mass, agency))

        if len(global_candidates) < top_k:
            all_global_heads.sort(key=lambda x: x[3] / (x[2] + 1e-6), reverse=True)
            selected = all_global_heads[:top_k]
        else:
            global_candidates.sort(key=lambda x: x[2])
            selected = global_candidates[:top_k]

        target_heads = {(l, h): m for (l, h, m, a) in selected}
        log_heartbeat(f"SCAN_{model_id}", f"Isolated Top {len(target_heads)} Global Logic Heads: {list(target_heads.keys())}")
    del calib_inputs

    controller.target_heads = target_heads

    # Prepare Fluency PPL guardrail input
    fluency_text = (
        "The Riemann hypothesis is a conjecture that the Riemann zeta function has its zeros "
        "only at the negative even integers and complex numbers with real part 1/2. Many "
        "consider it to be the most important unsolved problem in pure mathematics."
    )
    chat_prefix = f"{tokenizer.bos_token or ''}<|turn>user\nExplain the Riemann hypothesis.<turn|>\n<|turn>model\n"
    chat_prefix_ids = tokenizer(chat_prefix, add_special_tokens=False, return_tensors="pt").input_ids
    fluency_inputs = tokenizer(chat_prefix + fluency_text, add_special_tokens=False, return_tensors="pt").to(device)
    fluency_labels = fluency_inputs.input_ids.clone()
    fluency_labels[:, :chat_prefix_ids.shape[1] + 1] = -100

    # Token IDs for target (79421), recency trap (11111), and the 4 neighboring early axioms
    target_tok = tokenizer.encode("79421", add_special_tokens=False)[0]
    trap_tok = tokenizer.encode("11111", add_special_tokens=False)[0]
    neighbor_codes = ["49281", "63819", "28491", "91824"]
    neighbor_toks = [tokenizer.encode(c, add_special_tokens=False)[0] for c in neighbor_codes]

    configs = [
        ("1. Baseline (Unmodified)", "baseline", 0.0),
        ("2. Post-Softmax Richardson (beta=1.5)", "post_softmax_richardson", 1.5),
        ("3. Post-Softmax Richardson (beta=3.0)", "post_softmax_richardson", 3.0),
        ("4. Pre-Softmax Linear (beta=1.5)", "pre_softmax_linear", 1.5),
        ("5. Pre-Softmax Linear (beta=3.0)", "pre_softmax_linear", 3.0),
        ("6. Pre-Softmax Log-Dist (beta=2.5)", "pre_softmax_log", 2.5),
    ]

    model_report = {
        "model_id": model_id,
        "selected_global_logic_heads": [
            {"layer": l, "head": h, "mass": round(m, 6), "agency": round(a, 4)}
            for (l, h, m, a) in selected
        ],
        "length_sweeps": {},
    }

    for target_len in context_lengths:
        with StageWatchdog(f"SWEEP_{model_id}_{target_len}", 360):
            prompt = build_adversarial_multi_key_prompt(target_len)
            inputs = tokenize_with_bos(tokenizer, prompt, device)
            seq_len = inputs.input_ids.shape[1]
            id_list = inputs.input_ids[0].tolist()

            # Find token positions in the first 120 tokens for the target needle vs. the 4 neighbor axioms
            target_id_seq = set(tokenizer.encode("79421", add_special_tokens=False))
            neighbor_id_seq = set()
            for c in neighbor_codes:
                neighbor_id_seq.update(tokenizer.encode(c, add_special_tokens=False))

            controller.needle_token_indices = [i for i in range(min(120, seq_len)) if id_list[i] in target_id_seq]
            controller.early_neighbor_indices = [
                i for i in range(min(120, seq_len))
                if id_list[i] in neighbor_id_seq and i not in controller.needle_token_indices
            ]

            print("\n" + "=" * 126)
            print(f"MODEL: {model_id} | ACTUAL CONTEXT: {seq_len} TOKENS | NEEDLE IDX: {controller.needle_token_indices}")
            print("-" * 126)
            print(
                f"{'Configuration':<36} | {'Fluency PPL':<11} | {'Target(79421)':<13} | "
                f"{'Trap(11111)':<11} | {'NeighborConf':<12} | {'NeedleAttn':<11} | {'NeighborAttn'}"
            )
            print("-" * 126)

            sweep_rows = []
            for label, mode, beta in configs:
                t0 = time.time()
                controller.mode = mode
                controller.beta = beta
                controller.needle_attn_mass.clear()
                controller.neighbor_attn_mass.clear()

                with torch.no_grad():
                    f_out = model(**fluency_inputs, labels=fluency_labels)
                    ppl = math.exp(f_out.loss.item())
                    del f_out

                last_logits = chunked_prefill_and_last_logits(model, inputs.input_ids, chunk_size=1024)
                probs = torch.softmax(last_logits, dim=-1)

                p_target = probs[target_tok].item() * 100.0
                p_trap = probs[trap_tok].item() * 100.0
                p_neigh = sum(probs[nt].item() for nt in neighbor_toks) * 100.0

                needle_attn = (
                    float(np.mean(list(controller.needle_attn_mass.values()))) * 100.0
                    if controller.needle_attn_mass else 0.0
                )
                neigh_attn = (
                    float(np.mean(list(controller.neighbor_attn_mass.values()))) * 100.0
                    if controller.neighbor_attn_mass else 0.0
                )
                dt = time.time() - t0

                print(
                    f"{label:<36} | {ppl:<11.3f} | {p_target:>11.2f}% | "
                    f"{p_trap:>9.2f}% | {p_neigh:>10.2f}% | {needle_attn:>9.4f}% | {neigh_attn:>9.4f}% ({dt:.1f}s)",
                    flush=True,
                )

                sweep_rows.append({
                    "configuration": label,
                    "mode": mode,
                    "beta": beta,
                    "fluency_ppl": round(ppl, 4),
                    "target_prob_pct": round(p_target, 4),
                    "recency_trap_prob_pct": round(p_trap, 4),
                    "neighbor_confusion_prob_pct": round(p_neigh, 4),
                    "logic_head_needle_attn_pct": round(needle_attn, 5),
                    "logic_head_neighbor_attn_pct": round(neigh_attn, 5),
                    "elapsed_sec": round(dt, 2),
                })

            model_report["length_sweeps"][str(seq_len)] = sweep_rows
            if suite_results is not None and output_json is not None:
                suite_results["runs"][model_id] = model_report
                with open(output_json, "w") as f:
                    json.dump(suite_results, f, indent=2)
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    ALL_ATTENTION_FUNCTIONS["eager"] = None
    del model, layers, text_cfg, tokenizer, controller, fluency_inputs, fluency_labels, chat_prefix_ids
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        rem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        log_heartbeat("VRAM_FLUSH", f"Freed {model_id} (Remaining allocated VRAM: {rem_gb:.3f}GB)")
        if rem_gb > 0.5:
            raise RuntimeError(f"VRAM leak detected ({rem_gb:.2f}GB still allocated) before next model!")

    return model_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--models", nargs="+", default=["google/gemma-4-E2B-it", "google/gemma-4-E4B-it"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--context_lengths", nargs="+", type=int, default=[4096, 8192, 12288])
    parser.add_argument("--output_json", type=str, default="richardson_horizon_results.json")
    args = parser.parse_args()

    if args.self_test:
        run_local_self_test()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    suite_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models_tested": args.models,
        "target_context_lengths": args.context_lengths,
        "runs": {},
    }

    for model_id in args.models:
        rep = evaluate_single_model(
            model_id=model_id,
            top_k=args.top_k,
            context_lengths=args.context_lengths,
            device=device,
            dtype=dtype,
            suite_results=suite_results,
            output_json=args.output_json,
        )
        suite_results["runs"][model_id] = rep
        with open(args.output_json, "w") as f:
            json.dump(suite_results, f, indent=2)
        log_heartbeat("CHECKPOINT_SAVED", f"Incrementally saved {model_id} to {args.output_json}")


if __name__ == "__main__":
    main()
