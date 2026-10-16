import argparse
import json
import math
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
        alloc = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / (1024 ** 3)
        vram = f" | Total VRAM: {alloc:.2f}GB"
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
        log_heartbeat(self.stage_name, f"Started (Timeout: {self.timeout_sec}s)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)
        elapsed = time.time() - self.t0
        if exc_type is None:
            log_heartbeat(self.stage_name, f"Completed in {elapsed:.2f}s")
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


def build_causal_distance_and_positions(q_start: int, q_end: int, q_len: int, k_len: int, device: torch.device):
    q_pos = torch.arange(k_len - q_len + q_start, k_len - q_len + q_end, device=device, dtype=torch.float32).unsqueeze(1)
    k_pos = torch.arange(0, k_len, device=device, dtype=torch.float32).unsqueeze(0)
    dist = (q_pos - k_pos).clamp(min=0.0)
    if k_len > 32:
        dist[:, 0] = 0.0  # Protect <bos> token-0 attention sink
    return dist, q_pos


class Phase4RichardsonController:
    def __init__(self, text_cfg, layers, calib_horizon: float = 1400.0, q_chunk_size: int = 2048):
        self.mode = "baseline"
        self.beta = 1.0
        self.eta = 0.65
        self.use_sharpening = False
        self.calib_horizon = calib_horizon
        self.q_chunk_size = q_chunk_size
        self.target_heads = {}  # Dict[(layer_idx, head_idx) -> calibrated_kappa]
        self.capture_stats = False
        self.head_masses = {}
        self.head_entropies = {}
        self.zone_a_indices = []
        self.zone_b_indices = []
        self.zone_c_indices = []
        self.zone_a_attn = {}
        self.zone_b_attn = {}
        self.zone_c_attn = {}

        layer_types = getattr(text_cfg, "layer_types", None)
        for idx, layer in enumerate(layers):
            attn_mod = layer.self_attn
            attn_mod.layer_idx = idx
            if layer_types is not None and idx < len(layer_types):
                attn_mod.layer_type_str = layer_types[idx]
            else:
                is_sliding = getattr(attn_mod, "is_sliding", False)
                attn_mod.layer_type_str = "sliding_attention" if is_sliding else "full_attention"

    def clear_zone_attn(self):
        self.zone_a_attn.clear()
        self.zone_b_attn.clear()
        self.zone_c_attn.clear()

    def _record_zone_attn(self, layer_idx: int, num_heads: int, attn_probs: torch.Tensor):
        k_len = attn_probs.shape[-1]
        if self.zone_a_indices and self.zone_c_indices and k_len > self.zone_c_indices[-1]:
            with torch.no_grad():
                for h in range(num_heads):
                    if (layer_idx, h) in self.target_heads:
                        last_row = attn_probs[0, h, -1, :]
                        self.zone_a_attn[(layer_idx, h)] = last_row[self.zone_a_indices].sum().item()
                        self.zone_b_attn[(layer_idx, h)] = last_row[self.zone_b_indices].sum().item()
                        self.zone_c_attn[(layer_idx, h)] = last_row[self.zone_c_indices].sum().item()

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

        bsz, num_heads, q_len, head_dim = query.shape
        k_len = key_states.shape[-2]
        layer_idx = getattr(module, "layer_idx", -1)
        layer_has_targets = any((layer_idx, h) in self.target_heads for h in range(num_heads))

        attn_output = torch.empty((bsz, num_heads, q_len, value_states.shape[-1]), dtype=value_states.dtype, device=query.device)
        chunk_step = q_len if self.capture_stats else self.q_chunk_size

        for q_start in range(0, q_len, chunk_step):
            q_end = min(q_start + chunk_step, q_len)
            q_slice = query[:, :, q_start:q_end, :]

            attn_weights = torch.matmul(q_slice, key_states.transpose(2, 3)) * scaling
            if softcap is not None:
                attn_weights = torch.tanh(attn_weights / softcap) * softcap
            if attention_mask is not None:
                if attention_mask.shape[-2] == q_len:
                    mask_slice = attention_mask[:, :, q_start:q_end, :k_len]
                else:
                    mask_slice = attention_mask[:, :, :, :k_len]
                attn_weights = attn_weights + mask_slice.to(attn_weights.device)

            # Fast path for untouched sliding-window layers and baseline evaluation
            if not self.capture_stats and (self.mode == "baseline" or not layer_has_targets):
                logits_fp32 = attn_weights.float()
                del attn_weights
                attn_probs = nn.functional.softmax(logits_fp32, dim=-1)
                del logits_fp32
                if layer_has_targets and q_end == q_len:
                    self._record_zone_attn(layer_idx, num_heads, attn_probs)
                attn_output[:, :, q_start:q_end, :] = torch.matmul(attn_probs.to(value_states.dtype), value_states)
                del attn_probs
                continue

            logits_fp32 = attn_weights.float()
            del attn_weights
            dist, q_pos = build_causal_distance_and_positions(q_start, q_end, q_len, k_len, query.device)

            if self.mode == "phase3_global_log":
                log_tilt = torch.log1p(dist) / math.log1p(max(k_len, 2))
                for h in range(num_heads):
                    if (layer_idx, h) in self.target_heads:
                        kappa = self.target_heads[(layer_idx, h)]
                        logits_fp32[:, h, :, :] += (self.beta * kappa * log_tilt).unsqueeze(0)
                attn_probs = nn.functional.softmax(logits_fp32, dim=-1)
                del logits_fp32, log_tilt

            elif self.mode in ("row_horizon_log", "dual_state_log_richardson"):
                # [TODO-1] Row-Wise Causal Horizon Normalization + 32-token local syntax guardband
                dist_long = (dist - 32.0).clamp_(min=0.0)
                row_denom = torch.log1p(q_pos.clamp(min=self.calib_horizon))
                log_tilt_row = torch.log1p_(dist_long) / row_denom  # [q_slice_len, k_len]
                del dist_long, row_denom

                # [TODO-4] Optional Log-Length Attention Sharpening per query row i
                if self.use_sharpening:
                    sharpen_row = (torch.log(q_pos.clamp(min=2.0)) / math.log(self.calib_horizon)).clamp(min=1.0, max=1.22)
                else:
                    sharpen_row = None

                if self.mode == "row_horizon_log":
                    for h in range(num_heads):
                        if (layer_idx, h) in self.target_heads:
                            kappa = self.target_heads[(layer_idx, h)]
                            head_logits = logits_fp32[:, h, :, :]
                            if sharpen_row is not None:
                                valid_mask = (head_logits > -1e4).float()
                                head_logits = head_logits * (1.0 + (sharpen_row.unsqueeze(0) - 1.0) * valid_mask)
                                del valid_mask
                            logits_fp32[:, h, :, :] = head_logits + (self.beta * kappa * log_tilt_row).unsqueeze(0)
                    attn_probs = nn.functional.softmax(logits_fp32, dim=-1)
                    del logits_fp32, log_tilt_row

                else:
                    # [TODO-2] Dual-State Symmetric Log-Richardson: norm(max(0, (1+eta)*S_free - eta*S_heavy))
                    for h in range(num_heads):
                        head_logits = logits_fp32[:, h, :, :]
                        if (layer_idx, h) in self.target_heads:
                            kappa = self.target_heads[(layer_idx, h)]
                            if sharpen_row is not None:
                                valid_mask = (head_logits > -1e4).float()
                                head_logits = head_logits * (1.0 + (sharpen_row.unsqueeze(0) - 1.0) * valid_mask)
                                del valid_mask

                            delta_l = (self.beta * kappa * log_tilt_row).unsqueeze(0)
                            s_free = nn.functional.softmax(head_logits + delta_l, dim=-1)
                            s_heavy = nn.functional.softmax(head_logits - delta_l, dim=-1)
                            s_super = ((1.0 + self.eta) * s_free - self.eta * s_heavy).clamp_(min=0.0)
                            s_super = s_super / (s_super.sum(dim=-1, keepdim=True) + 1e-9)
                            logits_fp32[:, h, :, :] = s_super
                            del delta_l, s_free, s_heavy, s_super
                        else:
                            logits_fp32[:, h, :, :] = nn.functional.softmax(head_logits, dim=-1)
                    attn_probs = logits_fp32
                    del log_tilt_row
            else:
                attn_probs = nn.functional.softmax(logits_fp32, dim=-1)
                del logits_fp32

            if self.capture_stats and q_len > 1:
                with torch.no_grad():
                    probs_b0 = attn_probs[0]
                    mean_dist = (probs_b0 * dist.unsqueeze(0)).sum(dim=-1).mean(dim=-1)
                    eff_mass = 1.0 / (mean_dist + 1e-6)
                    entropy = -torch.sum(probs_b0 * torch.log(probs_b0 + 1e-9), dim=-1)
                    for h in range(num_heads):
                        self.head_masses[(layer_idx, h)] = eff_mass[h].item()
                        self.head_entropies[(layer_idx, h)] = entropy[h].cpu().numpy()

            del dist, q_pos

            if layer_has_targets and q_end == q_len:
                self._record_zone_attn(layer_idx, num_heads, attn_probs)

            attn_output[:, :, q_start:q_end, :] = torch.matmul(attn_probs.to(value_states.dtype), value_states)
            del attn_probs

        return attn_output.transpose(1, 2).contiguous(), None


def build_phase4_multihop_prompts(target_tokens: int = 8000):
    """
    [TODO-5] Strict Temporal Separation:
    - Zone A (toks ~90), Barrier 1 (pure telemetry), Zone B (toks ~1730), Barrier 2 (pure telemetry),
      and Zone C (toks ~3370) establish all Tier-0 Genesis axioms with ZERO early decoys.
    - All Tier-2 runtime audit decoys are placed strictly in Barrier 3 (chronologically after Zone C).
    """
    zone_a = (
        "=== ZONE A: TIER-0 IMMUTABLE VAULT CODE REGISTRY ===\n"
        "- VAULT-ALPHA    | TIER-0 GENESIS CODE: 49281\n"
        "- VAULT-HELIOS   | TIER-0 GENESIS CODE: 54712\n"
        "- VAULT-HYPERION | TIER-0 GENESIS CODE: 83914\n"
        "- VAULT-LAZARUS  | TIER-0 GENESIS CODE: 79421\n"
        "- VAULT-NYX      | TIER-0 GENESIS CODE: 37192\n"
        "- VAULT-OMEGA    | TIER-0 GENESIS CODE: 63819\n"
        "- VAULT-VULCAN   | TIER-0 GENESIS CODE: 28491\n"
        "- VAULT-ZEPHYR   | TIER-0 GENESIS CODE: 91824\n\n"
    )
    zone_b = (
        "\n=== ZONE B: TIER-0 COMMAND SECTOR ROUTING REGISTRY ===\n"
        "- SECTOR-ATLAS    | ASSIGNED PRIMARY VAULT: VAULT-OMEGA\n"
        "- SECTOR-BOREALIS | ASSIGNED PRIMARY VAULT: VAULT-ALPHA\n"
        "- SECTOR-CYGNUS   | ASSIGNED PRIMARY VAULT: VAULT-ZEPHYR\n"
        "- SECTOR-DRACO    | ASSIGNED PRIMARY VAULT: VAULT-HELIOS\n"
        "- SECTOR-EREBUS   | ASSIGNED PRIMARY VAULT: VAULT-VULCAN\n"
        "- SECTOR-KRONOS   | ASSIGNED PRIMARY VAULT: VAULT-LAZARUS\n"
        "- SECTOR-ORION    | ASSIGNED PRIMARY VAULT: VAULT-NYX\n"
        "- SECTOR-PEGASUS  | ASSIGNED PRIMARY VAULT: VAULT-HYPERION\n\n"
    )
    zone_c = (
        "\n=== ZONE C: TIER-0 DIRECTORATE COMMAND REGISTRY ===\n"
        "- DIRECTOR-CHEN     | AUTHORIZED COMMAND SECTOR: SECTOR-BOREALIS\n"
        "- DIRECTOR-KAEL     | AUTHORIZED COMMAND SECTOR: SECTOR-ORION\n"
        "- DIRECTOR-MERCER   | AUTHORIZED COMMAND SECTOR: SECTOR-DRACO\n"
        "- DIRECTOR-ROSTOVA  | AUTHORIZED COMMAND SECTOR: SECTOR-PEGASUS\n"
        "- DIRECTOR-STERLING | AUTHORIZED COMMAND SECTOR: SECTOR-ATLAS\n"
        "- DIRECTOR-THORNE   | AUTHORIZED COMMAND SECTOR: SECTOR-CYGNUS\n"
        "- DIRECTOR-VANCE    | AUTHORIZED COMMAND SECTOR: SECTOR-KRONOS\n"
        "- DIRECTOR-WEBB     | AUTHORIZED COMMAND SECTOR: SECTOR-EREBUS\n\n"
    )

    filler = (
        "Telemetry Log: Cryogenic bus voltage nominal at 4.2K; Josephson junction phase coherence locked. "
        "Syndrome parity check completed with zero faults; spectral damper calibration steady. "
    )
    decoy = (
        "\n[TIER-2 RUNTIME AUDIT] Temporary cross-check: DIRECTOR-VANCE inspected SECTOR-ATLAS and "
        "VAULT-OMEGA using temporary maintenance override key 11111 (non-binding).\n"
    )

    # Barrier 1 and Barrier 2 are 100% pure telemetry (>1,350 tokens each) so zero decoys precede Zone C
    barrier_1 = filler * 40
    barrier_2 = filler * 40

    # Build Barrier 3 in two decoy regimes of identical ~8,000-token length:
    # 1. Moderate Decoy Regime (6 decoys in Barrier 3)
    # 2. Heavy Decoy Regime (16 decoys in Barrier 3)
    b3_moderate = (filler * 20 + decoy) * 6 + filler * 12
    b3_heavy = (filler * 7 + decoy) * 16 + filler * 4

    prefix_mod = zone_a + barrier_1 + zone_b + barrier_2 + zone_c + b3_moderate
    prefix_heavy = zone_a + barrier_1 + zone_b + barrier_2 + zone_c + b3_heavy

    q_1hop = (
        "\nQuestion: What is the 5-digit TIER-0 GENESIS CODE for VAULT-LAZARUS?\n"
        "Answer: The 5-digit TIER-0 GENESIS CODE is "
    )
    q_2hop = (
        "\nQuestion: Trace the Sector Routing Registry to the Vault Code Registry: what is the 5-digit "
        "TIER-0 GENESIS CODE for the primary vault assigned to SECTOR-KRONOS?\n"
        "Answer: The 5-digit TIER-0 GENESIS CODE is "
    )
    q_3hop = (
        "\nQuestion: Trace the Directorate Command Registry to the Sector Routing Registry and Vault Code "
        "Registry: what is the 5-digit TIER-0 GENESIS CODE for the primary vault commanded by DIRECTOR-VANCE?\n"
        "Answer: The 5-digit TIER-0 GENESIS CODE is "
    )

    return {
        "1-Hop": prefix_heavy + q_1hop,
        "2-Hop": prefix_heavy + q_2hop,
        "3-Hop (Clean-6d)": prefix_mod + q_3hop,
        "3-Hop (Heavy-16d)": prefix_heavy + q_3hop,
    }


def find_subseq(full_ids: list[int], sub_ids: list[int]) -> list[int]:
    inner = sub_ids[1:-1] if len(sub_ids) >= 4 else sub_ids
    m = len(inner)
    for i in range(len(full_ids) - m + 1):
        if full_ids[i : i + m] == inner:
            return list(range(i, i + m))
    return []


def build_quorum_head_map(by_layer: dict, heads_per_layer: int, calib_horizon: float) -> dict:
    """
    Selects the top `heads_per_layer` low-mass heads per Global Layer and calibrates
    their dimensionless coupling kappa_lh = clamp(alpha_lh * calib_horizon, 1.8, 2.6)
    so higher-mass secondary heads in the quorum do not over-saturate logits.
    """
    quorum_map = {}
    for l_idx, h_list in sorted(by_layer.items()):
        # Sort primarily by lowest effective mass (longest natural lookback), breaking ties by agency
        ranked = sorted(h_list, key=lambda x: (x[2], -x[3]))
        for (l, h, m, a) in ranked[:heads_per_layer]:
            raw_kappa = m * calib_horizon
            calibrated_kappa = float(np.clip(raw_kappa, 1.8, 2.6))
            quorum_map[(l, h)] = calibrated_kappa
    return quorum_map


def run_self_test():
    with StageWatchdog("LOCAL_SELF_TEST", 20):
        prompts = build_phase4_multihop_prompts(8000)
        assert set(prompts.keys()) == {"1-Hop", "2-Hop", "3-Hop (Clean-6d)", "3-Hop (Heavy-16d)"}

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
        ctrl = Phase4RichardsonController(DummyCfg(), layers, calib_horizon=1400.0)
        ctrl.target_heads = {(1, 0): 2.3, (1, 1): 2.3, (1, 2): 2.3, (1, 3): 2.3}
        ctrl.zone_a_indices = [2, 3]
        ctrl.zone_b_indices = [10, 11]
        ctrl.zone_c_indices = [20, 21]

        q = torch.randn(1, 4, 64, 8)
        k = torch.randn(1, 2, 64, 8)
        v = torch.randn(1, 2, 64, 8)
        mask = torch.triu(torch.full((1, 1, 64, 64), float("-inf")), diagonal=1)

        for mode in ["baseline", "phase3_global_log", "row_horizon_log", "dual_state_log_richardson"]:
            ctrl.mode = mode
            ctrl.use_sharpening = (mode == "dual_state_log_richardson")
            ctrl.q_chunk_size = 64
            out_full, _ = ctrl.attention_forward(layers[1].self_attn, q, k, v, mask)
            ctrl.q_chunk_size = 16
            out_chunk, _ = ctrl.attention_forward(layers[1].self_attn, q, k, v, mask)
            assert not torch.isnan(out_chunk).any(), f"NaN in {mode}"
            assert torch.allclose(out_full, out_chunk, atol=1e-5), f"Chunk mismatch in {mode}"

        # Also verify short fluency sequences (k_len=16 < zone_c_indices[-1]=3345) do not index out of bounds
        ctrl.zone_a_indices = [88, 111]
        ctrl.zone_b_indices = [1728, 1749]
        ctrl.zone_c_indices = [3327, 3345]
        out_short, _ = ctrl.attention_forward(layers[1].self_attn, q[:, :, :16, :], k[:, :, :16, :], v[:, :, :16, :], mask[:, :, :16, :16])
        assert not torch.isnan(out_short).any()

        log_heartbeat("LOCAL_SELF_TEST", "ALL PHASE 4 TODO MODES, QUORUM MAPS, AND DUAL-STATE FILTERS PASSED!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-E4B-it")
    parser.add_argument("--target_tokens", type=int, default=8000)
    parser.add_argument("--output_json", type=str, default="richardson_phase4_results.json")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    with StageWatchdog("LOAD_MODEL", 240):
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="eager",
        )
        model.eval()

    text_cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    layers = (
        model.model.language_model.layers
        if hasattr(model, "model") and hasattr(model.model, "language_model")
        else model.model.layers
    )

    bos = tokenizer.bos_token or ""
    calib_text = bos + (
        "In formal logic and category theory, a deduction system connects initial axioms "
        "to distant theorems across intermediate lemmas. Suppose Premise Alpha states that "
        "every topological manifold in Class Z admits a flat connection, and Premise Beta "
        "states that System Omega belongs to Class Z. "
    ) * 25
    calib_inputs = tokenizer(calib_text, return_tensors="pt").to(device)
    calib_horizon = float(calib_inputs.input_ids.shape[1])

    controller = Phase4RichardsonController(text_cfg, layers, calib_horizon=calib_horizon)
    ALL_ATTENTION_FUNCTIONS["eager"] = controller.attention_forward

    # STAGE 1: BRAIN SCAN & QUORUM CONSTRUCTION
    with StageWatchdog("STAGE1_SCAN", 120):
        controller.capture_stats = True
        with torch.no_grad():
            model(**calib_inputs, use_cache=False, logits_to_keep=1)
        controller.capture_stats = False
        del calib_inputs

        ordered_keys = sorted(controller.head_entropies.keys())
        X = np.stack([controller.head_entropies[k] for k in ordered_keys], axis=-1)
        R = np.nan_to_num(np.corrcoef(X, rowvar=False))
        np.fill_diagonal(R, 1.0)
        agency_scores = np.diag(np.linalg.pinv(R))

        by_layer = {}
        for idx, (l_idx, h_idx) in enumerate(ordered_keys):
            if layers[l_idx].self_attn.layer_type_str == "full_attention":
                m = controller.head_masses[(l_idx, h_idx)]
                a = float(agency_scores[idx])
                by_layer.setdefault(l_idx, []).append((l_idx, h_idx, m, a))

        relay1_map = build_quorum_head_map(by_layer, heads_per_layer=1, calib_horizon=calib_horizon)
        quorum4_map = build_quorum_head_map(by_layer, heads_per_layer=4, calib_horizon=calib_horizon)
        quorum6_map = build_quorum_head_map(by_layer, heads_per_layer=6, calib_horizon=calib_horizon)

        log_heartbeat("STAGE1_SCAN", f"Relay-1 (7 Global Heads): {list(relay1_map.keys())}")
        log_heartbeat("STAGE1_SCAN", f"Quorum-4 (28 Global Heads): {list(quorum4_map.keys())}")

    # Prepare Fluency PPL Guardrail
    fluency_text = (
        "The Riemann hypothesis is a conjecture that the Riemann zeta function has its zeros "
        "only at the negative even integers and complex numbers with real part 1/2. Many "
        "consider it to be the most important unsolved problem in pure mathematics."
    )
    chat_prefix = f"{bos}<|turn>user\nExplain the Riemann hypothesis.<turn|>\n<|turn>model\n"
    chat_prefix_ids = tokenizer(chat_prefix, add_special_tokens=False, return_tensors="pt").input_ids
    fluency_inputs = tokenizer(chat_prefix + fluency_text, add_special_tokens=False, return_tensors="pt").to(device)
    fluency_labels = fluency_inputs.input_ids.clone()
    fluency_labels[:, :chat_prefix_ids.shape[1] + 1] = -100

    # STAGE 2: PHASE 4 ABLATION SUITE
    with StageWatchdog("STAGE2_PHASE4_SUITE", 420):
        prompts = build_phase4_multihop_prompts(args.target_tokens)
        tokenized_prompts = {
            hop: tokenizer(bos + text, return_tensors="pt").to(device)
            for hop, text in prompts.items()
        }
        sample_ids = tokenized_prompts["3-Hop (Heavy-16d)"].input_ids[0].tolist()
        seq_len = len(sample_ids)

        controller.zone_a_indices = find_subseq(
            sample_ids, tokenizer.encode("\n- VAULT-LAZARUS  | TIER-0 GENESIS CODE: 79421\n", add_special_tokens=False)
        )
        controller.zone_b_indices = find_subseq(
            sample_ids, tokenizer.encode("\n- SECTOR-KRONOS   | ASSIGNED PRIMARY VAULT: VAULT-LAZARUS\n", add_special_tokens=False)
        )
        controller.zone_c_indices = find_subseq(
            sample_ids, tokenizer.encode("\n- DIRECTOR-VANCE    | AUTHORIZED COMMAND SECTOR: SECTOR-KRONOS\n", add_special_tokens=False)
        )
        assert controller.zone_a_indices and controller.zone_b_indices and controller.zone_c_indices, (
            f"Failed to locate Zone indices: A={controller.zone_a_indices}, B={controller.zone_b_indices}, C={controller.zone_c_indices}"
        )

        target_tok = tokenizer.encode("79421", add_special_tokens=False)[0]
        omega_tok = tokenizer.encode("63819", add_special_tokens=False)[0]  # Decoy VAULT-OMEGA code
        trap_tok = tokenizer.encode("11111", add_special_tokens=False)[0]
        competing_codes = ["49281", "54712", "83914", "37192", "63819", "28491", "91824"]
        competing_toks = [tokenizer.encode(c, add_special_tokens=False)[0] for c in competing_codes]
        all_vault_toks = [target_tok] + competing_toks

        # (label, mode, beta, eta, sharpen, head_map)
        configs = [
            ("1. Baseline (Unmodified)", "baseline", 0.0, 0.0, False, relay1_map),
            ("2. Phase-3 Ref (Global-Log, 7-Head, b=2.5)", "phase3_global_log", 2.5, 0.0, False, relay1_map),
            ("3. [TODO-1+5] Row-Log (7-Head, b=2.5)", "row_horizon_log", 2.5, 0.0, False, relay1_map),
            ("4. [TODO-1+3] Row-Log (Quorum-28, b=3.0)", "row_horizon_log", 3.0, 0.0, False, quorum4_map),
            ("5. [TODO-1+2+3] Dual-Rich (Quorum-28, b=3.0)", "dual_state_log_richardson", 3.0, 0.65, False, quorum4_map),
            ("6. [ALL TODOs 1-5] Full Relay (Quorum-28+Sharp, b=3.5)", "dual_state_log_richardson", 3.5, 0.75, True, quorum4_map),
            ("7. [ALL TODOs 1-5] Deep Quorum-42 (b=3.5)", "dual_state_log_richardson", 3.5, 0.75, True, quorum6_map),
        ]

        print("\n" + "=" * 148)
        print(f"MODEL: {args.model_id} | CONTEXT: ~{seq_len} TOKENS | PHASE 4 TODO VERIFICATION SUITE")
        print("=" * 148)
        print(
            f"{'Configuration':<52} | {'Hop Task':<17} | {'PPL':<6} | {'Target(79421)':<13} | "
            f"{'DecoyOmega':<10} | {'VaultRank':<9} | {'ZoneA':<7} | {'ZoneB':<7} | {'ZoneC'}"
        )
        print("-" * 148)

        results = {
            "model_id": args.model_id,
            "context_tokens": seq_len,
            "zone_a_span": [controller.zone_a_indices[0], controller.zone_a_indices[-1]],
            "zone_b_span": [controller.zone_b_indices[0], controller.zone_b_indices[-1]],
            "zone_c_span": [controller.zone_c_indices[0], controller.zone_c_indices[-1]],
            "relay1_heads": [list(k) for k in relay1_map.keys()],
            "quorum28_heads": [list(k) for k in quorum4_map.keys()],
            "evaluations": [],
        }

        for label, mode, beta, eta, sharpen, head_map in configs:
            controller.mode = mode
            controller.beta = beta
            controller.eta = eta
            controller.use_sharpening = sharpen
            controller.target_heads = head_map

            with torch.no_grad():
                f_out = model(**fluency_inputs, labels=fluency_labels, use_cache=False)
                ppl = math.exp(f_out.loss.item())
                del f_out

            for hop_name in ["1-Hop", "2-Hop", "3-Hop (Clean-6d)", "3-Hop (Heavy-16d)"]:
                t0 = time.time()
                controller.clear_zone_attn()
                inp = tokenized_prompts[hop_name]

                with torch.no_grad():
                    out = model(**inp, use_cache=False, logits_to_keep=1)
                    logits = out.logits[0, -1, :].float()
                    probs = torch.softmax(logits, dim=-1)
                    del out

                p_target = probs[target_tok].item() * 100.0
                p_omega = probs[omega_tok].item() * 100.0
                p_wrong = sum(probs[ct].item() for ct in competing_toks) * 100.0
                p_trap = probs[trap_tok].item() * 100.0

                # Compute rank of Target (79421) among the 8 Tier-0 Vault Codes (1 = Winner!)
                vault_probs = [(tok, probs[tok].item()) for tok in all_vault_toks]
                vault_probs.sort(key=lambda x: x[1], reverse=True)
                vault_rank = [idx + 1 for idx, (tok, _) in enumerate(vault_probs) if tok == target_tok][0]
                top1_tok_str = tokenizer.decode([vault_probs[0][0]]).strip()

                za = float(np.mean(list(controller.zone_a_attn.values()))) * 100.0
                zb = float(np.mean(list(controller.zone_b_attn.values()))) * 100.0
                zc = float(np.mean(list(controller.zone_c_attn.values()))) * 100.0
                dt = time.time() - t0

                rank_str = f"#{vault_rank} ({'WIN' if vault_rank == 1 else top1_tok_str})"
                print(
                    f"{label:<52} | {hop_name:<17} | {ppl:<6.2f} | {p_target:>11.2f}% | "
                    f"{p_omega:>8.2f}% | {rank_str:<9} | {za:>6.2f}% | {zb:>6.2f}% | {zc:>6.2f}% ({dt:.1f}s)",
                    flush=True,
                )

                results["evaluations"].append({
                    "configuration": label,
                    "hop_task": hop_name,
                    "fluency_ppl": round(ppl, 4),
                    "target_prob_pct": round(p_target, 4),
                    "decoy_omega_prob_pct": round(p_omega, 4),
                    "wrong_vault_sum_pct": round(p_wrong, 4),
                    "recency_trap_prob_pct": round(p_trap, 4),
                    "vault_rank": vault_rank,
                    "top1_vault_token": top1_tok_str,
                    "zone_a_attn_pct": round(za, 4),
                    "zone_b_attn_pct": round(zb, 4),
                    "zone_c_attn_pct": round(zc, 4),
                    "elapsed_sec": round(dt, 2),
                })

                with open(args.output_json, "w") as f:
                    json.dump(results, f, indent=2)
            print("-" * 148)

        log_heartbeat("STAGE2_PHASE4_SUITE", f"Saved full Phase 4 results to {args.output_json}")


if __name__ == "__main__":
    main()
