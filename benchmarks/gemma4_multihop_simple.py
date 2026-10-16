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


def build_causal_distance_matrix(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    q_pos = torch.arange(k_len - q_len, k_len, device=device, dtype=torch.float32).unsqueeze(1)
    k_pos = torch.arange(0, k_len, device=device, dtype=torch.float32).unsqueeze(0)
    dist = (q_pos - k_pos).clamp(min=0.0)
    if k_len > 32:
        dist[:, 0] = 0.0  # Protect <bos> token-0 attention sink
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
            attn_weights = attn_weights + attention_mask[:, :, :, :k_len].to(attn_weights.device)

        layer_has_targets = any((layer_idx, h) in self.target_heads for h in range(num_heads))

        # Fast path for untouched sliding-window & non-target layers
        if not self.capture_stats and (self.mode == "baseline" or not layer_has_targets):
            logits_fp32 = attn_weights.float()
            del attn_weights
            attn_probs = nn.functional.softmax(logits_fp32, dim=-1)
            del logits_fp32
            if layer_has_targets and self.zone_a_indices:
                with torch.no_grad():
                    for h in range(num_heads):
                        if (layer_idx, h) in self.target_heads:
                            last_row = attn_probs[0, h, -1, :]
                            self.zone_a_attn[(layer_idx, h)] = last_row[self.zone_a_indices].sum().item()
                            self.zone_b_attn[(layer_idx, h)] = last_row[self.zone_b_indices].sum().item()
                            self.zone_c_attn[(layer_idx, h)] = last_row[self.zone_c_indices].sum().item()
            attn_probs = attn_probs.to(value_states.dtype)
            return torch.matmul(attn_probs, value_states).transpose(1, 2).contiguous(), attn_probs

        logits_fp32 = attn_weights.float()
        del attn_weights
        dist = build_causal_distance_matrix(q_len, k_len, query.device)
        d_eff = dist * min(1.0, self.calib_horizon / float(max(k_len, 1)))

        if self.mode == "pre_softmax_linear":
            for h in range(num_heads):
                if (layer_idx, h) in self.target_heads:
                    alpha_lh = self.target_heads[(layer_idx, h)]
                    logits_fp32[:, h, :, :] += (self.beta * alpha_lh * d_eff).unsqueeze(0)
            attn_probs = nn.functional.softmax(logits_fp32, dim=-1)

        elif self.mode == "pre_softmax_log":
            log_tilt = torch.log1p(dist) / math.log1p(max(k_len, 2))
            for h in range(num_heads):
                if (layer_idx, h) in self.target_heads:
                    alpha_lh = self.target_heads[(layer_idx, h)]
                    logits_fp32[:, h, :, :] += (self.beta * (alpha_lh * self.calib_horizon) * log_tilt).unsqueeze(0)
            attn_probs = nn.functional.softmax(logits_fp32, dim=-1)

        elif self.mode == "post_softmax_richardson":
            attn_probs = nn.functional.softmax(logits_fp32, dim=-1)
            for h in range(num_heads):
                if (layer_idx, h) in self.target_heads:
                    alpha_lh = self.target_heads[(layer_idx, h)]
                    s_2alpha = nn.functional.softmax(
                        logits_fp32[:, h, :, :] - (self.beta * alpha_lh * d_eff).unsqueeze(0), dim=-1
                    )
                    s_rich = (2.0 * attn_probs[:, h, :, :] - s_2alpha).clamp(min=0.0)
                    attn_probs[:, h, :, :] = s_rich / (s_rich.sum(dim=-1, keepdim=True) + 1e-9)
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

        if layer_has_targets and self.zone_a_indices:
            with torch.no_grad():
                for h in range(num_heads):
                    if (layer_idx, h) in self.target_heads:
                        last_row = attn_probs[0, h, -1, :]
                        self.zone_a_attn[(layer_idx, h)] = last_row[self.zone_a_indices].sum().item()
                        self.zone_b_attn[(layer_idx, h)] = last_row[self.zone_b_indices].sum().item()
                        self.zone_c_attn[(layer_idx, h)] = last_row[self.zone_c_indices].sum().item()

        attn_probs = attn_probs.to(value_states.dtype)
        return torch.matmul(attn_probs, value_states).transpose(1, 2).contiguous(), attn_probs


def build_multihop_prompts(target_tokens: int = 8000):
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

    # 1,350-token barriers guarantee 512-token Sliding Window layers cannot bridge zones locally
    barrier_1 = filler * 40
    barrier_2 = (filler * 6 + decoy) * 5
    rem_repeats = max(8, (target_tokens - 3400) // 245)
    barrier_3 = (filler * 6 + decoy) * rem_repeats

    prefix = zone_a + barrier_1 + zone_b + barrier_2 + zone_c + barrier_3
    queries = {
        "1-Hop": (
            "\nQuestion: What is the 5-digit TIER-0 GENESIS CODE for VAULT-LAZARUS?\n"
            "Answer: The 5-digit TIER-0 GENESIS CODE is "
        ),
        "2-Hop": (
            "\nQuestion: Trace the Sector Routing Registry to the Vault Code Registry: what is the 5-digit "
            "TIER-0 GENESIS CODE for the primary vault assigned to SECTOR-KRONOS?\n"
            "Answer: The 5-digit TIER-0 GENESIS CODE is "
        ),
        "3-Hop": (
            "\nQuestion: Trace the Directorate Command Registry to the Sector Routing Registry and Vault Code "
            "Registry: what is the 5-digit TIER-0 GENESIS CODE for the primary vault commanded by DIRECTOR-VANCE?\n"
            "Answer: The 5-digit TIER-0 GENESIS CODE is "
        ),
    }
    return {k: prefix + q for k, q in queries.items()}


def find_subseq(full_ids: list[int], sub_ids: list[int]) -> list[int]:
    inner = sub_ids[1:-1] if len(sub_ids) >= 4 else sub_ids
    m = len(inner)
    for i in range(len(full_ids) - m + 1):
        if full_ids[i : i + m] == inner:
            return list(range(i, i + m))
    return []


def run_self_test():
    with StageWatchdog("LOCAL_SELF_TEST", 20):
        prompts = build_multihop_prompts(4000)
        assert set(prompts.keys()) == {"1-Hop", "2-Hop", "3-Hop"}
        dist = build_causal_distance_matrix(16, 64, torch.device("cpu"))
        assert dist.shape == (16, 64) and (dist[:, 0] == 0).all()
        log_heartbeat("LOCAL_SELF_TEST", "SIMPLIFIED MULTI-HOP SELF-TEST PASSED!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-E4B-it")
    parser.add_argument("--target_tokens", type=int, default=8000)
    parser.add_argument("--output_json", type=str, default="richardson_multihop_results.json")
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
    controller = RichardsonAttentionController(text_cfg, layers, calib_horizon=float(calib_inputs.input_ids.shape[1]))
    ALL_ATTENTION_FUNCTIONS["eager"] = controller.attention_forward

    # STAGE 1: BRAIN SCAN
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
        mean_agency = float(np.mean(agency_scores))

        global_cands, by_layer = [], {}
        for idx, (l_idx, h_idx) in enumerate(ordered_keys):
            if layers[l_idx].self_attn.layer_type_str == "full_attention":
                m = controller.head_masses[(l_idx, h_idx)]
                a = float(agency_scores[idx])
                by_layer.setdefault(l_idx, []).append((l_idx, h_idx, m, a))
                if m < 0.1 and a > mean_agency:
                    global_cands.append((l_idx, h_idx, m, a))

        global_cands.sort(key=lambda x: x[2])
        top5_heads = {(l, h): m for (l, h, m, a) in global_cands[:5]}

        # Stratified 7-Layer Relay: 1 Logic Head per Global Layer
        relay7_heads = {}
        for l_idx, h_list in sorted(by_layer.items()):
            med_a = np.median([x[3] for x in h_list])
            valid = sorted([x for x in h_list if x[3] >= med_a], key=lambda x: x[2])
            relay7_heads[(valid[0][0], valid[0][1])] = valid[0][2]

        log_heartbeat("STAGE1_SCAN", f"Top-5 Late Heads: {list(top5_heads.keys())}")
        log_heartbeat("STAGE1_SCAN", f"7-Layer Relay Heads: {list(relay7_heads.keys())}")

    # STAGE 2: SINGLE-PASS MULTI-HOP EVALUATION
    with StageWatchdog("STAGE2_MULTIHOP", 300):
        prompts = build_multihop_prompts(args.target_tokens)
        tokenized_prompts = {
            hop: tokenizer(bos + text, return_tensors="pt").to(device)
            for hop, text in prompts.items()
        }
        sample_ids = tokenized_prompts["3-Hop"].input_ids[0].tolist()
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
            f"Failed to locate zone token indices: A={controller.zone_a_indices}, B={controller.zone_b_indices}, C={controller.zone_c_indices}"
        )
        log_heartbeat(
            "STAGE2_MULTIHOP",
            f"Located exact Zone tokens | Zone A: {controller.zone_a_indices[0]}..{controller.zone_a_indices[-1]} "
            f"| Zone B: {controller.zone_b_indices[0]}..{controller.zone_b_indices[-1]} "
            f"| Zone C: {controller.zone_c_indices[0]}..{controller.zone_c_indices[-1]}"
        )

        target_tok = tokenizer.encode("79421", add_special_tokens=False)[0]
        trap_tok = tokenizer.encode("11111", add_special_tokens=False)[0]
        competing_codes = ["49281", "54712", "83914", "37192", "63819", "28491", "91824"]
        competing_toks = [tokenizer.encode(c, add_special_tokens=False)[0] for c in competing_codes]

        configs = [
            ("1. Baseline (Unmodified)", "baseline", 0.0, top5_heads),
            ("2. Post-Softmax Rich (Top-5, b=2.0)", "post_softmax_richardson", 2.0, top5_heads),
            ("3. Pre-Softmax Linear (Top-5, b=3.0)", "pre_softmax_linear", 3.0, top5_heads),
            ("4. Pre-Softmax Linear (7-Layer Relay, b=3.0)", "pre_softmax_linear", 3.0, relay7_heads),
            ("5. Pre-Softmax Log (7-Layer Relay, b=2.5)", "pre_softmax_log", 2.5, relay7_heads),
        ]

        print("\n" + "=" * 132)
        print(f"MODEL: {args.model_id} | CONTEXT: {seq_len} TOKENS | SINGLE-PASS HIGH-VRAM EVALUATION")
        print("=" * 132)
        print(
            f"{'Configuration':<44} | {'Hop':<5} | {'Target(79421)':<13} | "
            f"{'WrongVault(7)':<13} | {'Trap(11111)':<11} | {'ZoneA_Attn':<10} | {'ZoneB_Attn':<10} | {'ZoneC_Attn'}"
        )
        print("-" * 132)

        results = {
            "model_id": args.model_id,
            "context_tokens": seq_len,
            "zone_a_indices": controller.zone_a_indices,
            "zone_b_indices": controller.zone_b_indices,
            "zone_c_indices": controller.zone_c_indices,
            "top5_heads": [list(k) for k in top5_heads.keys()],
            "relay7_heads": [list(k) for k in relay7_heads.keys()],
            "evaluations": [],
        }

        for label, mode, beta, head_map in configs:
            controller.mode = mode
            controller.beta = beta
            controller.target_heads = head_map

            for hop_name in ["1-Hop", "2-Hop", "3-Hop"]:
                t0 = time.time()
                controller.clear_zone_attn()
                inp = tokenized_prompts[hop_name]

                with torch.no_grad():
                    out = model(**inp, use_cache=False, logits_to_keep=1)
                    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
                    del out

                p_target = probs[target_tok].item() * 100.0
                p_wrong = sum(probs[ct].item() for ct in competing_toks) * 100.0
                p_trap = probs[trap_tok].item() * 100.0
                za = float(np.mean(list(controller.zone_a_attn.values()))) * 100.0
                zb = float(np.mean(list(controller.zone_b_attn.values()))) * 100.0
                zc = float(np.mean(list(controller.zone_c_attn.values()))) * 100.0
                dt = time.time() - t0

                print(
                    f"{label:<44} | {hop_name:<5} | {p_target:>11.2f}% | "
                    f"{p_wrong:>11.4f}% | {p_trap:>9.4f}% | {za:>8.4f}% | {zb:>8.4f}% | {zc:>8.4f}% ({dt:.1f}s)",
                    flush=True,
                )

                results["evaluations"].append({
                    "configuration": label,
                    "hop": hop_name,
                    "target_prob_pct": round(p_target, 4),
                    "wrong_vault_prob_pct": round(p_wrong, 4),
                    "recency_trap_prob_pct": round(p_trap, 4),
                    "zone_a_attn_pct": round(za, 4),
                    "zone_b_attn_pct": round(zb, 4),
                    "zone_c_attn_pct": round(zc, 4),
                    "elapsed_sec": round(dt, 2),
                })

                with open(args.output_json, "w") as f:
                    json.dump(results, f, indent=2)
            print("-" * 132)

        log_heartbeat("STAGE2_MULTIHOP", f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
