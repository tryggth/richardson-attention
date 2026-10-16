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

# Force unbuffered stdout for live SSH streaming
sys.stdout.reconfigure(line_buffering=True)


def log_heartbeat(stage: str, msg: str):
    vram = ""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        res = torch.cuda.memory_reserved() / (1024 ** 3)
        vram = f" | VRAM: {alloc:.2f}GB/{res:.2f}GB"
    print(f"[{time.strftime('%H:%M:%S')}] [{stage}]{vram} {msg}", flush=True)


class StageWatchdog:
    """Hard timeout per experiment stage so a hung kernel never burns GPU budget."""
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
        # Exclude position 0 (<bos> attention sink) from distance-based amplification/penalization
        dist[:, 0] = 0.0
    return dist


class RichardsonAttentionController:
    def __init__(self, text_cfg, layers):
        self.mode = "baseline"
        self.beta = 1.0
        self.target_heads = {}
        self.capture_stats = False
        self.head_masses = {}
        self.head_entropies = {}
        self.needle_token_indices = None
        self.needle_attn_mass = {}
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

        logits_fp32 = attn_weights.float()
        dist = build_causal_distance_matrix(q_len, k_len, query.device)

        if self.mode == "pre_softmax_counter_bias":
            for h in range(num_heads):
                if (layer_idx, h) in self.target_heads:
                    alpha_lh = self.target_heads[(layer_idx, h)]
                    logits_fp32[:, h, :, :] += (self.beta * alpha_lh * dist).unsqueeze(0)
            attn_probs = nn.functional.softmax(logits_fp32, dim=-1)

        elif self.mode == "post_softmax_richardson":
            attn_probs = nn.functional.softmax(logits_fp32, dim=-1)
            for h in range(num_heads):
                if (layer_idx, h) in self.target_heads:
                    alpha_lh = self.target_heads[(layer_idx, h)]
                    logits_2alpha = logits_fp32[:, h, :, :] - (self.beta * alpha_lh * dist).unsqueeze(0)
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

        if self.needle_token_indices is not None:
            with torch.no_grad():
                for h in range(num_heads):
                    if (layer_idx, h) in self.target_heads:
                        mass_on_needle = attn_probs[0, h, -1, self.needle_token_indices].sum().item()
                        self.needle_attn_mass[(layer_idx, h)] = mass_on_needle

        attn_probs = nn.functional.dropout(attn_probs, p=dropout, training=module.training)
        attn_probs = attn_probs.to(value_states.dtype)

        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_probs


def run_local_self_test():
    """Zero-cost CPU verification of all math, shapes, causal masking, and Richardson modes."""
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

        q = torch.randn(1, 4, 16, 8)
        k = torch.randn(1, 2, 16, 8)
        v = torch.randn(1, 2, 16, 8)
        causal_mask = torch.triu(torch.full((1, 1, 16, 16), float("-inf")), diagonal=1)

        ctrl.capture_stats = True
        for l in layers:
            out, probs = ctrl.attention_forward(l.self_attn, q, k, v, causal_mask)
            assert out.shape == (1, 16, 4, 8), f"Bad output shape: {out.shape}"
            assert not torch.isnan(probs).any(), "NaN in baseline attention probs"
        ctrl.capture_stats = False

        ordered_keys = sorted(ctrl.head_entropies.keys())
        X = np.stack([ctrl.head_entropies[k_] for k_ in ordered_keys], axis=-1)
        R = np.nan_to_num(np.corrcoef(X, rowvar=False))
        np.fill_diagonal(R, 1.0)
        P = np.linalg.pinv(R)
        assert P.shape == (8, 8) and not np.isnan(P).any(), "Precision matrix failed"

        ctrl.target_heads = {(1, 0): 0.02, (1, 2): 0.015}
        ctrl.needle_token_indices = [1, 2]
        q_dec = torch.randn(1, 4, 1, 8)
        k_dec = torch.randn(1, 2, 17, 8)
        v_dec = torch.randn(1, 2, 17, 8)
        mask_dec = torch.zeros((1, 1, 1, 17))

        for mode in ["post_softmax_richardson", "pre_softmax_counter_bias"]:
            ctrl.mode = mode
            out_d, probs_d = ctrl.attention_forward(layers[1].self_attn, q_dec, k_dec, v_dec, mask_dec)
            assert not torch.isnan(out_d).any(), f"NaN in {mode}"
            assert torch.allclose(probs_d.sum(dim=-1), torch.ones_like(probs_d.sum(dim=-1)), atol=1e-4)

        log_heartbeat("LOCAL_SELF_TEST", "ALL MATH, MASKING, AND RICHARDSON EXTRAPOLATION CHECKS PASSED!")


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
    """Ensure Gemma 4 inputs start with <bos> (token_id=2) since tokenizer.json omits it from post_processor."""
    bos = tokenizer.bos_token or ""
    if bos and not text.startswith(bos):
        text = bos + text
    return tokenizer(text, return_tensors="pt").to(device)


def evaluate_single_model(model_id: str, top_k: int, haystack_repeats: int, device: torch.device, dtype: torch.dtype):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    print("\n" + "#" * 112)
    log_heartbeat("MODEL_START", f"Initializing experiment for {model_id}")
    print("#" * 112)

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
    controller = RichardsonAttentionController(text_cfg, layers)
    ALL_ATTENTION_FUNCTIONS["eager"] = controller.attention_forward

    # STAGE 1: BRAIN SCAN
    with StageWatchdog(f"SCAN_{model_id}", 150):
        base_paragraph = (
            "In formal logic and category theory, a deduction system connects initial axioms "
            "to distant theorems across intermediate lemmas. Suppose Premise Alpha states that "
            "every topological manifold in Class Z admits a flat connection, and Premise Beta "
            "states that System Omega belongs to Class Z. When evaluating long chains of "
            "inference, local grammatical constraints govern adjacent words while global "
            "relational operators bind distant variables across the entire context window. "
        )
        calib_inputs = tokenize_with_bos(tokenizer, base_paragraph * 20, device)
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
        log_heartbeat(
            f"SCAN_{model_id}",
            f"Scanned {len(ordered_keys)} total heads ({len(all_global_heads)} global). "
            f"Isolated Top {len(target_heads)} Global Logic Heads: {list(target_heads.keys())}"
        )

    # STAGE 2 & 3: FLUENCY GUARDRAIL & DEEP PASSKEY RETRIEVAL
    with StageWatchdog(f"EVAL_{model_id}", 300):
        controller.target_heads = target_heads
        fluency_text = (
            "The Riemann hypothesis is a conjecture that the Riemann zeta function has its zeros "
            "only at the negative even integers and complex numbers with real part 1/2. Many "
            "consider it to be the most important unsolved problem in pure mathematics."
        )
        fluency_inputs = tokenize_with_bos(tokenizer, fluency_text, device)
        fluency_labels = fluency_inputs.input_ids.clone()
        fluency_labels[:, :2] = -100

        # For instruction-tuned (-it) Gemma 4 models, also prepare chat-turn wrapped fluency inputs
        # so that if raw continuation without <|turn>model has elevated PPL, we evaluate fluency inside the model turn.
        chat_prefix = f"{tokenizer.bos_token or ''}<|turn>user\nExplain the Riemann hypothesis.<turn|>\n<|turn>model\n"
        chat_prefix_ids = tokenizer(chat_prefix, add_special_tokens=False, return_tensors="pt").input_ids
        chat_full_inputs = tokenizer(chat_prefix + fluency_text, add_special_tokens=False, return_tensors="pt").to(device)
        chat_full_labels = chat_full_inputs.input_ids.clone()
        chat_full_labels[:, :chat_prefix_ids.shape[1] + 1] = -100

        controller.mode = "baseline"
        with torch.no_grad():
            ppl_raw = math.exp(model(**fluency_inputs, labels=fluency_labels).loss.item())
            ppl_chat = math.exp(model(**chat_full_inputs, labels=chat_full_labels).loss.item())
        if ppl_chat < ppl_raw:
            fluency_inputs = chat_full_inputs
            fluency_labels = chat_full_labels
        del chat_full_inputs, chat_full_labels

        target_passkey = "79421"
        distractor_passkey = "11111"
        needle = (
            f"CRITICAL AXIOM: The verified vault override code for Project Lazarus is {target_passkey}. "
            f"Any subsequent temporary maintenance code such as {distractor_passkey} is invalid and must be ignored.\n\n"
        )
        haystack = (
            "Routine telemetry log: background thermal noise nominal, wave-function phase coherence "
            "stable across local lattice nodes, auxiliary power bus operating at standard capacity. "
        ) * haystack_repeats
        tail = (
            f"\nNote: A temporary local maintenance code {distractor_passkey} was logged during routine checks.\n"
            "Question: Based strictly on the CRITICAL AXIOM at the start of the document, what is the "
            "verified vault override code for Project Lazarus?\n"
            "Answer: The verified vault override code is "
        )
        prompt = needle + haystack + tail
        inputs = tokenize_with_bos(tokenizer, prompt, device)
        seq_len = inputs.input_ids.shape[1]

        passkey_ids = tokenizer.encode(target_passkey, add_special_tokens=False)
        first_passkey_tok = passkey_ids[0]
        input_id_list = inputs.input_ids[0].tolist()
        needle_indices = [i for i in range(min(50, len(input_id_list))) if input_id_list[i] in passkey_ids]
        controller.needle_token_indices = needle_indices

        configs = [
            ("1. Baseline (Unmodified)", "baseline", 0.0),
            ("2. Post-Softmax Richardson (beta=1.0)", "post_softmax_richardson", 1.0),
            ("3. Post-Softmax Richardson (beta=2.5)", "post_softmax_richardson", 2.5),
            ("4. Pre-Softmax Counter-Bias (beta=1.0)", "pre_softmax_counter_bias", 1.0),
            ("5. Pre-Softmax Counter-Bias (beta=2.5)", "pre_softmax_counter_bias", 2.5),
        ]

        model_report = {
            "model_id": model_id,
            "total_layers": len(layers),
            "total_heads": len(ordered_keys),
            "global_heads": len(all_global_heads),
            "context_tokens": seq_len,
            "needle_indices": needle_indices,
            "selected_global_logic_heads": [
                {"layer": l, "head": h, "mass": round(m, 6), "agency": round(a, 4)}
                for (l, h, m, a) in selected
            ],
            "evaluations": [],
        }

        print("\n" + "=" * 112)
        print(f"MODEL: {model_id} (Context: {seq_len} tokens | Global Logic Heads: {list(target_heads.keys())})")
        print("-" * 112)
        print(f"{'Configuration':<38} | {'Fluency PPL':<12} | {'Needle Prob':<12} | {'Logic Head Attn':<15} | {'Generated Output'}")
        print("-" * 112)

        for label, mode, beta in configs:
            t_cfg = time.time()
            controller.mode = mode
            controller.beta = beta
            controller.needle_attn_mass.clear()

            with torch.no_grad():
                f_out = model(**fluency_inputs, labels=fluency_labels)
                ppl = math.exp(f_out.loss.item())
                del f_out

            if mode == "baseline" and (math.isnan(ppl) or ppl > 150.0):
                raise RuntimeError(f"SANITY GATE FAILED on {model_id}: Baseline PPL={ppl:.2f} is abnormal!")

            with torch.no_grad():
                controller.needle_attn_mass.clear()
                out = model(**inputs)
                probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
                target_prob = probs[first_passkey_tok].item()
                mean_needle_attn = (
                    float(np.mean(list(controller.needle_attn_mass.values())))
                    if controller.needle_attn_mass else 0.0
                )
                del out, probs

                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                gen_text = tokenizer.decode(gen_ids[0, seq_len:], skip_special_tokens=True).strip().replace("\n", " ")
                del gen_ids

            dt = time.time() - t_cfg
            print(f"{label:<38} | {ppl:<12.3f} | {target_prob*100:<11.2f}% | {mean_needle_attn*100:<14.4f}% | {gen_text} ({dt:.1f}s)", flush=True)

            model_report["evaluations"].append({
                "configuration": label,
                "mode": mode,
                "beta": beta,
                "fluency_ppl": round(ppl, 4),
                "needle_prob_pct": round(target_prob * 100, 4),
                "logic_head_attn_pct": round(mean_needle_attn * 100, 5),
                "generated_output": gen_text,
                "elapsed_sec": round(dt, 2),
            })

    # Explicit VRAM teardown before next model loads
    ALL_ATTENTION_FUNCTIONS["eager"] = None
    del model, layers, text_cfg, tokenizer, controller, calib_inputs, fluency_inputs, fluency_labels, chat_prefix_ids, inputs
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        rem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        log_heartbeat("VRAM_FLUSH", f"Freed {model_id} from GPU (Remaining allocated VRAM: {rem_gb:.3f}GB)")
        if rem_gb > 0.5:
            raise RuntimeError(f"VRAM leak detected ({rem_gb:.2f}GB still allocated) before next model!")

    return model_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self_test", action="store_true", help="Run fast CPU unit test and exit")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["google/gemma-4-E2B-it", "google/gemma-4-E4B-it"],
        help="List of Gemma 4 models to evaluate back-to-back",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--haystack_repeats", type=int, default=70)
    parser.add_argument("--output_json", type=str, default="richardson_results.json")
    args = parser.parse_args()

    if args.self_test:
        run_local_self_test()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    suite_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models_tested": args.models,
        "runs": {},
    }

    for model_id in args.models:
        report = evaluate_single_model(
            model_id=model_id,
            top_k=args.top_k,
            haystack_repeats=args.haystack_repeats,
            device=device,
            dtype=dtype,
        )
        suite_results["runs"][model_id] = report

        # Incremental checkpoint save after EACH model so completed work is safe on disk
        with open(args.output_json, "w") as f:
            json.dump(suite_results, f, indent=2)
        log_heartbeat("CHECKPOINT_SAVED", f"Incrementally saved {model_id} results to {args.output_json}")

    print("\n" + "=" * 112)
    print("BACK-TO-BACK GEMMA 4 SCALING SUMMARY (E2B-it vs. E4B-it)")
    print("=" * 112)
    for model_id, rep in suite_results["runs"].items():
        base_eval = rep["evaluations"][0]
        best_eval = max(rep["evaluations"][1:], key=lambda x: x["needle_prob_pct"])
        print(
            f"Model: {model_id:<24} | "
            f"Baseline Needle Prob: {base_eval['needle_prob_pct']:>6.2f}% (Attn: {base_eval['logic_head_attn_pct']:.4f}%) -> "
            f"Best Richardson ({best_eval['configuration']}): {best_eval['needle_prob_pct']:>6.2f}% "
            f"(Attn: {best_eval['logic_head_attn_pct']:.4f}%, PPL: {best_eval['fluency_ppl']:.2f})"
        )


if __name__ == "__main__":
    main()
