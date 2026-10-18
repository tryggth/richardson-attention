import argparse
import json
import math
import random
import signal
import sys
import time
import torch
import torch.nn as nn
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from gemma4_phase6_vertical import build_5hop_prompts, find_subseq
from gemma4_phase7_standard_benchmarks import generate_babilong_qa3_instance, generate_ruler_vt_instance


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
        dist[:, 0] = 0.0
    return dist, q_pos


def build_quorum_head_map(by_layer: dict, heads_per_layer: int, calib_horizon: float) -> dict:
    quorum_map = {}
    for l_idx, h_list in sorted(by_layer.items()):
        ranked = sorted(h_list, key=lambda x: (x[2], -x[3]))
        for (l, h, m, a) in ranked[:heads_per_layer]:
            raw_kappa = m * calib_horizon
            quorum_map[(l, h)] = float(np.clip(raw_kappa, 1.8, 2.6))
    return quorum_map


class Phase8FiberBundleController:
    """
    Implements Gauge-Covariant Vertical Extrapolation:
    1. Orthogonal Procrustes Parallel Transport (Omega_{k -> k+1}^{(r)}) + Unit-Norm Matching across SWA gaps.
    2. Vocabulary-Fiber Gauge Pullback Extrapolation across Global Layers {L23, L29, L35, L41}.
    """
    def __init__(self, model, text_cfg, layers, calib_horizon: float = 1400.0, q_chunk_size: int = 2048):
        self.model = model
        self.layers = layers
        self.mode = "baseline"
        self.beta = 3.5
        self.eta = 0.75
        self.swa_shield_beta = 0.0
        self.query_slice_only = True
        self.query_window_len = 64
        self.plateau_power = 3.0
        self.calib_horizon = calib_horizon
        self.q_chunk_size = q_chunk_size

        # Procrustes Parallel Transport parameters
        self.procrustes_gamma = 0.0        # Weight of Procrustes-transported update (e.g. 0.12)
        self.procrustes_rank = 128         # Low-rank principal bundle section rank r
        self.procrustes_matrices = {}      # Dict[(l_prev, l_next) -> Tensor[d, d] on CPU/GPU]
        self.calib_u_norms = {}            # Dict[l_idx -> float mean update norm]
        self.calib_layer_out = {}          # Captured during Stage 1 calibration
        self.calib_layer_in = {}
        self.calib_attn_u = {}
        self.fiber_diagnostics = {}        # Pre vs Post Procrustes cosine & singular spectrum

        # Vocabulary-Fiber Pullback parameters
        self.vocab_pullback_mode = "none"  # "none", "unsoftcapped_l41", "dual_35_41_minus_23"
        self.vocab_gamma = 0.35
        self.layer_lens_probs = {}         # Dict[int -> Tensor[V] unsoftcapped probability vector]

        self.target_heads = {}
        self.layer_horizons = {}
        self.capture_stats = False
        self.head_masses = {}
        self.head_entropies = {}

        self.prev_global_u = None
        self.prev_global_idx = None
        self.global_attn_updates = {}

        self.global_layer_indices = []
        layer_types = getattr(text_cfg, "layer_types", None)
        for idx, layer in enumerate(layers):
            attn_mod = layer.self_attn
            attn_mod.layer_idx = idx
            if layer_types is not None and idx < len(layer_types):
                attn_mod.layer_type_str = layer_types[idx]
            else:
                is_sliding = getattr(attn_mod, "is_sliding", False)
                attn_mod.layer_type_str = "sliding_attention" if is_sliding else "full_attention"
            if attn_mod.layer_type_str == "full_attention":
                self.global_layer_indices.append(idx)

        self._hooks = []
        self._register_fiber_hooks()

    def _get_final_norm_and_lm_head(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            norm = self.model.model.language_model.norm
        elif hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            norm = self.model.model.norm
        else:
            norm = None
        lm_head = getattr(self.model, "lm_head", None)
        return norm, lm_head

    def _register_fiber_hooks(self):
        for idx in self.global_layer_indices:
            layer = self.layers[idx]

            if hasattr(layer, "post_attention_layernorm"):
                def make_post_attn_hook(l_idx):
                    def hook(mod, inputs, output):
                        if self.capture_stats:
                            self.calib_attn_u[l_idx] = output[0].detach().float().cpu()
                        elif output.shape[1] > 256:
                            w = min(self.query_window_len, output.shape[1])
                            self.global_attn_updates[l_idx] = output[:, -w:, :].detach()
                    return hook
                self._hooks.append(layer.post_attention_layernorm.register_forward_hook(make_post_attn_hook(idx)))

            def make_pre_hook(l_idx):
                def pre_hook(mod, args):
                    hidden_states = args[0]
                    seq_len = hidden_states.shape[1]
                    if self.capture_stats:
                        self.calib_layer_in[l_idx] = hidden_states[0].detach().float().cpu()
                        return None
                    if seq_len <= 256:
                        return None

                    if l_idx == self.global_layer_indices[0]:
                        self.prev_global_u = None
                        self.prev_global_idx = None

                    # Apply Gauge-Covariant Procrustes Parallel Transport across the 5-layer SWA gap
                    if (
                        self.procrustes_gamma > 0.0
                        and self.prev_global_u is not None
                        and (self.prev_global_idx, l_idx) in self.procrustes_matrices
                    ):
                        w = min(self.query_window_len, seq_len)
                        cur_slice = hidden_states[:, -w:, :]
                        dev = cur_slice.device
                        omega = self.procrustes_matrices[(self.prev_global_idx, l_idx)].to(dev)
                        u_prev = self.prev_global_u.to(dev).float()  # [1, w, d]

                        # 1. Parallel transport u_prev from l_prev's coordinate chart into l_idx's chart
                        u_transported = torch.matmul(u_prev, omega)

                        # 2. Unit-norm match to target layer's native scale (eliminates L23 292.16 spike!)
                        u_unit = u_transported / (u_transported.norm(dim=-1, keepdim=True) + 1e-6)
                        target_scale = self.calib_u_norms.get(l_idx, 50.0)
                        u_scaled = u_unit * target_scale

                        orig_rms = cur_slice.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
                        boosted = cur_slice.float() + self.procrustes_gamma * u_scaled
                        new_rms = boosted.pow(2).mean(dim=-1, keepdim=True).sqrt()
                        boosted = (boosted * (orig_rms / (new_rms + 1e-6))).to(hidden_states.dtype)

                        new_hidden = hidden_states.clone()
                        new_hidden[:, -w:, :] = boosted
                        return (new_hidden,) + args[1:]
                    return None
                return pre_hook

            def make_post_hook(l_idx):
                def post_hook(mod, args, output):
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    seq_len = hidden_states.shape[1]
                    if self.capture_stats:
                        self.calib_layer_out[l_idx] = hidden_states[0].detach().float().cpu()
                        return None
                    if seq_len <= 256:
                        return None

                    u_attn = self.global_attn_updates.get(l_idx, None)
                    if u_attn is not None:
                        self.prev_global_u = u_attn
                        self.prev_global_idx = l_idx

                    # Capture unsoftcapped Vocabulary-Fiber Pullback probabilities at L23, L29, L35, L41
                    if l_idx in (23, 29, 35, 41):
                        norm_mod, lm_head = self._get_final_norm_and_lm_head()
                        if norm_mod is not None and lm_head is not None:
                            with torch.no_grad():
                                last_h = hidden_states[:, -1:, :]
                                norm_dev = next(norm_mod.parameters()).device
                                lm_dev = next(lm_head.parameters()).device
                                normed_h = norm_mod(last_h.to(norm_dev))
                                raw_logits = lm_head(normed_h.to(lm_dev))[0, -1, :].float()
                                self.layer_lens_probs[l_idx] = torch.softmax(raw_logits, dim=-1).detach()
                    return None
                return post_hook

            self._hooks.append(layer.register_forward_pre_hook(make_pre_hook(idx)))
            self._hooks.append(layer.register_forward_hook(make_post_hook(idx)))

    def compute_procrustes_transport_operators(self):
        """
        Computes the rank-r Orthogonal Procrustes parallel transport matrices Omega_{k -> k+1}^{(r)}
        between consecutive Global layers from Stage 1 calibration states, and measures alignment gain.
        """
        for i in range(len(self.global_layer_indices)):
            l_cur = self.global_layer_indices[i]
            if l_cur in self.calib_attn_u:
                self.calib_u_norms[l_cur] = float(self.calib_attn_u[l_cur].norm(dim=-1).mean().item())

        for i in range(len(self.global_layer_indices) - 1):
            l_prev = self.global_layer_indices[i]
            l_next = self.global_layer_indices[i + 1]
            U_prev = self.calib_attn_u[l_prev]      # [T_cal, d]
            X_next = self.calib_layer_in[l_next]    # [T_cal, d]

            # Normalize rows before Procrustes SVD so high-norm tokens don't skew the basis
            U_norm = U_prev / (U_prev.norm(dim=-1, keepdim=True) + 1e-6)
            X_norm = X_next / (X_next.norm(dim=-1, keepdim=True) + 1e-6)

            raw_cos = float(nn.functional.cosine_similarity(U_norm, X_norm, dim=-1).mean().item())

            cov = torch.matmul(U_norm.T, X_norm)    # [d, d]
            U_svd, S_svd, Vh_svd = torch.linalg.svd(cov, full_matrices=False)

            r = min(self.procrustes_rank, U_svd.shape[1])
            omega_r = torch.matmul(U_svd[:, :r], Vh_svd[:r, :])  # Rank-r principal bundle transport
            omega_full = torch.matmul(U_svd, Vh_svd)

            transported_r = torch.matmul(U_norm, omega_r)
            transported_full = torch.matmul(U_norm, omega_full)
            aligned_cos_r = float(nn.functional.cosine_similarity(transported_r, X_norm, dim=-1).mean().item())
            aligned_cos_full = float(nn.functional.cosine_similarity(transported_full, X_norm, dim=-1).mean().item())

            top16_energy = float((S_svd[:16].sum() / (S_svd.sum() + 1e-6)).item() * 100.0)
            top128_energy = float((S_svd[:r].sum() / (S_svd.sum() + 1e-6)).item() * 100.0)

            self.procrustes_matrices[(l_prev, l_next)] = omega_r
            gap_key = f"L{l_prev}->L{l_next}"
            self.fiber_diagnostics[gap_key] = {
                "raw_unrotated_cosine": round(raw_cos, 4),
                "procrustes_rank128_cosine": round(aligned_cos_r, 4),
                "procrustes_full_cosine": round(aligned_cos_full, 4),
                "top16_singular_energy_pct": round(top16_energy, 2),
                "top128_singular_energy_pct": round(top128_energy, 2),
                "mean_calib_u_norm": round(self.calib_u_norms.get(l_prev, 0.0), 2),
            }

        self.calib_layer_in.clear()
        self.calib_layer_out.clear()
        self.calib_attn_u.clear()

    def evaluate_output_probabilities(self, final_model_logits: torch.Tensor) -> torch.Tensor:
        """
        Computes final probability distribution over the vocabulary, optionally applying
        Vocabulary-Fiber Gauge Pullback across {L23, L29, L35, L41}.
        """
        p_final = torch.softmax(final_model_logits.float(), dim=-1)
        if self.vocab_pullback_mode == "none":
            return p_final

        p_l41 = self.layer_lens_probs.get(41, p_final)
        p_l35 = self.layer_lens_probs.get(35, p_final)
        p_l29 = self.layer_lens_probs.get(29, p_final)
        p_l23 = self.layer_lens_probs.get(23, p_final)

        if self.vocab_pullback_mode == "unsoftcapped_l41":
            # Bypasses Gemma 4's tanh(logits/30)*30 compression ceiling by blending unsoftcapped L41 + L35 pullback
            p_blend = 0.70 * p_l41 + 0.30 * p_l35
            return p_blend / (p_blend.sum() + 1e-9)

        elif self.vocab_pullback_mode == "dual_35_41_minus_23":
            # Dual-State Vertical Richardson in the flat Vocabulary Fiber:
            # Late resolved state (0.65 * P_41 + 0.35 * P_35) minus early unresolved decoy state (P_23)
            p_late = 0.65 * p_l41 + 0.35 * p_l35
            p_early = 0.50 * p_l23 + 0.50 * p_l29
            p_vert = ((1.0 + self.vocab_gamma) * p_late - self.vocab_gamma * p_early).clamp(min=0.0)
            return p_vert / (p_vert.sum() + 1e-9)

        return p_final

    def clear_step_state(self):
        self.layer_lens_probs.clear()
        self.global_attn_updates.clear()
        self.prev_global_u = None
        self.prev_global_idx = None

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

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
        layer_type = getattr(module, "layer_type_str", "full_attention")
        layer_has_targets = any((layer_idx, h) in self.target_heads for h in range(num_heads))

        attn_output = torch.empty(
            (bsz, num_heads, q_len, value_states.shape[-1]),
            dtype=value_states.dtype,
            device=query.device,
        )
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

            active_in_chunk = (q_end > max(0, q_len - self.query_window_len))

            if (
                not self.capture_stats
                and layer_type == "sliding_attention"
                and self.swa_shield_beta > 0.0
                and k_len > 512
                and active_in_chunk
            ):
                logits_fp32 = attn_weights.float()
                del attn_weights
                global_act_start = max(q_start, q_len - self.query_window_len)
                local_start = global_act_start - q_start
                dist, _ = build_causal_distance_and_positions(global_act_start, q_end, q_len, k_len, query.device)
                swa_damp = ((dist - 48.0).clamp(min=0.0, max=464.0) / 464.0) * self.swa_shield_beta
                logits_fp32[:, :, local_start:, :] -= swa_damp.unsqueeze(0).unsqueeze(0)
                attn_probs = nn.functional.softmax(logits_fp32, dim=-1)
                del logits_fp32, dist, swa_damp
                attn_output[:, :, q_start:q_end, :] = torch.matmul(attn_probs.to(value_states.dtype), value_states)
                del attn_probs
                continue

            if not self.capture_stats and (self.mode == "baseline" or not layer_has_targets or not active_in_chunk):
                logits_fp32 = attn_weights.float()
                del attn_weights
                attn_probs = nn.functional.softmax(logits_fp32, dim=-1)
                del logits_fp32
                attn_output[:, :, q_start:q_end, :] = torch.matmul(attn_probs.to(value_states.dtype), value_states)
                del attn_probs
                continue

            logits_fp32 = attn_weights.float()
            del attn_weights

            if self.capture_stats:
                dist, _ = build_causal_distance_and_positions(q_start, q_end, q_len, k_len, query.device)
                attn_probs = nn.functional.softmax(logits_fp32, dim=-1)
                del logits_fp32
                with torch.no_grad():
                    probs_b0 = attn_probs[0]
                    mean_dist = (probs_b0 * dist.unsqueeze(0)).sum(dim=-1).mean(dim=-1)
                    eff_mass = 1.0 / (mean_dist + 1e-6)
                    entropy = -torch.sum(probs_b0 * torch.log(probs_b0 + 1e-9), dim=-1)
                    for h in range(num_heads):
                        self.head_masses[(layer_idx, h)] = eff_mass[h].item()
                        self.head_entropies[(layer_idx, h)] = entropy[h].cpu().numpy()
                del dist
                attn_output[:, :, q_start:q_end, :] = torch.matmul(attn_probs.to(value_states.dtype), value_states)
                del attn_probs
                continue

            global_act_start = max(q_start, q_len - self.query_window_len)
            local_start = global_act_start - q_start
            dist, _ = build_causal_distance_and_positions(global_act_start, q_end, q_len, k_len, query.device)
            dist_long = (dist - 32.0).clamp_(min=0.0)

            c_l = self.layer_horizons.get(layer_idx, 0.45)
            horizon_tokens = c_l * max(float(k_len), self.calib_horizon)
            potential = (dist_long / max(horizon_tokens, 1.0)).clamp_(min=0.0, max=1.0).pow_(self.plateau_power)
            del dist, dist_long

            if local_start > 0:
                logits_fp32[:, :, :local_start, :] = nn.functional.softmax(logits_fp32[:, :, :local_start, :], dim=-1)

            for h in range(num_heads):
                head_slice = logits_fp32[:, h, local_start:, :]
                if (layer_idx, h) in self.target_heads:
                    kappa = self.target_heads[(layer_idx, h)]
                    delta_l = (self.beta * kappa * potential).unsqueeze(0)
                    s_free = nn.functional.softmax(head_slice + delta_l, dim=-1)
                    if self.eta > 0.0:
                        s_heavy = nn.functional.softmax(head_slice - delta_l, dim=-1)
                        s_super = ((1.0 + self.eta) * s_free - self.eta * s_heavy).clamp_(min=0.0)
                        s_super = s_super / (s_super.sum(dim=-1, keepdim=True) + 1e-9)
                        logits_fp32[:, h, local_start:, :] = s_super
                        del s_heavy, s_super
                    else:
                        logits_fp32[:, h, local_start:, :] = s_free
                    del delta_l, s_free
                else:
                    logits_fp32[:, h, local_start:, :] = nn.functional.softmax(head_slice, dim=-1)

            attn_probs = logits_fp32
            del potential
            attn_output[:, :, q_start:q_end, :] = torch.matmul(attn_probs.to(value_states.dtype), value_states)
            del attn_probs

        return attn_output.transpose(1, 2).contiguous(), None


def run_self_test():
    with StageWatchdog("LOCAL_SELF_TEST", 20):
        # Verify Orthogonal Procrustes SVD math on synthetic rotated tensors
        torch.manual_seed(42)
        T_cal, d = 128, 64
        U = torch.randn(T_cal, d)
        true_rot, _ = torch.linalg.qr(torch.randn(d, d))
        X = torch.matmul(U, true_rot) + 0.05 * torch.randn(T_cal, d)

        U_n = U / (U.norm(dim=-1, keepdim=True) + 1e-6)
        X_n = X / (X.norm(dim=-1, keepdim=True) + 1e-6)
        raw_cos = nn.functional.cosine_similarity(U_n, X_n, dim=-1).mean().item()

        u_svd, _, vh_svd = torch.linalg.svd(torch.matmul(U_n.T, X_n), full_matrices=False)
        omega = torch.matmul(u_svd[:, :32], vh_svd[:32, :])
        aligned_cos = nn.functional.cosine_similarity(torch.matmul(U_n, omega), X_n, dim=-1).mean().item()
        assert aligned_cos > raw_cos + 0.5, f"Procrustes failed: {raw_cos} -> {aligned_cos}"
        log_heartbeat("LOCAL_SELF_TEST", f"PROCRUSTES PARALLEL TRANSPORT VERIFIED (Cosine: {raw_cos:.4f} -> {aligned_cos:.4f})!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-E4B-it")
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--output_json", type=str, default="results/richardson_phase8_fiber_bundle_results.json")
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

    controller = Phase8FiberBundleController(model, text_cfg, layers, calib_horizon=calib_horizon)
    ALL_ATTENTION_FUNCTIONS["eager"] = controller.attention_forward

    # STAGE 1: BRAIN SCAN + PROCRUSTES PARALLEL TRANSPORT CALIBRATION
    with StageWatchdog("STAGE1_FIBER_SCAN", 120):
        controller.capture_stats = True
        with torch.no_grad():
            model(**calib_inputs, use_cache=False, logits_to_keep=1)
        controller.capture_stats = False
        del calib_inputs

        controller.compute_procrustes_transport_operators()

        print("\n" + "=" * 124)
        print("STAGE 1B: FIBER-BUNDLE PROCRUSTES CONNECTION DIAGNOSTICS ACROSS THE 6 SWA GAPS")
        print("=" * 124)
        print(
            f"{'SWA Layer Gap':<15} | {'Unrotated Cos':<15} | {'Procrustes (r=128) Cos':<24} | "
            f"{'Full SVD Cos':<15} | {'Top-128 SVD Energy':<20} | {'Mean ||u_l||_2'}"
        )
        print("-" * 124)
        for gap_k, diag in controller.fiber_diagnostics.items():
            print(
                f"{gap_k:<15} | {diag['raw_unrotated_cosine']:>+13.4f}  | "
                f"{diag['procrustes_rank128_cosine']:>+21.4f}   | "
                f"{diag['procrustes_full_cosine']:>+13.4f}  | "
                f"{diag['top128_singular_energy_pct']:>17.2f}%  | {diag['mean_calib_u_norm']:>10.2f}",
                flush=True,
            )

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

        global_layers = sorted(by_layer.keys())
        quorum2_map = build_quorum_head_map(by_layer, heads_per_layer=2, calib_horizon=calib_horizon)
        quorum4_map = build_quorum_head_map(by_layer, heads_per_layer=4, calib_horizon=calib_horizon)

        unif_5hop_h = {l_idx: 0.40 for l_idx in global_layers}
        rg_5hop_h = {5: 0.38, 11: 0.44, 17: 0.52, 23: 0.64, 29: 0.74, 35: 0.85, 41: 0.90}
        unif_babi_h = {l_idx: 0.45 for l_idx in global_layers}

    # Prepare 5-Hop benchmark & 10-seed BABILong qa3 benchmark
    prompts_5hop = build_5hop_prompts()
    inp_5hop = tokenizer(bos + prompts_5hop["5-Hop"], return_tensors="pt").to(device)
    vault_codes = ["79421", "63819", "49281", "54712", "83914", "37192", "28491", "91824"]
    vault_toks_5hop = [tokenizer.encode(c, add_special_tokens=False)[0] for c in vault_codes]
    target_tok_5hop = vault_toks_5hop[0]
    omega_tok_5hop = vault_toks_5hop[1]

    babi_instances = []
    for s in range(args.num_seeds):
        p_qa, t_qa, d_qa, all_qa = generate_babilong_qa3_instance(200 + s)
        babi_instances.append((
            s,
            tokenizer(bos + p_qa, return_tensors="pt").to(device),
            tokenizer.encode(t_qa, add_special_tokens=False)[0],
            tokenizer.encode(d_qa, add_special_tokens=False)[0],
            [tokenizer.encode(v, add_special_tokens=False)[0] for v in all_qa],
        ))

    # (label, mode, beta, eta, sw_shield, proc_gamma, pullback_mode, head_map, h_5hop, h_babi)
    configs = [
        ("1. Unmodified Baseline", "baseline", 0.0, 0.0, 0.0, 0.0, "none", quorum2_map, unif_5hop_h, unif_babi_h),
        ("2. Horiz Cubic Q14 + SWA Shield (Phase 6/7 Ref)", "dual_cubic_plateau", 3.5, 0.75, 2.5, 0.0, "none", quorum2_map, unif_5hop_h, unif_babi_h),
        ("3. Horiz Stepped RG Q28 (Phase 6/7 Ref)", "dual_cubic_plateau", 3.5, 0.75, 0.0, 0.0, "none", quorum4_map, rg_5hop_h, unif_babi_h),
        ("4. Covariant Procrustes Highway (Q14, gv=0.12)", "dual_cubic_plateau", 3.5, 0.75, 2.5, 0.12, "none", quorum2_map, unif_5hop_h, unif_babi_h),
        ("5. Vocab-Fiber Pullback (Q14, Unsoftcapped L35+L41)", "dual_cubic_plateau", 3.5, 0.75, 2.5, 0.0, "unsoftcapped_l41", quorum2_map, unif_5hop_h, unif_babi_h),
        ("6. Dual-State Vocab-Fiber Richardson (Q14, L35/41 - L23/29)", "dual_cubic_plateau", 3.5, 0.75, 2.5, 0.0, "dual_35_41_minus_23", quorum2_map, unif_5hop_h, unif_babi_h),
        ("7. Full 2D Covariant (Procrustes + Dual Vocab Pullback, Q14)", "dual_cubic_plateau", 3.5, 0.75, 2.5, 0.10, "dual_35_41_minus_23", quorum2_map, unif_5hop_h, unif_babi_h),
        ("8. Full 2D Stepped RG (Q28 + Dual Vocab Pullback)", "dual_cubic_plateau", 3.5, 0.75, 2.0, 0.08, "dual_35_41_minus_23", quorum4_map, rg_5hop_h, unif_babi_h),
    ]

    results = {
        "model_id": args.model_id,
        "fiber_procrustes_diagnostics": controller.fiber_diagnostics,
        "evaluations": [],
    }

    with StageWatchdog("STAGE2_PHASE8_EVAL", 900):
        print("\n" + "=" * 146)
        print("STAGE 2: PHASE 8 GAUGE-COVARIANT VERTICAL EXTRAPOLATION (5-HOP @ 9,912 TOKENS & 10-SEED BABILONG QA3)")
        print("=" * 146)
        print(
            f"{'Configuration':<60} | {'5-Hop Prob':<12} | {'5-Hop Decoy':<12} | "
            f"{'BABILong Top-1':<15} | {'BABILong Mean Prob':<20} | {'BABILong Decoy'}"
        )
        print("-" * 146)

        for label, mode, beta, eta, sw_shield, proc_g, pb_mode, head_map, h_5hop, h_babi in configs:
            t0 = time.time()
            controller.mode = mode
            controller.beta = beta
            controller.eta = eta
            controller.swa_shield_beta = sw_shield
            controller.procrustes_gamma = proc_g
            controller.vocab_pullback_mode = pb_mode
            controller.target_heads = head_map

            # 1. Evaluate 5-Hop (9,912 tokens)
            controller.layer_horizons = h_5hop
            controller.clear_step_state()
            with torch.no_grad():
                out_5 = model(**inp_5hop, use_cache=False, logits_to_keep=1)
                probs_5 = controller.evaluate_output_probabilities(out_5.logits[0, -1, :])
                del out_5

            p5_target = probs_5[target_tok_5hop].item() * 100.0
            p5_decoy = probs_5[omega_tok_5hop].item() * 100.0
            scores_5 = sorted([(tok, probs_5[tok].item()) for tok in vault_toks_5hop], key=lambda x: x[1], reverse=True)
            rank_5 = [idx + 1 for idx, (tok, _) in enumerate(scores_5) if tok == target_tok_5hop][0]

            # 2. Evaluate 10-Seed BABILong qa3
            controller.layer_horizons = h_babi
            babi_wins = 0
            babi_t_probs = []
            babi_d_probs = []

            for s_idx, b_inp, t_tok, d_tok, all_toks in babi_instances:
                controller.clear_step_state()
                with torch.no_grad():
                    out_b = model(**b_inp, use_cache=False, logits_to_keep=1)
                    probs_b = controller.evaluate_output_probabilities(out_b.logits[0, -1, :])
                    del out_b

                pt = probs_b[t_tok].item() * 100.0
                pd = probs_b[d_tok].item() * 100.0
                c_scores = sorted([(tok, probs_b[tok].item()) for tok in all_toks], key=lambda x: x[1], reverse=True)
                rk = [idx + 1 for idx, (tok, _) in enumerate(c_scores) if tok == t_tok][0]
                if rk == 1:
                    babi_wins += 1
                babi_t_probs.append(pt)
                babi_d_probs.append(pd)

            babi_acc = (babi_wins / len(babi_instances)) * 100.0
            babi_mean_t = float(np.mean(babi_t_probs))
            babi_std_t = float(np.std(babi_t_probs))
            babi_mean_d = float(np.mean(babi_d_probs))
            dt = time.time() - t0

            print(
                f"{label:<60} | {p5_target:>8.2f}% #{rank_5} | {p5_decoy:>10.2f}% | "
                f"{babi_acc:>11.1f}%    | {babi_mean_t:>7.2f}% ± {babi_std_t:<6.2f}% | {babi_mean_d:>9.2f}% ({dt:.1f}s)",
                flush=True,
            )

            results["evaluations"].append({
                "configuration": label,
                "procrustes_gamma": proc_g,
                "vocab_pullback_mode": pb_mode,
                "five_hop_target_prob_pct": round(p5_target, 4),
                "five_hop_decoy_prob_pct": round(p5_decoy, 4),
                "five_hop_rank": rank_5,
                "babilong_qa3_top1_acc_pct": round(babi_acc, 2),
                "babilong_qa3_mean_target_prob_pct": round(babi_mean_t, 4),
                "babilong_qa3_std_target_prob_pct": round(babi_std_t, 4),
                "babilong_qa3_mean_decoy_prob_pct": round(babi_mean_d, 4),
                "elapsed_sec": round(dt, 2),
            })

            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)

        controller.remove_hooks()
        log_heartbeat("STAGE2_PHASE8_EVAL", f"Saved Phase 8 results to {args.output_json}")


if __name__ == "__main__":
    main()
