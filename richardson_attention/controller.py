import torch
import torch.nn as nn
import numpy as np

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:
    ALL_ATTENTION_FUNCTIONS = {}


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
        dist[:, 0] = 0.0  # Pin <bos> token-0 attention sink
    return dist, q_pos


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


def build_quorum_head_map(by_layer: dict, heads_per_layer: int, calib_horizon: float) -> dict:
    quorum_map = {}
    for l_idx, h_list in sorted(by_layer.items()):
        ranked = sorted(h_list, key=lambda x: (x[2], -x[3]))
        for (l, h, m, a) in ranked[:heads_per_layer]:
            raw_kappa = m * calib_horizon
            quorum_map[(l, h)] = float(np.clip(raw_kappa, 1.8, 2.6))
    return quorum_map


class Phase5RGRichardsonController:
    """
    Runtime inference controller implementing Surgical Dual-State Symmetric
    Boltzmann Extrapolation, Query-Slice Gating, Iso-Potential Cubic Plateaus,
    SWA Tail Decoy Shielding, and Depth-Progressive Renormalization Group (RG) Relays.
    """
    def __init__(self, text_cfg, layers, calib_horizon: float = 1400.0, q_chunk_size: int = 2048):
        self.mode = "dual_cubic_plateau"
        self.beta = 3.5
        self.eta = 0.75
        self.query_slice_only = True
        self.query_window_len = 64
        self.plateau_power = 3.0
        self.swa_shield_beta = 0.0
        self.calib_horizon = calib_horizon
        self.q_chunk_size = q_chunk_size
        self.target_heads = {}      # Dict[(layer_idx, head_idx) -> calibrated_kappa]
        self.layer_horizons = {}    # Dict[layer_idx -> horizon_fraction c_l in (0, 1)]
        self.capture_stats = False
        self.head_masses = {}
        self.head_entropies = {}

        layer_types = getattr(text_cfg, "layer_types", None)
        for idx, layer in enumerate(layers):
            attn_mod = layer.self_attn
            attn_mod.layer_idx = idx
            if layer_types is not None and idx < len(layer_types):
                attn_mod.layer_type_str = layer_types[idx]
            else:
                is_sliding = getattr(attn_mod, "is_sliding", False)
                attn_mod.layer_type_str = "sliding_attention" if is_sliding else "full_attention"

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

            active_in_chunk = True
            if self.query_slice_only and not self.capture_stats:
                active_in_chunk = (q_end > max(0, q_len - self.query_window_len))

            # Optional SWA Tail Decoy Shield on sliding_attention layers
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
                dist, q_pos = build_causal_distance_and_positions(q_start, q_end, q_len, k_len, query.device)
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
                del dist, q_pos
                attn_output[:, :, q_start:q_end, :] = torch.matmul(attn_probs.to(value_states.dtype), value_states)
                del attn_probs
                continue

            if self.query_slice_only:
                global_act_start = max(q_start, q_len - self.query_window_len)
                local_start = global_act_start - q_start
            else:
                global_act_start = q_start
                local_start = 0

            dist, q_pos = build_causal_distance_and_positions(global_act_start, q_end, q_len, k_len, query.device)
            dist_long = (dist - 32.0).clamp_(min=0.0)

            if self.mode == "dual_log":
                row_denom = torch.log1p(q_pos.clamp(min=self.calib_horizon))
                potential = torch.log1p(dist_long) / row_denom
                del row_denom
            elif self.mode == "dual_cubic_plateau":
                c_l = self.layer_horizons.get(layer_idx, 0.54)
                horizon_tokens = c_l * max(float(k_len), self.calib_horizon)
                potential = (dist_long / max(horizon_tokens, 1.0)).clamp_(min=0.0, max=1.0).pow_(self.plateau_power)
            else:
                potential = torch.zeros_like(dist_long)

            del dist, q_pos, dist_long

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


def attach_richardson_attention(
    model,
    tokenizer,
    heads_per_layer: int = 2,
    mode: str = "dual_cubic_plateau",
    beta: float = 3.5,
    eta: float = 0.75,
    query_window_len: int = 64,
    rg_schedule: bool | str = False,
    uniform_c: float = 0.54,
    swa_shield_beta: float = 0.0,
):
    """
    One-line attachment helper: runs the zero-shot Stage 1 Precision-Matrix Brain Scan,
    isolates the Global Logic Quorum, and attaches the Richardson Attention hook.

    Args:
        rg_schedule: False for uniform_c, "3hop" for [0.52, 0.68, 0.85], or "5hop" / True
            for the 7-step 5-Hop RG schedule [0.38, 0.44, 0.52, 0.64, 0.74, 0.85, 0.90].
        swa_shield_beta: Optional damping (e.g. 2.5) for 48 < d <= 512 inside SWA layers.
    """
    device = next(model.parameters()).device
    text_cfg, layers = get_text_config_and_layers(model)

    bos = tokenizer.bos_token or ""
    calib_text = bos + (
        "In formal logic and category theory, a deduction system connects initial axioms "
        "to distant theorems across intermediate lemmas. Suppose Premise Alpha states that "
        "every topological manifold in Class Z admits a flat connection, and Premise Beta "
        "states that System Omega belongs to Class Z. "
    ) * 25
    calib_inputs = tokenizer(calib_text, return_tensors="pt").to(device)
    calib_horizon = float(calib_inputs.input_ids.shape[1])

    controller = Phase5RGRichardsonController(text_cfg, layers, calib_horizon=calib_horizon)
    ALL_ATTENTION_FUNCTIONS["eager"] = controller.attention_forward

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

    global_layers = sorted(by_layer.keys())
    controller.target_heads = build_quorum_head_map(by_layer, heads_per_layer=heads_per_layer, calib_horizon=calib_horizon)

    if (rg_schedule is True or rg_schedule == "5hop") and len(global_layers) >= 3:
        schedule_7 = [0.38, 0.44, 0.52, 0.64, 0.74, 0.85, 0.90]
        controller.layer_horizons = {}
        for i, l_idx in enumerate(global_layers):
            step_idx = min(i, len(schedule_7) - 1)
            controller.layer_horizons[l_idx] = schedule_7[step_idx]
    elif rg_schedule == "3hop" and len(global_layers) >= 3:
        n_g = len(global_layers)
        controller.layer_horizons = {}
        for i, l_idx in enumerate(global_layers):
            if i < n_g * 0.45:
                controller.layer_horizons[l_idx] = 0.52
            elif i < n_g * 0.75:
                controller.layer_horizons[l_idx] = 0.68
            else:
                controller.layer_horizons[l_idx] = 0.85
    else:
        controller.layer_horizons = {l_idx: uniform_c for l_idx in global_layers}

    controller.mode = mode
    controller.beta = beta
    controller.eta = eta
    controller.query_window_len = query_window_len
    controller.query_slice_only = True
    controller.swa_shield_beta = swa_shield_beta
    return controller
