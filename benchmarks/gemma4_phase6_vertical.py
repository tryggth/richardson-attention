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


class Phase6Vertical2DController:
    def __init__(self, model, text_cfg, layers, calib_horizon: float = 1400.0, q_chunk_size: int = 2048):
        self.model = model
        self.layers = layers
        self.mode = "baseline"
        self.beta = 3.5
        self.eta = 0.75
        self.query_slice_only = True
        self.query_window_len = 64
        self.plateau_power = 3.0
        self.calib_horizon = calib_horizon
        self.q_chunk_size = q_chunk_size

        # Vertical & SWA-Shield parameters
        self.vertical_gamma = 0.0      # Strength of Global-to-Global vertical highway injection
        self.vertical_decay = 0.75     # Momentum mu_v across successive global layers
        self.swa_shield_beta = 0.0     # Suppresses >48-token decoy tail inside 512-tok SWA layers

        self.target_heads = {}         # Dict[(layer_idx, head_idx) -> calibrated_kappa]
        self.layer_horizons = {}       # Dict[layer_idx -> horizon_fraction c_l in (0, 1)]
        self.capture_stats = False
        self.head_masses = {}
        self.head_entropies = {}

        # 5-Zone token index spans (Zone A = Hop 5 code, Zone B = Hop 4, Zone C = Hop 3, Zone D = Hop 2, Zone E = Hop 1)
        self.zone_spans = {}           # Dict["A"|"B"|"C"|"D"|"E" -> list[int]]
        self.per_layer_zone_attn = {}  # Dict[layer_idx -> Dict["A".."E" -> float]]

        # Vertical telemetry & Logit Lens state
        self.collect_vertical_telemetry = False
        self.target_tok_id = None
        self.omega_tok_id = None
        self.highway_buffer = None
        self.last_global_post_x = None
        self.last_global_attn_u = None
        self.last_global_idx = None
        self.current_attn_in = {}
        self.global_attn_updates = {}
        self.vertical_gap_stats = {}
        self.logit_lens_trajectory = {}

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
        self._register_vertical_hooks()

    def _get_final_norm_and_lm_head(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            norm = self.model.model.language_model.norm
        elif hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            norm = self.model.model.norm
        else:
            norm = None
        lm_head = getattr(self.model, "lm_head", None)
        return norm, lm_head

    def _register_vertical_hooks(self):
        """Registers lightweight hooks on Global layers to measure SWA gap drift, Logit Lens, and inject the Vertical Highway."""
        for idx in self.global_layer_indices:
            layer = self.layers[idx]

            # 1. Hook post_attention_layernorm (or self_attn) to capture exact Global Attention update u_k
            if hasattr(layer, "post_attention_layernorm"):
                def make_post_attn_hook(l_idx):
                    def hook(mod, inputs, output):
                        if not self.capture_stats and output.shape[1] > 256:
                            w = min(self.query_window_len, output.shape[1])
                            self.global_attn_updates[l_idx] = output[:, -w:, :].detach()
                    return hook
                self._hooks.append(layer.post_attention_layernorm.register_forward_hook(make_post_attn_hook(idx)))

            # 2. Pre-hook on Global Decoder Layer: measure SWA gap drift from previous Global Layer & inject Vertical Highway
            def make_layer_pre_hook(l_idx):
                def pre_hook(mod, args):
                    hidden_states = args[0]
                    seq_len = hidden_states.shape[1]
                    if self.capture_stats or seq_len <= 256:
                        return None

                    if l_idx == self.global_layer_indices[0]:
                        self.highway_buffer = None
                        self.last_global_post_x = None
                        self.last_global_attn_u = None
                        self.last_global_idx = None

                    w = min(self.query_window_len, seq_len)
                    cur_slice = hidden_states[:, -w:, :]
                    dev = cur_slice.device

                    # Record vertical gap telemetry across the preceding 5 SWA layers
                    if self.collect_vertical_telemetry and self.last_global_post_x is not None:
                        with torch.no_grad():
                            prev_x_last = self.last_global_post_x[0, -1, :].to(dev).float()
                            cur_x_last = cur_slice[0, -1, :].float()
                            u_prev_last = (
                                self.last_global_attn_u[0, -1, :].to(dev).float()
                                if self.last_global_attn_u is not None
                                else torch.zeros_like(prev_x_last)
                            )
                            swa_drift = cur_x_last - prev_x_last

                            u_norm = u_prev_last.norm().item()
                            swa_norm = swa_drift.norm().item()
                            norm_infl = (cur_x_last.norm() / (prev_x_last.norm() + 1e-6)).item()
                            cos_ret = nn.functional.cosine_similarity(u_prev_last.unsqueeze(0), cur_x_last.unsqueeze(0)).item()

                            gap_key = f"L{self.last_global_idx}->L{l_idx}"
                            self.vertical_gap_stats[gap_key] = {
                                "global_attn_u_norm": round(u_norm, 4),
                                "swa_5layer_drift_norm": round(swa_norm, 4),
                                "drift_to_signal_ratio": round(swa_norm / (u_norm + 1e-6), 4),
                                "rms_norm_inflation": round(norm_infl, 4),
                                "cosine_retention": round(cos_ret, 4),
                            }

                    # Apply Vertical Highway injection if enabled
                    if self.vertical_gamma > 0.0 and self.highway_buffer is not None:
                        hw_buf = self.highway_buffer.to(dev).float()
                        orig_rms = cur_slice.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
                        boosted = cur_slice.float() + self.vertical_gamma * hw_buf
                        new_rms = boosted.pow(2).mean(dim=-1, keepdim=True).sqrt()
                        boosted = (boosted * (orig_rms / (new_rms + 1e-6))).to(hidden_states.dtype)

                        new_hidden = hidden_states.clone()
                        new_hidden[:, -w:, :] = boosted
                        return (new_hidden,) + args[1:]
                    return None
                return pre_hook

            # 3. Post-hook on Global Decoder Layer: update Vertical Highway buffer & compute Logit Lens
            def make_layer_post_hook(l_idx):
                def post_hook(mod, args, output):
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    seq_len = hidden_states.shape[1]
                    if self.capture_stats or seq_len <= 256:
                        return None

                    w = min(self.query_window_len, seq_len)
                    post_slice = hidden_states[:, -w:, :].detach()
                    u_attn = self.global_attn_updates.get(l_idx, None)

                    if u_attn is not None:
                        if self.highway_buffer is None:
                            self.highway_buffer = u_attn.clone()
                        else:
                            self.highway_buffer = self.vertical_decay * self.highway_buffer.to(u_attn.device) + u_attn

                    self.last_global_post_x = post_slice
                    self.last_global_attn_u = u_attn
                    self.last_global_idx = l_idx

                    # Logit Lens projection on final query token after this Global Layer
                    if self.collect_vertical_telemetry and self.target_tok_id is not None:
                        norm_mod, lm_head = self._get_final_norm_and_lm_head()
                        if norm_mod is not None and lm_head is not None:
                            with torch.no_grad():
                                last_h = hidden_states[:, -1:, :]
                                norm_dev = next(norm_mod.parameters()).device
                                lm_dev = next(lm_head.parameters()).device
                                normed_h = norm_mod(last_h.to(norm_dev))
                                lens_logits = lm_head(normed_h.to(lm_dev))[0, -1, :].float()
                                lens_probs = torch.softmax(lens_logits, dim=-1)
                                self.logit_lens_trajectory[f"L{l_idx}"] = {
                                    "target_79421_pct": round(lens_probs[self.target_tok_id].item() * 100.0, 4),
                                    "decoy_63819_pct": round(lens_probs[self.omega_tok_id].item() * 100.0, 4),
                                }
                    return None
                return post_hook

            self._hooks.append(layer.register_forward_pre_hook(make_layer_pre_hook(idx)))
            self._hooks.append(layer.register_forward_hook(make_layer_post_hook(idx)))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear_telemetry(self):
        self.per_layer_zone_attn.clear()
        self.vertical_gap_stats.clear()
        self.logit_lens_trajectory.clear()
        self.global_attn_updates.clear()
        self.highway_buffer = None
        self.last_global_post_x = None
        self.last_global_attn_u = None
        self.last_global_idx = None

    def _record_zone_attn(self, layer_idx: int, num_heads: int, attn_probs: torch.Tensor):
        k_len = attn_probs.shape[-1]
        if self.zone_spans and "E" in self.zone_spans and k_len > self.zone_spans["E"][-1]:
            with torch.no_grad():
                target_h = [h for h in range(num_heads) if (layer_idx, h) in self.target_heads]
                if not target_h:
                    target_h = list(range(num_heads))
                layer_dict = {}
                for z_key, z_indices in self.zone_spans.items():
                    vals = [attn_probs[0, h, -1, z_indices].sum().item() * 100.0 for h in target_h]
                    layer_dict[z_key] = round(float(np.mean(vals)), 4)
                self.per_layer_zone_attn[f"L{layer_idx}"] = layer_dict

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

            # Handle SWA Decoy Shield on sliding_attention layers when active on long sequences
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
                # Shield query tokens (dist <= 48 untouched; 48 < dist <= 512 gently damped against Barrier-5 decoys)
                swa_damp = ((dist - 48.0).clamp(min=0.0, max=464.0) / 464.0) * self.swa_shield_beta
                logits_fp32[:, :, local_start:, :] -= swa_damp.unsqueeze(0).unsqueeze(0)
                attn_probs = nn.functional.softmax(logits_fp32, dim=-1)
                del logits_fp32, dist, swa_damp
                attn_output[:, :, q_start:q_end, :] = torch.matmul(attn_probs.to(value_states.dtype), value_states)
                del attn_probs
                continue

            # Fast path for untouched layers or prefill chunks
            if not self.capture_stats and (self.mode == "baseline" or not layer_has_targets or not active_in_chunk):
                logits_fp32 = attn_weights.float()
                del attn_weights
                attn_probs = nn.functional.softmax(logits_fp32, dim=-1)
                del logits_fp32
                if layer_type == "full_attention" and q_end == q_len:
                    self._record_zone_attn(layer_idx, num_heads, attn_probs)
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

            c_l = self.layer_horizons.get(layer_idx, 0.40)
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

            if layer_type == "full_attention" and q_end == q_len:
                self._record_zone_attn(layer_idx, num_heads, attn_probs)

            attn_output[:, :, q_start:q_end, :] = torch.matmul(attn_probs.to(value_states.dtype), value_states)
            del attn_probs

        return attn_output.transpose(1, 2).contiguous(), None


def build_5hop_prompts():
    """
    Constructs a 5-Zone deep registry separated by >1,150-token SWA-blocking barriers (~9,100 tokens total):
    - Zone A (toks ~90):   SUBVAULT-PRIME  -> 79421 (1-Hop target)
    - Zone B (toks ~1350): VAULT-LAZARUS   -> SUBVAULT-PRIME (2-Hop target)
    - Zone C (toks ~2600): SECTOR-KRONOS   -> VAULT-LAZARUS (3-Hop target)
    - Zone D (toks ~3850): DIRECTOR-VANCE  -> SECTOR-KRONOS (4-Hop target)
    - Zone E (toks ~5100): ENVOY-SOLARIS   -> DIRECTOR-VANCE (5-Hop target)
    - Barrier 5 (toks ~5250..9050): Telemetry + 12 adversarial Tier-2 audit decoys pointing to SUBVAULT-OMEGA (63819).
    """
    zone_a = (
        "=== ZONE A: TIER-0 IMMUTABLE SUBVAULT CODE REGISTRY ===\n"
        "- SUBVAULT-ALPHA    | TIER-0 GENESIS CODE: 49281\n"
        "- SUBVAULT-HELIOS   | TIER-0 GENESIS CODE: 54712\n"
        "- SUBVAULT-HYPERION | TIER-0 GENESIS CODE: 83914\n"
        "- SUBVAULT-PRIME    | TIER-0 GENESIS CODE: 79421\n"
        "- SUBVAULT-NYX      | TIER-0 GENESIS CODE: 37192\n"
        "- SUBVAULT-OMEGA    | TIER-0 GENESIS CODE: 63819\n"
        "- SUBVAULT-VULCAN   | TIER-0 GENESIS CODE: 28491\n"
        "- SUBVAULT-ZEPHYR   | TIER-0 GENESIS CODE: 91824\n\n"
    )
    zone_b = (
        "\n=== ZONE B: TIER-0 MASTER VAULT ALLOCATION REGISTRY ===\n"
        "- VAULT-ALPHA    | ASSIGNED SUBVAULT: SUBVAULT-ALPHA\n"
        "- VAULT-HELIOS   | ASSIGNED SUBVAULT: SUBVAULT-HELIOS\n"
        "- VAULT-HYPERION | ASSIGNED SUBVAULT: SUBVAULT-HYPERION\n"
        "- VAULT-LAZARUS  | ASSIGNED SUBVAULT: SUBVAULT-PRIME\n"
        "- VAULT-NYX      | ASSIGNED SUBVAULT: SUBVAULT-NYX\n"
        "- VAULT-OMEGA    | ASSIGNED SUBVAULT: SUBVAULT-OMEGA\n"
        "- VAULT-VULCAN   | ASSIGNED SUBVAULT: SUBVAULT-VULCAN\n"
        "- VAULT-ZEPHYR   | ASSIGNED SUBVAULT: SUBVAULT-ZEPHYR\n\n"
    )
    zone_c = (
        "\n=== ZONE C: TIER-0 COMMAND SECTOR ROUTING REGISTRY ===\n"
        "- SECTOR-ATLAS    | ASSIGNED PRIMARY VAULT: VAULT-OMEGA\n"
        "- SECTOR-BOREALIS | ASSIGNED PRIMARY VAULT: VAULT-ALPHA\n"
        "- SECTOR-CYGNUS   | ASSIGNED PRIMARY VAULT: VAULT-ZEPHYR\n"
        "- SECTOR-DRACO    | ASSIGNED PRIMARY VAULT: VAULT-HELIOS\n"
        "- SECTOR-EREBUS   | ASSIGNED PRIMARY VAULT: VAULT-VULCAN\n"
        "- SECTOR-KRONOS   | ASSIGNED PRIMARY VAULT: VAULT-LAZARUS\n"
        "- SECTOR-ORION    | ASSIGNED PRIMARY VAULT: VAULT-NYX\n"
        "- SECTOR-PEGASUS  | ASSIGNED PRIMARY VAULT: VAULT-HYPERION\n\n"
    )
    zone_d = (
        "\n=== ZONE D: TIER-0 DIRECTORATE COMMAND REGISTRY ===\n"
        "- DIRECTOR-CHEN     | AUTHORIZED COMMAND SECTOR: SECTOR-BOREALIS\n"
        "- DIRECTOR-KAEL     | AUTHORIZED COMMAND SECTOR: SECTOR-ORION\n"
        "- DIRECTOR-MERCER   | AUTHORIZED COMMAND SECTOR: SECTOR-DRACO\n"
        "- DIRECTOR-ROSTOVA  | AUTHORIZED COMMAND SECTOR: SECTOR-PEGASUS\n"
        "- DIRECTOR-STERLING | AUTHORIZED COMMAND SECTOR: SECTOR-ATLAS\n"
        "- DIRECTOR-THORNE   | AUTHORIZED COMMAND SECTOR: SECTOR-CYGNUS\n"
        "- DIRECTOR-VANCE    | AUTHORIZED COMMAND SECTOR: SECTOR-KRONOS\n"
        "- DIRECTOR-WEBB     | AUTHORIZED COMMAND SECTOR: SECTOR-EREBUS\n\n"
    )
    zone_e = (
        "\n=== ZONE E: TIER-0 DIPLOMATIC ENVOY DISPATCH REGISTRY ===\n"
        "- ENVOY-ALTAIR   | REPORTING TO DIRECTOR: DIRECTOR-CHEN\n"
        "- ENVOY-CASTOR   | REPORTING TO DIRECTOR: DIRECTOR-KAEL\n"
        "- ENVOY-DENEB    | REPORTING TO DIRECTOR: DIRECTOR-MERCER\n"
        "- ENVOY-LYRA     | REPORTING TO DIRECTOR: DIRECTOR-ROSTOVA\n"
        "- ENVOY-POLARIS  | REPORTING TO DIRECTOR: DIRECTOR-STERLING\n"
        "- ENVOY-RIGEL    | REPORTING TO DIRECTOR: DIRECTOR-THORNE\n"
        "- ENVOY-SOLARIS  | REPORTING TO DIRECTOR: DIRECTOR-VANCE\n"
        "- ENVOY-VEGA     | REPORTING TO DIRECTOR: DIRECTOR-WEBB\n\n"
    )

    filler = (
        "Telemetry Log: Cryogenic bus voltage nominal at 4.2K; Josephson junction phase coherence locked. "
        "Syndrome parity check completed with zero faults; spectral damper calibration steady. "
    )
    decoy = (
        "\n[TIER-2 RUNTIME AUDIT] Temporary cross-check: ENVOY-SOLARIS and DIRECTOR-VANCE inspected "
        "SECTOR-ATLAS, VAULT-OMEGA, and SUBVAULT-OMEGA using temporary override key 11111 (non-binding).\n"
    )

    # 4 pure telemetry barriers (~1,150 tokens each, >2.2x the 512-token SWA window)
    b1 = filler * 33
    b2 = filler * 33
    b3 = filler * 33
    b4 = filler * 33
    b5 = (filler * 8 + decoy) * 12 + filler * 4

    prefix = zone_a + b1 + zone_b + b2 + zone_c + b3 + zone_d + b4 + zone_e + b5

    queries = {
        "1-Hop": (
            "\nQuestion: What is the 5-digit TIER-0 GENESIS CODE for SUBVAULT-PRIME?\n"
            "Answer: The 5-digit TIER-0 GENESIS CODE is "
        ),
        "2-Hop": (
            "\nQuestion: Trace the Master Vault Allocation Registry to the Subvault Code Registry: what is the "
            "5-digit TIER-0 GENESIS CODE for the subvault assigned to VAULT-LAZARUS?\n"
            "Answer: The 5-digit TIER-0 GENESIS CODE is "
        ),
        "3-Hop": (
            "\nQuestion: Trace the Command Sector Routing Registry, Master Vault Registry, and Subvault Code "
            "Registry: what is the 5-digit TIER-0 GENESIS CODE for the subvault assigned to the primary vault of SECTOR-KRONOS?\n"
            "Answer: The 5-digit TIER-0 GENESIS CODE is "
        ),
        "4-Hop": (
            "\nQuestion: Trace the Directorate Command Registry, Sector Routing Registry, Master Vault Registry, "
            "and Subvault Code Registry: what is the 5-digit TIER-0 GENESIS CODE for the subvault under the command of DIRECTOR-VANCE?\n"
            "Answer: The 5-digit TIER-0 GENESIS CODE is "
        ),
        "5-Hop": (
            "\nQuestion: Trace the Diplomatic Envoy Dispatch Registry, Directorate Command Registry, Sector Routing "
            "Registry, Master Vault Registry, and Subvault Code Registry: what is the 5-digit TIER-0 GENESIS CODE "
            "for the subvault under the director to whom ENVOY-SOLARIS reports?\n"
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


def build_quorum_head_map(by_layer: dict, heads_per_layer: int, calib_horizon: float) -> dict:
    quorum_map = {}
    for l_idx, h_list in sorted(by_layer.items()):
        ranked = sorted(h_list, key=lambda x: (x[2], -x[3]))
        for (l, h, m, a) in ranked[:heads_per_layer]:
            raw_kappa = m * calib_horizon
            quorum_map[(l, h)] = float(np.clip(raw_kappa, 1.8, 2.6))
    return quorum_map


def run_self_test():
    with StageWatchdog("LOCAL_SELF_TEST", 20):
        prompts = build_5hop_prompts()
        assert set(prompts.keys()) == {"1-Hop", "2-Hop", "3-Hop", "4-Hop", "5-Hop"}
        for k, text in prompts.items():
            assert "SUBVAULT-PRIME" in text and "ENVOY-SOLARIS" in text
        log_heartbeat("LOCAL_SELF_TEST", "PHASE 6 5-HOP PROMPTS & VERTICAL CONTROLLER SELF-TEST PASSED!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-E4B-it")
    parser.add_argument("--output_json", type=str, default="results/richardson_phase6_vertical_results.json")
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

    controller = Phase6Vertical2DController(model, text_cfg, layers, calib_horizon=calib_horizon)
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

        by_layer = {}
        for idx, (l_idx, h_idx) in enumerate(ordered_keys):
            if layers[l_idx].self_attn.layer_type_str == "full_attention":
                m = controller.head_masses[(l_idx, h_idx)]
                a = float(agency_scores[idx])
                by_layer.setdefault(l_idx, []).append((l_idx, h_idx, m, a))

        global_layer_indices = sorted(by_layer.keys())  # [5, 11, 17, 23, 29, 35, 41]
        quorum1_map = build_quorum_head_map(by_layer, heads_per_layer=1, calib_horizon=calib_horizon)
        quorum2_map = build_quorum_head_map(by_layer, heads_per_layer=2, calib_horizon=calib_horizon)
        quorum4_map = build_quorum_head_map(by_layer, heads_per_layer=4, calib_horizon=calib_horizon)

        # Uniform 5-Hop Cutoff (c=0.40 covers Zone E at d/T=0.44 up to Zone A at d/T=0.99)
        uniform_5hop_horizons = {l_idx: 0.40 for l_idx in global_layer_indices}
        # Stepped 5-Hop RG Cutoff across the 7 Global Layers
        rg_5hop_horizons = {
            5: 0.38, 11: 0.44, 17: 0.52,
            23: 0.64, 29: 0.74,
            35: 0.85, 41: 0.90,
        }

    # Fluency PPL Guardrail
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

    # STAGE 2: 1-TO-5 HOP VERTICAL & DEEP-TELEMETRY SUITE
    with StageWatchdog("STAGE2_PHASE6_SUITE", 600):
        prompts = build_5hop_prompts()
        tokenized_prompts = {
            hop: tokenizer(bos + text, return_tensors="pt").to(device)
            for hop, text in prompts.items()
        }
        sample_ids = tokenized_prompts["5-Hop"].input_ids[0].tolist()
        seq_len = len(sample_ids)

        controller.zone_spans = {
            "A": find_subseq(sample_ids, tokenizer.encode("\n- SUBVAULT-PRIME    | TIER-0 GENESIS CODE: 79421\n", add_special_tokens=False)),
            "B": find_subseq(sample_ids, tokenizer.encode("\n- VAULT-LAZARUS  | ASSIGNED SUBVAULT: SUBVAULT-PRIME\n", add_special_tokens=False)),
            "C": find_subseq(sample_ids, tokenizer.encode("\n- SECTOR-KRONOS   | ASSIGNED PRIMARY VAULT: VAULT-LAZARUS\n", add_special_tokens=False)),
            "D": find_subseq(sample_ids, tokenizer.encode("\n- DIRECTOR-VANCE    | AUTHORIZED COMMAND SECTOR: SECTOR-KRONOS\n", add_special_tokens=False)),
            "E": find_subseq(sample_ids, tokenizer.encode("\n- ENVOY-SOLARIS  | REPORTING TO DIRECTOR: DIRECTOR-VANCE\n", add_special_tokens=False)),
        }
        for zk, zv in controller.zone_spans.items():
            assert zv, f"Failed to locate Zone {zk} token span!"

        vault_code_map = {
            "SUBVAULT-PRIME (79421)": "79421",
            "SUBVAULT-OMEGA (63819)": "63819",
            "SUBVAULT-ALPHA (49281)": "49281",
            "SUBVAULT-HELIOS (54712)": "54712",
            "SUBVAULT-HYPERION (83914)": "83914",
            "SUBVAULT-NYX (37192)": "37192",
            "SUBVAULT-VULCAN (28491)": "28491",
            "SUBVAULT-ZEPHYR (91824)": "91824",
        }
        vault_tok_map = {k: tokenizer.encode(v, add_special_tokens=False)[0] for k, v in vault_code_map.items()}
        target_tok = vault_tok_map["SUBVAULT-PRIME (79421)"]
        omega_tok = vault_tok_map["SUBVAULT-OMEGA (63819)"]
        trap_tok = tokenizer.encode("11111", add_special_tokens=False)[0]
        all_vault_toks = list(vault_tok_map.values())

        controller.target_tok_id = target_tok
        controller.omega_tok_id = omega_tok
        controller.collect_vertical_telemetry = True

        # (label, mode, beta, eta, v_gamma, swa_shield, head_map, horizon_map)
        configs = [
            ("1. Baseline (Unmodified)", "baseline", 0.0, 0.0, 0.0, 0.0, quorum2_map, uniform_5hop_horizons),
            ("2. Horiz-Only Sparse (Q7, c=0.40, b=3.5)", "dual_cubic_plateau", 3.5, 0.75, 0.0, 0.0, quorum1_map, uniform_5hop_horizons),
            ("3. Horiz-Only Cubic (Q14, c=0.40, b=3.5)", "dual_cubic_plateau", 3.5, 0.75, 0.0, 0.0, quorum2_map, uniform_5hop_horizons),
            ("4. Horiz-Only Stepped RG (Q28, RG, b=3.5)", "dual_cubic_plateau", 3.5, 0.75, 0.0, 0.0, quorum4_map, rg_5hop_horizons),
            ("5. 2D Vert-Highway on Sparse Q7 (gv=0.30)", "dual_cubic_plateau", 3.5, 0.75, 0.30, 0.0, quorum1_map, uniform_5hop_horizons),
            ("6. 2D Vert-Highway on Cubic Q14 (gv=0.25)", "dual_cubic_plateau", 3.5, 0.75, 0.25, 0.0, quorum2_map, uniform_5hop_horizons),
            ("7. 2D Cubic Q14 + SWA Decoy Shield (sw=2.5)", "dual_cubic_plateau", 3.5, 0.75, 0.0, 2.5, quorum2_map, uniform_5hop_horizons),
            ("8. Full 2D Superconducting (Q14 + gv=0.25 + sw=2.5)", "dual_cubic_plateau", 3.5, 0.75, 0.25, 2.5, quorum2_map, uniform_5hop_horizons),
        ]

        print("\n" + "=" * 156)
        print(
            f"MODEL: {args.model_id} | CONTEXT: ~{seq_len} TOKENS | "
            f"ZONES A..E: {[controller.zone_spans[z][0] for z in ['A','B','C','D','E']]}"
        )
        print("=" * 156)
        print(
            f"{'Configuration':<52} | {'Hop':<6} | {'PPL':<6} | {'Target(79421)':<13} | "
            f"{'Decoy(63819)':<12} | {'Rank':<9} | {'L29_Lens':<9} | {'L35_Lens':<9} | {'L41_Lens'}"
        )
        print("-" * 156)

        results = {
            "model_id": args.model_id,
            "context_tokens": seq_len,
            "zone_token_spans": {k: [v[0], v[-1]] for k, v in controller.zone_spans.items()},
            "global_layers": global_layer_indices,
            "evaluations": [],
        }

        for label, mode, beta, eta, v_gamma, swa_shield, head_map, h_map in configs:
            controller.mode = mode
            controller.beta = beta
            controller.eta = eta
            controller.vertical_gamma = v_gamma
            controller.swa_shield_beta = swa_shield
            controller.target_heads = head_map
            controller.layer_horizons = h_map

            with torch.no_grad():
                f_out = model(**fluency_inputs, labels=fluency_labels, use_cache=False)
                ppl = math.exp(f_out.loss.item())
                del f_out

            for hop_name in ["1-Hop", "2-Hop", "3-Hop", "4-Hop", "5-Hop"]:
                t0 = time.time()
                controller.clear_telemetry()
                inp = tokenized_prompts[hop_name]

                with torch.no_grad():
                    out = model(**inp, use_cache=False, logits_to_keep=1)
                    logits = out.logits[0, -1, :].float()
                    probs = torch.softmax(logits, dim=-1)
                    del out

                p_target = probs[target_tok].item() * 100.0
                p_omega = probs[omega_tok].item() * 100.0
                p_trap = probs[trap_tok].item() * 100.0

                vault_dist = {
                    v_label: round(probs[v_tok].item() * 100.0, 4)
                    for v_label, v_tok in vault_tok_map.items()
                }
                vault_probs = [(tok, probs[tok].item()) for tok in all_vault_toks]
                vault_probs.sort(key=lambda x: x[1], reverse=True)
                vault_rank = [idx + 1 for idx, (tok, _) in enumerate(vault_probs) if tok == target_tok][0]
                top1_tok_str = tokenizer.decode([vault_probs[0][0]]).strip()

                l29_p = controller.logit_lens_trajectory.get("L29", {}).get("target_79421_pct", 0.0)
                l35_p = controller.logit_lens_trajectory.get("L35", {}).get("target_79421_pct", 0.0)
                l41_p = controller.logit_lens_trajectory.get("L41", {}).get("target_79421_pct", 0.0)
                dt = time.time() - t0

                rank_str = f"#{vault_rank} ({'WIN' if vault_rank == 1 else top1_tok_str})"
                print(
                    f"{label:<52} | {hop_name:<6} | {ppl:<6.2f} | {p_target:>11.2f}% | "
                    f"{p_omega:>10.2f}% | {rank_str:<9} | {l29_p:>7.2f}% | {l35_p:>7.2f}% | {l41_p:>7.2f}% ({dt:.1f}s)",
                    flush=True,
                )

                results["evaluations"].append({
                    "configuration": label,
                    "hop_task": hop_name,
                    "fluency_ppl": round(ppl, 4),
                    "target_prob_pct": round(p_target, 4),
                    "decoy_omega_prob_pct": round(p_omega, 4),
                    "recency_trap_prob_pct": round(p_trap, 4),
                    "vault_rank": vault_rank,
                    "top1_vault_token": top1_tok_str,
                    "vault_probability_distribution": vault_dist,
                    "logit_lens_trajectory": dict(controller.logit_lens_trajectory),
                    "vertical_gap_telemetry": dict(controller.vertical_gap_stats),
                    "per_layer_zone_attn_pct": dict(controller.per_layer_zone_attn),
                    "elapsed_sec": round(dt, 2),
                })

                with open(args.output_json, "w") as f:
                    json.dump(results, f, indent=2)
            print("-" * 156)

        controller.remove_hooks()
        log_heartbeat("STAGE2_PHASE6_SUITE", f"Saved complete Phase 6 telemetry to {args.output_json}")


if __name__ == "__main__":
    main()
