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


class Phase7StandardBenchmarkController:
    def __init__(self, text_cfg, layers, calib_horizon: float = 1400.0, q_chunk_size: int = 2048):
        self.mode = "baseline"
        self.beta = 3.5
        self.eta = 0.75
        self.swa_shield_beta = 0.0
        self.query_slice_only = True
        self.query_window_len = 64
        self.plateau_power = 3.0
        self.calib_horizon = calib_horizon
        self.q_chunk_size = q_chunk_size
        self.target_heads = {}
        self.layer_horizons = {}
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

            c_l = self.layer_horizons.get(layer_idx, 0.54)
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


# Diverse natural-language essay/literature paragraphs for realistic benchmark haystacks
PROSE_PARAGRAPHS = [
    "In the history of maritime navigation, determining longitude at sea required mechanical timekeepers capable of maintaining accuracy despite violent pitching, humidity shifts, and thermal expansion across ocean voyages. ",
    "Classical thermodynamics emerged from practical inquiries into steam engine efficiency, eventually crystallizing into universal conservation laws and the statistical interpretation of entropy over phase space ensembles. ",
    "Architectural acoustics in nineteenth-century concert halls relied on empirical ratios of volume to seating surface, long before Sabinian reverberation equations formalized sound absorption across porous plaster walls. ",
    "Alpine glaciology tracks seasonal mass balance by measuring snowpack accumulation in high accumulation cirques against summer ablation rates near the terminal moraine, revealing long-term climatic oscillations. ",
    "Early telegraph networks faced severe signal distortion across submarine cables due to distributed capacitance, prompting the development of mirror galvanometers and mathematical transmission line theory. ",
    "Botanical taxonomy transitioned from morphological classification toward phylogenetic trees as comparative genomics uncovered deep evolutionary divergences concealed beneath convergent leaf and floral structures. ",
]


def build_prose_block(num_sentences: int, seed_offset: int) -> str:
    return "".join(PROSE_PARAGRAPHS[(i + seed_offset) % len(PROSE_PARAGRAPHS)] for i in range(num_sentences))


def generate_ruler_vt_instance(seed: int):
    """
    NVIDIA RULER `VT` (Multi-Hop Variable Tracking) protocol (Hsieh et al., COLM 2024):
    - 8 concurrent variable chains (K=8), each starting with a distinct leading digit ('1'..'8')
      so 1st-token BPE evaluation is 100% exact and unbiased across seeds.
    - 3 variable binding hops (Hop 1: VAR_0 -> VALUE, Hop 2: VAR_1 -> VAR_0, Hop 3: VAR_2 -> VAR_1)
      embedded inside natural prose separated by >1,400 tokens.
    """
    rng = random.Random(seed)
    leading_digits = ["1", "2", "3", "4", "5", "6", "7", "8"]
    rng.shuffle(leading_digits)

    values = [d + f"{rng.randint(1000, 9999)}" for d in leading_digits]
    var_tier1 = [f"REG_ALPHA_{i}" for i in range(8)]
    var_tier2 = [f"PTR_BETA_{i}" for i in range(8)]
    var_tier3 = [f"HANDLE_GAMMA_{i}" for i in range(8)]

    perm1 = list(range(8))
    perm2 = list(range(8))
    perm3 = list(range(8))
    rng.shuffle(perm1)
    rng.shuffle(perm2)
    rng.shuffle(perm3)

    target_chain_idx = seed % 8
    decoy_chain_idx = (target_chain_idx + 1 + (seed // 8)) % 8

    target_val = values[target_chain_idx]
    decoy_val = values[decoy_chain_idx]
    target_v1 = var_tier1[target_chain_idx]
    target_v2 = var_tier2[target_chain_idx]
    target_v3 = var_tier3[target_chain_idx]
    decoy_v1 = var_tier1[decoy_chain_idx]
    decoy_v2 = var_tier2[decoy_chain_idx]

    # Zone 1: Variable Initialization (Early Context)
    z1_lines = ["=== RULER VT BLOCK 1: INITIAL VARIABLE BINDINGS ==="]
    for i in perm1:
        z1_lines.append(f"VARIABLE {var_tier1[i]} = {values[i]} ;")
    z1_text = "\n".join(z1_lines) + "\n\n"

    # Zone 2: Pointer Reference Hop 1 (Mid-Early Context)
    z2_lines = ["\n=== RULER VT BLOCK 2: POINTER ASSIGNMENTS ==="]
    for i in perm2:
        z2_lines.append(f"VARIABLE {var_tier2[i]} = {var_tier1[i]} ;")
    z2_text = "\n".join(z2_lines) + "\n\n"

    # Zone 3: Handle Reference Hop 2 (Mid Context)
    z3_lines = ["\n=== RULER VT BLOCK 3: HANDLE ALIASING ==="]
    for i in perm3:
        z3_lines.append(f"VARIABLE {var_tier3[i]} = {var_tier2[i]} ;")
    z3_text = "\n".join(z3_lines) + "\n\n"

    b1 = build_prose_block(45, seed * 3)
    b2 = build_prose_block(45, seed * 7)

    # Late context prose with unverified debug comments (distractor interference)
    debug_note = (
        f"\n[DEBUG TRACE COMMENT - NON-BINDING] Temporary shadow check compared {target_v3} "
        f"against fallback register {decoy_v2} and {decoy_v1} (test value {decoy_val}).\n"
    )
    b3_chunks = []
    for k in range(10):
        b3_chunks.append(build_prose_block(9, seed + k))
        b3_chunks.append(debug_note)
    b3_text = "".join(b3_chunks)

    query = (
        f"\nQuestion: Trace the exact variable assignment chain from {target_v3} through "
        f"the pointer and initial variable bindings above (ignoring non-binding debug trace comments). "
        f"What is the exact 5-digit integer value bound to {target_v3}?\n"
        f"Answer: The 5-digit integer value bound to {target_v3} is "
    )

    prompt = z1_text + b1 + z2_text + b2 + z3_text + b3_text + query
    return prompt, target_val, decoy_val, values


def generate_babilong_qa3_instance(seed: int):
    """
    BABILong `qa3` (3-Supporting-Facts Relational World-State Tracking in Natural Text, Kuratov et al., NeurIPS 2024):
    - Fact Set 1 (Zone A): 8 Artifacts sealed inside 8 numbered lockers (distinct leading digits '1'..'8').
    - Fact Set 2 (Zone B): 8 Couriers retrieve and transport the 8 Artifacts.
    - Fact Set 3 (Zone C): 8 Expedition Leads assume command over the 8 Couriers.
    - Late Narrative Distractors: Routine inspection notes where the queried Expedition Lead inspects a decoy artifact/locker.
    """
    rng = random.Random(seed + 500)
    leading_digits = ["1", "2", "3", "4", "5", "6", "7", "8"]
    rng.shuffle(leading_digits)
    lockers = [d + f"{rng.randint(1000, 9999)}" for d in leading_digits]

    artifacts = [
        "Crystalline-Astrolabe", "Obsidian-Monolith", "Platinum-Chronometer", "Beryllium-Gyroscope",
        "Sapphire-Interferometer", "Titanium-Sextant", "Iridium-Compass", "Quartz-Magnetometer",
    ]
    couriers = [
        "Courier-Elena", "Courier-Marcus", "Courier-Soren", "Courier-Nadia",
        "Courier-Julian", "Courier-Freya", "Courier-Damon", "Courier-Linnea",
    ]
    directors = [
        "Commander-Vance", "Commander-Rostova", "Commander-Thorne", "Commander-Mercer",
        "Commander-Sterling", "Commander-Kael", "Commander-Chen", "Commander-Webb",
    ]

    p1, p2, p3 = list(range(8)), list(range(8)), list(range(8))
    rng.shuffle(p1)
    rng.shuffle(p2)
    rng.shuffle(p3)

    t_idx = seed % 8
    d_idx = (t_idx + 2 + (seed // 8)) % 8

    target_locker = lockers[t_idx]
    decoy_locker = lockers[d_idx]
    target_dir = directors[t_idx]
    decoy_art = artifacts[d_idx]
    decoy_cour = couriers[d_idx]

    f1_lines = ["=== ARCHIVE LOG PART 1: PERMANENT ARTIFACT LOCKER SEALING ==="]
    for i in p1:
        f1_lines.append(f"- The {artifacts[i]} is permanently sealed inside Locker {lockers[i]}.")
    f1_text = "\n".join(f1_lines) + "\n\n"

    f2_lines = ["\n=== ARCHIVE LOG PART 2: COURIER CUSTODY DISPATCH ==="]
    for i in p2:
        f2_lines.append(f"- {couriers[i]} took custody of the {artifacts[i]} for primary transport.")
    f2_text = "\n".join(f2_lines) + "\n\n"

    f3_lines = ["\n=== ARCHIVE LOG PART 3: EXPEDITION COMMAND ROSTER ==="]
    for i in p3:
        f3_lines.append(f"- {directors[i]} holds exclusive command over {couriers[i]}.")
    f3_text = "\n".join(f3_lines) + "\n\n"

    b1 = build_prose_block(45, seed * 5)
    b2 = build_prose_block(45, seed * 11)

    inspection_decoy = (
        f"\n[TEMPORARY FIELD NOTE - NON-CUSTODIAL] During a routine walk-through, {target_dir} "
        f"briefly visually inspected {decoy_cour}'s {decoy_art} (Locker {decoy_locker}) without changing custody.\n"
    )
    b3_chunks = []
    for k in range(10):
        b3_chunks.append(build_prose_block(9, seed * 2 + k))
        b3_chunks.append(inspection_decoy)
    b3_text = "".join(b3_chunks)

    query = (
        f"\nQuestion: Based strictly on the Permanent Artifact Locker Sealing, Courier Custody Dispatch, "
        f"and Expedition Command Roster (disregarding non-custodial field inspection notes), which 5-digit "
        f"Locker number permanently holds the artifact transported by the courier under {target_dir}?\n"
        f"Answer: The 5-digit Locker number is "
    )

    prompt = f1_text + b1 + f2_text + b2 + f3_text + b3_text + query
    return prompt, target_locker, decoy_locker, lockers


def generate_lost_in_the_middle_instance(depth_pct: int, seed: int = 42):
    """
    Liu et al. (TACL 2024) "Lost in the Middle" Depth Sweep:
    Places Zone A at the start (d/T ~ 0.99) and sweeps the crucial 2nd/3rd hop registries
    to exact fractional document depth `depth_pct` in {15%, 30%, 45%, 60%, 75%}.
    """
    rng = random.Random(seed)
    leading_digits = ["1", "2", "3", "4", "5", "6", "7", "8"]
    lockers = ["79421", "63819", "49281", "54712", "83914", "37192", "28491", "91824"]

    zone_a = (
        "=== DOCUMENT 0: TIER-0 VAULT CODE INDEX ===\n"
        "- VAULT-ALPHA: 49281 | VAULT-HELIOS: 54712 | VAULT-HYPERION: 83914 | VAULT-LAZARUS: 79421\n"
        "- VAULT-NYX: 37192 | VAULT-OMEGA: 63819 | VAULT-VULCAN: 28491 | VAULT-ZEPHYR: 91824\n\n"
    )
    bridge_doc = (
        "\n=== CRITICAL ROUTING DOCUMENT (AUTHORITATIVE) ===\n"
        "- SECTOR-KRONOS is assigned primary vault VAULT-LAZARUS.\n"
        "- DIRECTOR-VANCE commands SECTOR-KRONOS.\n\n"
    )
    decoy_note = (
        "\n[ROUTINE LOG] Temporary cross-check: DIRECTOR-VANCE reviewed SECTOR-ATLAS and VAULT-OMEGA (63819).\n"
    )

    total_prose_blocks = 160
    pre_blocks = max(10, int(total_prose_blocks * (depth_pct / 100.0)))
    post_blocks = total_prose_blocks - pre_blocks

    pre_text = build_prose_block(pre_blocks, 1)
    post_chunks = []
    for k in range(post_blocks):
        post_chunks.append(PROSE_PARAGRAPHS[k % len(PROSE_PARAGRAPHS)])
        if k >= post_blocks - 36 and k % 3 == 0:
            post_chunks.append(decoy_note)
    post_text = "".join(post_chunks)

    query = (
        "\nQuestion: Based on the Tier-0 Vault Code Index and the Authoritative Critical Routing Document, "
        "what is the 5-digit code for the primary vault under DIRECTOR-VANCE?\n"
        "Answer: The 5-digit code is "
    )
    return zone_a + pre_text + bridge_doc + post_text + query, "79421", "63819", lockers


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
        for s in range(5):
            p_vt, t_vt, d_vt, all_vt = generate_ruler_vt_instance(s)
            assert t_vt in p_vt and d_vt in p_vt and len({v[0] for v in all_vt}) == 8
            p_qa, t_qa, d_qa, all_qa = generate_babilong_qa3_instance(s)
            assert t_qa in p_qa and d_qa in p_qa and len({v[0] for v in all_qa}) == 8
        for dp in [15, 30, 45, 60, 75]:
            p_lm, t_lm, d_lm, _ = generate_lost_in_the_middle_instance(dp)
            assert t_lm in p_lm and d_lm in p_lm
        log_heartbeat("LOCAL_SELF_TEST", "ALL RULER-VT, BABILONG-QA3, AND LOST-IN-THE-MIDDLE GENERATORS PASSED!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-E4B-it")
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--output_json", type=str, default="results/richardson_phase7_standard_benchmarks.json")
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

    controller = Phase7StandardBenchmarkController(text_cfg, layers, calib_horizon=calib_horizon)
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

        global_layers = sorted(by_layer.keys())
        quorum2_map = build_quorum_head_map(by_layer, heads_per_layer=2, calib_horizon=calib_horizon)
        quorum4_map = build_quorum_head_map(by_layer, heads_per_layer=4, calib_horizon=calib_horizon)

        unif_horizons = {l_idx: 0.45 for l_idx in global_layers}
        rg_horizons = {
            5: 0.42, 11: 0.45, 17: 0.50,
            23: 0.62, 29: 0.70,
            35: 0.82, 41: 0.88,
        }

    # (label, mode, beta, eta, swa_shield, head_map, horizon_map)
    configs = [
        ("1. Unmodified Baseline", "baseline", 0.0, 0.0, 0.0, quorum2_map, unif_horizons),
        ("2. Query-Slice Cubic Plateau (Q14, b=3.5)", "dual_cubic_plateau", 3.5, 0.75, 0.0, quorum2_map, unif_horizons),
        ("3. Cubic Q14 + SWA Decoy Shield (sw=2.5)", "dual_cubic_plateau", 3.5, 0.75, 2.5, quorum2_map, unif_horizons),
        ("4. Stepped 7-Layer RG Relay (Q28, b=3.5)", "dual_cubic_plateau", 3.5, 0.75, 2.0, quorum4_map, rg_horizons),
    ]

    results = {
        "model_id": args.model_id,
        "num_seeds": args.num_seeds,
        "ruler_vt_summary": [],
        "babilong_qa3_summary": [],
        "lost_in_the_middle_sweep": [],
        "per_seed_trials": {"ruler_vt": [], "babilong_qa3": []},
    }

    # Pre-tokenize instances so tokenization overhead is zero during the sweep
    ruler_instances = []
    babilong_instances = []
    for s in range(args.num_seeds):
        p_vt, t_vt, d_vt, all_vt = generate_ruler_vt_instance(100 + s)
        ruler_instances.append((
            s,
            tokenizer(bos + p_vt, return_tensors="pt").to(device),
            tokenizer.encode(t_vt, add_special_tokens=False)[0],
            tokenizer.encode(d_vt, add_special_tokens=False)[0],
            [tokenizer.encode(v, add_special_tokens=False)[0] for v in all_vt],
        ))
        p_qa, t_qa, d_qa, all_qa = generate_babilong_qa3_instance(200 + s)
        babilong_instances.append((
            s,
            tokenizer(bos + p_qa, return_tensors="pt").to(device),
            tokenizer.encode(t_qa, add_special_tokens=False)[0],
            tokenizer.encode(d_qa, add_special_tokens=False)[0],
            [tokenizer.encode(v, add_special_tokens=False)[0] for v in all_qa],
        ))

    litm_instances = []
    for dp in [15, 30, 45, 60, 75]:
        p_lm, t_lm, d_lm, all_lm = generate_lost_in_the_middle_instance(dp)
        litm_instances.append((
            dp,
            tokenizer(bos + p_lm, return_tensors="pt").to(device),
            tokenizer.encode(t_lm, add_special_tokens=False)[0],
            tokenizer.encode(d_lm, add_special_tokens=False)[0],
            [tokenizer.encode(v, add_special_tokens=False)[0] for v in all_lm],
        ))

    with StageWatchdog("STAGE2_STANDARD_BENCHMARKS", 900):
        print("\n" + "=" * 136)
        print(f"PHASE 7A: NVIDIA RULER `VT` (VARIABLE TRACKING) & BABILONG `qa3` ({args.num_seeds} RANDOMIZED SEEDS EACH)")
        print("=" * 136)
        print(f"{'Benchmark Suite':<22} | {'Configuration':<44} | {'Top-1 Acc':<11} | {'Mean Target Prob':<18} | {'Mean Decoy Err'}")
        print("-" * 136)

        for suite_name, inst_list, summary_key, trial_key in [
            ("RULER-VT (3-Hop)", ruler_instances, "ruler_vt_summary", "ruler_vt"),
            ("BABILong-qa3 (3-Hop)", babilong_instances, "babilong_qa3_summary", "babilong_qa3"),
        ]:
            for label, mode, beta, eta, sw_shield, head_map, h_map in configs:
                controller.mode = mode
                controller.beta = beta
                controller.eta = eta
                controller.swa_shield_beta = sw_shield
                controller.target_heads = head_map
                controller.layer_horizons = h_map

                wins = 0
                t_probs = []
                d_probs = []
                t0 = time.time()

                for s_idx, inp, t_tok, d_tok, all_toks in inst_list:
                    with torch.no_grad():
                        out = model(**inp, use_cache=False, logits_to_keep=1)
                        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
                        del out

                    pt = probs[t_tok].item() * 100.0
                    pd = probs[d_tok].item() * 100.0
                    c_scores = sorted([(tok, probs[tok].item()) for tok in all_toks], key=lambda x: x[1], reverse=True)
                    rank = [idx + 1 for idx, (tok, _) in enumerate(c_scores) if tok == t_tok][0]
                    if rank == 1:
                        wins += 1
                    t_probs.append(pt)
                    d_probs.append(pd)

                    results["per_seed_trials"][trial_key].append({
                        "seed": s_idx,
                        "configuration": label,
                        "context_tokens": int(inp.input_ids.shape[1]),
                        "target_prob_pct": round(pt, 4),
                        "decoy_prob_pct": round(pd, 4),
                        "rank": rank,
                    })

                acc_pct = (wins / len(inst_list)) * 100.0
                mean_t = float(np.mean(t_probs))
                std_t = float(np.std(t_probs))
                mean_d = float(np.mean(d_probs))
                dt = time.time() - t0

                print(
                    f"{suite_name:<22} | {label:<44} | {acc_pct:>8.1f}%   | "
                    f"{mean_t:>7.2f}% ± {std_t:<5.2f}% | {mean_d:>9.2f}% ({dt:.1f}s)",
                    flush=True,
                )

                results[summary_key].append({
                    "configuration": label,
                    "top1_accuracy_pct": round(acc_pct, 2),
                    "mean_target_prob_pct": round(mean_t, 4),
                    "std_target_prob_pct": round(std_t, 4),
                    "mean_decoy_prob_pct": round(mean_d, 4),
                    "elapsed_sec": round(dt, 2),
                })

                with open(args.output_json, "w") as f:
                    json.dump(results, f, indent=2)
            print("-" * 136)

        print("\n" + "=" * 136)
        print("PHASE 7B: 'LOST IN THE MIDDLE' MULTI-DOCUMENT DEPTH SWEEP (15% TO 75% DOCUMENT DEPTH)")
        print("=" * 136)
        print(f"{'Configuration':<44} | {'15% Depth':<12} | {'30% Depth':<12} | {'45% Depth':<12} | {'60% Depth':<12} | {'75% Depth'}")
        print("-" * 136)

        for label, mode, beta, eta, sw_shield, head_map, h_map in configs:
            controller.mode = mode
            controller.beta = beta
            controller.eta = eta
            controller.swa_shield_beta = sw_shield
            controller.target_heads = head_map
            controller.layer_horizons = {l_idx: 0.20 for l_idx in global_layers} if mode != "baseline" else unif_horizons

            depth_scores = {}
            for dp, inp, t_tok, d_tok, all_toks in litm_instances:
                with torch.no_grad():
                    out = model(**inp, use_cache=False, logits_to_keep=1)
                    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
                    del out
                depth_scores[f"{dp}%"] = round(probs[t_tok].item() * 100.0, 4)

            print(
                f"{label:<44} | {depth_scores['15%']:>9.2f}%  | {depth_scores['30%']:>9.2f}%  | "
                f"{depth_scores['45%']:>9.2f}%  | {depth_scores['60%']:>9.2f}%  | {depth_scores['75%']:>9.2f}%",
                flush=True,
            )
            results["lost_in_the_middle_sweep"].append({
                "configuration": label,
                "depth_target_prob_pct": depth_scores,
            })
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)

        log_heartbeat("STAGE2_STANDARD_BENCHMARKS", f"Saved Phase 7 benchmark results to {args.output_json}")


if __name__ == "__main__":
    main()
