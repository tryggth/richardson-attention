#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/tryggth2009/hex/venv/bin:$PATH"
WORKDIR="/home/tryggth2009/.gemini/antigravity/scratch/richardson_gemma4_gcp"
INSTANCE_NAME="lazarus-gemma4-$(date +%s)"
MACHINE_TYPE="g2-standard-8"
candidate_zones=("us-central1-a" "us-central1-b" "us-east4-a" "us-west1-a" "us-west1-b")
SELECTED_ZONE=""

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" | tr -d '[:space:]')"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not found in env or ~/.cache/huggingface/token."
    exit 1
fi

cleanup() {
    echo ""
    echo "[$(date +%H:%M:%S)] [TEARDOWN] Executing fail-safe artifact recovery & GPU teardown for ${INSTANCE_NAME}..."
    if [[ -n "${SELECTED_ZONE}" ]]; then
        # Attempt to pull back incremental or final JSON before deleting VM
        gcloud compute scp "${INSTANCE_NAME}:~/richardson_results.json" "${WORKDIR}/richardson_results.json" \
            --zone="${SELECTED_ZONE}" --quiet 2>/dev/null || true

        gcloud compute instances delete "${INSTANCE_NAME}" \
            --zone="${SELECTED_ZONE}" \
            --quiet || true
        echo "[$(date +%H:%M:%S)] [TEARDOWN] Confirmed ${INSTANCE_NAME} deleted. GPU billing stopped."
    fi
}
trap cleanup EXIT INT TERM

echo "[$(date +%H:%M:%S)] [PRE-FLIGHT] Running local CPU unit test (\$0.00 cost)..."
python3 "${WORKDIR}/gemma4_richardson_test.py" --self_test

for z in "${candidate_zones[@]}"; do
    echo "[$(date +%H:%M:%S)] [PROVISION] Attempting Spot L4 allocation in ${z}..."
    if gcloud compute instances create "${INSTANCE_NAME}" \
        --zone="${z}" \
        --machine-type="${MACHINE_TYPE}" \
        --provisioning-model=SPOT \
        --instance-termination-action=DELETE \
        --maintenance-policy=TERMINATE \
        --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=100GB \
        --boot-disk-type=pd-ssd \
        --metadata="install-nvidia-driver=True,startup-script=shutdown -h +30" \
        --quiet; then
        SELECTED_ZONE="${z}"
        echo "[$(date +%H:%M:%S)] [PROVISION] Spot L4 ${INSTANCE_NAME} live in ${SELECTED_ZONE} (30-min OS dead-man's switch armed)."
        break
    else
        echo "[$(date +%H:%M:%S)] [PROVISION] Zone ${z} unavailable, trying next..."
    fi
done

if [[ -z "${SELECTED_ZONE}" ]]; then
    echo "ERROR: Could not allocate Spot L4 across candidate zones."
    exit 1
fi

echo "[$(date +%H:%M:%S)] [GATE 1] Polling SSH & NVIDIA L4 driver readiness..."
for i in {1..36}; do
    if gcloud compute ssh "${INSTANCE_NAME}" --zone="${SELECTED_ZONE}" --quiet \
        --command="nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] [GATE 1] NVIDIA L4 GPU verified online!"
        break
    fi
    echo "  ...waiting for SSH/GPU (attempt ${i}/36)"
    sleep 5
    if [[ $i -eq 36 ]]; then
        echo "ERROR: Timeout waiting for SSH/GPU readiness."
        exit 1
    fi
done

gcloud compute scp "${WORKDIR}/gemma4_richardson_test.py" "${INSTANCE_NAME}:~/gemma4_richardson_test.py" \
    --zone="${SELECTED_ZONE}" --quiet

echo "[$(date +%H:%M:%S)] [GATE 2-4] Running back-to-back E2B-it + E4B-it validation on Spot L4..."
gcloud compute ssh "${INSTANCE_NAME}" --zone="${SELECTED_ZONE}" --quiet --command="
    set -euo pipefail
    export HF_TOKEN='${HF_TOKEN}'
    export HF_HUB_ENABLE_HF_TRANSFER=1

    echo \"[\$(date +%H:%M:%S)] [ENV] Installing transformers>=5.5.0, accelerate, and hf_transfer...\"
    sudo rm -rf /usr/local/lib/python3.10/dist-packages/torchaudio* /usr/local/lib/python3.10/dist-packages/torchvision* || true
    pip install -q --upgrade-strategy only-if-needed -U 'transformers>=5.5.0' accelerate hf_transfer

    echo \"[\$(date +%H:%M:%S)] [DOWNLOAD] Pulling google/gemma-4-E2B-it and google/gemma-4-E4B-it via datacenter fiber...\"
    python3 -c \"
import time, threading
from huggingface_hub import snapshot_download
done = False
def ticker():
    t0 = time.time()
    while not done:
        print(f'[HEARTBEAT] Downloading checkpoints... elapsed {time.time()-t0:.0f}s', flush=True)
        time.sleep(5)
t = threading.Thread(target=ticker, daemon=True)
t.start()
for repo in ['google/gemma-4-E2B-it', 'google/gemma-4-E4B-it']:
    t_repo = time.time()
    print(f'[HEARTBEAT] Fetching {repo}...', flush=True)
    snapshot_download(repo)
    print(f'[HEARTBEAT] Finished {repo} in {time.time()-t_repo:.1f}s', flush=True)
done = True
\"

    echo \"[\$(date +%H:%M:%S)] [EXPERIMENT] Executing Back-to-Back Gemma 4 Richardson Suite...\"
    python3 ~/gemma4_richardson_test.py \
        --models google/gemma-4-E2B-it google/gemma-4-E4B-it \
        --output_json ~/richardson_results.json
"

gcloud compute scp "${VM_NAME}:~/richardson_results.json" "${WORKDIR}/richardson_results.json" --zone="${ALLOCATED_ZONE}" --quiet 2>/dev/null || true
echo "[$(date +%H:%M:%S)] [COMPLETE] Displaying final results from ${WORKDIR}/richardson_results.json:"
cat "${WORKDIR}/richardson_results.json" || true
