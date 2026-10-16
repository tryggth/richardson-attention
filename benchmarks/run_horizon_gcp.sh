#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/tryggth2009/hex/venv/bin:$PATH"
WORKDIR="/home/tryggth2009/.gemini/antigravity/scratch/richardson_gemma4_gcp"
INSTANCE_NAME="lazarus-horizon-$(date +%s)"
MACHINE_TYPE="g2-standard-8"
candidate_zones=("us-central1-a" "us-central1-b" "us-east4-a" "us-west1-a" "us-west1-b")
SELECTED_ZONE=""

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" | tr -d '[:space:]')"
fi

cleanup() {
    echo ""
    echo "[$(date +%H:%M:%S)] [TEARDOWN] Recovering artifacts & deleting Spot L4 ${INSTANCE_NAME}..."
    if [[ -n "${SELECTED_ZONE}" ]]; then
        gcloud compute scp "${INSTANCE_NAME}:~/richardson_horizon_results.json" \
            "${WORKDIR}/richardson_horizon_results.json" \
            --zone="${SELECTED_ZONE}" --quiet 2>/dev/null || true

        gcloud compute instances delete "${INSTANCE_NAME}" \
            --zone="${SELECTED_ZONE}" --quiet || true
        echo "[$(date +%H:%M:%S)] [TEARDOWN] Confirmed ${INSTANCE_NAME} deleted. Billing stopped."
    fi
}
trap cleanup EXIT INT TERM

python3 "${WORKDIR}/gemma4_horizon_breaker.py" --self_test

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
        echo "[$(date +%H:%M:%S)] [PROVISION] Spot L4 ${INSTANCE_NAME} live in ${SELECTED_ZONE}."
        break
    fi
done

if [[ -z "${SELECTED_ZONE}" ]]; then
    echo "ERROR: Could not allocate Spot L4 across candidate zones."
    exit 1
fi

for i in {1..36}; do
    if gcloud compute ssh "${INSTANCE_NAME}" --zone="${SELECTED_ZONE}" --quiet \
        --command="nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null; then
        break
    fi
    sleep 5
done

gcloud compute scp "${WORKDIR}/gemma4_horizon_breaker.py" "${INSTANCE_NAME}:~/gemma4_horizon_breaker.py" \
    --zone="${SELECTED_ZONE}" --quiet

gcloud compute ssh "${INSTANCE_NAME}" --zone="${SELECTED_ZONE}" --quiet --command="
    set -euo pipefail
    export HF_TOKEN='${HF_TOKEN}'
    export HF_HUB_ENABLE_HF_TRANSFER=1

    sudo rm -rf /usr/local/lib/python3.10/dist-packages/torchaudio* /usr/local/lib/python3.10/dist-packages/torchvision* || true
    pip install -q --upgrade-strategy only-if-needed -U 'transformers>=5.5.0' accelerate hf_transfer

    python3 -c \"
from huggingface_hub import snapshot_download
for repo in ['google/gemma-4-E2B-it', 'google/gemma-4-E4B-it']:
    print(f'[HEARTBEAT] Fetching {repo}...', flush=True)
    snapshot_download(repo)
\"

    python3 ~/gemma4_horizon_breaker.py \
        --models google/gemma-4-E2B-it google/gemma-4-E4B-it \
        --context_lengths 4096 8192 12288 \
        --output_json ~/richardson_horizon_results.json
"

gcloud compute scp "${INSTANCE_NAME}:~/richardson_horizon_results.json" \
    "${WORKDIR}/richardson_horizon_results.json" \
    --zone="${SELECTED_ZONE}" --quiet 2>/dev/null || true
cat "${WORKDIR}/richardson_horizon_results.json" || true
