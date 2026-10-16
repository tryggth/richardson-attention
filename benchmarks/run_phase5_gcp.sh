#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/tryggth2009/hex/venv/bin:$PATH"
WORKDIR="/home/tryggth2009/.gemini/antigravity/scratch/richardson_gemma4_gcp"
INSTANCE_NAME="lazarus-phase5-$(date +%s)"
SELECTED_ZONE=""

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" | tr -d '[:space:]')"
fi

cleanup() {
    echo ""
    echo "[$(date +%H:%M:%S)] [TEARDOWN] Recovering Phase 5 artifacts & deleting ${INSTANCE_NAME}..."
    if [[ -n "${SELECTED_ZONE}" ]]; then
        gcloud compute scp "${INSTANCE_NAME}:~/richardson_phase5_results.json" \
            "${WORKDIR}/richardson_phase5_results.json" \
            --zone="${SELECTED_ZONE}" --quiet 2>/dev/null || true

        gcloud compute instances delete "${INSTANCE_NAME}" \
            --zone="${SELECTED_ZONE}" --quiet || true
        echo "[$(date +%H:%M:%S)] [TEARDOWN] Confirmed ${INSTANCE_NAME} deleted. Billing stopped."
    fi
}
trap cleanup EXIT INT TERM

python3 "${WORKDIR}/gemma4_phase5_rg.py" --self_test

specs=(
    "a2-highgpu-1g:us-central1-a"
    "a2-highgpu-1g:us-east4-a"
    "g2-standard-24:us-east4-a"
    "g2-standard-24:us-central1-a"
    "g2-standard-8:us-central1-a"
    "g2-standard-8:us-east4-a"
    "g2-standard-8:us-west1-a"
)

for spec in "${specs[@]}"; do
    mtype="${spec%%:*}"
    z="${spec##*:}"
    echo "[$(date +%H:%M:%S)] [PROVISION] Attempting ${mtype} in ${z}..."
    if gcloud compute instances create "${INSTANCE_NAME}" \
        --zone="${z}" \
        --machine-type="${mtype}" \
        --provisioning-model=SPOT \
        --instance-termination-action=DELETE \
        --maintenance-policy=TERMINATE \
        --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=100GB \
        --boot-disk-type=pd-ssd \
        --metadata="install-nvidia-driver=True,startup-script=shutdown -h +25" \
        --quiet 2>/dev/null; then
        SELECTED_ZONE="${z}"
        echo "[$(date +%H:%M:%S)] [PROVISION] Allocated ${mtype} in ${SELECTED_ZONE}!"
        break
    fi
done

if [[ -z "${SELECTED_ZONE}" ]]; then
    echo "ERROR: Could not allocate Spot GPU."
    exit 1
fi

for i in {1..36}; do
    if gcloud compute ssh "${INSTANCE_NAME}" --zone="${SELECTED_ZONE}" --quiet \
        --command="nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null; then
        break
    fi
    sleep 5
done

gcloud compute scp "${WORKDIR}/gemma4_phase5_rg.py" "${INSTANCE_NAME}:~/gemma4_phase5_rg.py" \
    --zone="${SELECTED_ZONE}" --quiet

gcloud compute ssh "${INSTANCE_NAME}" --zone="${SELECTED_ZONE}" --quiet --command="
    set -euo pipefail
    export HF_TOKEN='${HF_TOKEN}'
    export HF_HUB_ENABLE_HF_TRANSFER=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    sudo rm -rf /usr/local/lib/python3.10/dist-packages/torchaudio* /usr/local/lib/python3.10/dist-packages/torchvision* || true
    pip install -q --upgrade-strategy only-if-needed -U 'transformers>=5.5.0' accelerate hf_transfer

    python3 -c \"
from huggingface_hub import snapshot_download
print('[HEARTBEAT] Fetching google/gemma-4-E4B-it...', flush=True)
snapshot_download('google/gemma-4-E4B-it')
\"

    python3 ~/gemma4_phase5_rg.py \
        --model_id google/gemma-4-E4B-it \
        --target_tokens 8000 \
        --output_json ~/richardson_phase5_results.json
"

gcloud compute scp "${INSTANCE_NAME}:~/richardson_phase5_results.json" \
    "${WORKDIR}/richardson_phase5_results.json" \
    --zone="${SELECTED_ZONE}" --quiet 2>/dev/null || true
cat "${WORKDIR}/richardson_phase5_results.json" || true
