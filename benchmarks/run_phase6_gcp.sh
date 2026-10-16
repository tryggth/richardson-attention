#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/tryggth2009/hex/venv/bin:$PATH"
WORKDIR="/home/tryggth2009/.gemini/antigravity/scratch/richardson_gemma4_gcp"
INSTANCE_NAME="lazarus-phase6-$(date +%s)"
SELECTED_ZONE=""

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token" | tr -d '[:space:]')"
fi

cleanup() {
    echo ""
    echo "[$(date +%H:%M:%S)] [TEARDOWN] Recovering Phase 6 telemetry & deleting ${INSTANCE_NAME}..."
    if [[ -n "${SELECTED_ZONE}" ]]; then
        gcloud compute scp "${INSTANCE_NAME}:~/richardson_phase6_vertical_results.json" \
            "${WORKDIR}/results/richardson_phase6_vertical_results.json" \
            --zone="${SELECTED_ZONE}" --quiet 2>/dev/null || true

        gcloud compute instances delete "${INSTANCE_NAME}" \
            --zone="${SELECTED_ZONE}" --quiet || true
        echo "[$(date +%H:%M:%S)] [TEARDOWN] Confirmed ${INSTANCE_NAME} deleted. Billing stopped."
    fi
}
trap cleanup EXIT INT TERM

python3 "${WORKDIR}/benchmarks/gemma4_phase6_vertical.py" --self_test

specs=(
    "a2-highgpu-1g:us-central1-a:SPOT"
    "a2-highgpu-1g:us-central1-b:SPOT"
    "a2-highgpu-1g:us-central1-c:SPOT"
    "a2-highgpu-1g:us-central1-f:SPOT"
    "a2-highgpu-1g:us-east4-a:SPOT"
    "a2-highgpu-1g:us-east1-b:SPOT"
    "a2-highgpu-1g:us-west1-b:SPOT"
    "g2-standard-24:us-east4-a:SPOT"
    "g2-standard-24:us-central1-a:SPOT"
    "g2-standard-24:us-central1-b:SPOT"
    "g2-standard-24:us-central1-c:SPOT"
    "g2-standard-8:us-central1-a:SPOT"
    "g2-standard-8:us-central1-b:SPOT"
    "g2-standard-8:us-central1-c:SPOT"
    "g2-standard-8:us-east4-a:SPOT"
    "g2-standard-8:us-east4-b:SPOT"
    "g2-standard-8:us-east4-c:SPOT"
    "g2-standard-8:us-east1-b:SPOT"
    "g2-standard-8:us-east1-c:SPOT"
    "g2-standard-8:us-west1-a:SPOT"
    "g2-standard-8:us-west1-b:SPOT"
    "g2-standard-8:us-west1-c:SPOT"
    "g2-standard-8:us-west4-a:SPOT"
    "g2-standard-8:us-central1-a:STANDARD"
    "g2-standard-8:us-central1-b:STANDARD"
    "g2-standard-8:us-east4-a:STANDARD"
    "g2-standard-8:us-east1-b:STANDARD"
    "g2-standard-8:us-west1-a:STANDARD"
)

for spec in "${specs[@]}"; do
    IFS=':' read -r mtype z pmodel <<< "${spec}"
    echo "[$(date +%H:%M:%S)] [PROVISION] Attempting ${mtype} (${pmodel}) in ${z}..."
    extra_flags=(--maintenance-policy=TERMINATE)
    if [[ "${pmodel}" == "SPOT" ]]; then
        extra_flags+=(--provisioning-model=SPOT --instance-termination-action=DELETE)
    fi
    if gcloud compute instances create "${INSTANCE_NAME}" \
        --zone="${z}" \
        --machine-type="${mtype}" \
        "${extra_flags[@]}" \
        --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=100GB \
        --boot-disk-type=pd-ssd \
        --metadata="install-nvidia-driver=True,startup-script=shutdown -h +30" \
        --quiet 2>/dev/null; then
        SELECTED_ZONE="${z}"
        echo "[$(date +%H:%M:%S)] [PROVISION] Allocated ${mtype} (${pmodel}) in ${SELECTED_ZONE}!"
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

gcloud compute scp "${WORKDIR}/benchmarks/gemma4_phase6_vertical.py" "${INSTANCE_NAME}:~/gemma4_phase6_vertical.py" \
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

    python3 ~/gemma4_phase6_vertical.py \
        --model_id google/gemma-4-E4B-it \
        --output_json ~/richardson_phase6_vertical_results.json
"

gcloud compute scp "${INSTANCE_NAME}:~/richardson_phase6_vertical_results.json" \
    "${WORKDIR}/results/richardson_phase6_vertical_results.json" \
    --zone="${SELECTED_ZONE}" --quiet 2>/dev/null || true
ls -lh "${WORKDIR}/results/richardson_phase6_vertical_results.json"
