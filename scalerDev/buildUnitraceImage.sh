#!/usr/bin/env bash
# Build the llm-scaler-vllm dev image WITH unitrace (see Dockerfile.unitrace).
# Derives from an existing Dockerfile.dev image, so vLLM/kernels are not rebuilt;
# only unitrace is compiled (in a throwaway stage off intel/omix).
# Nothing is COPYed from the build context -> an empty dir is passed as context.
# Proxies are the lowercase ones on purpose: that is the pair scaler-dev:latest
# was built with, and the unitrace stage has to reach github.com.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_IMAGE=${BASE_IMAGE:-scaler-dev:latest}
IMAGE_TAG=${IMAGE_TAG:-scaler-dev-unitrace:latest}
# Minimal context holding just the two files the Dockerfile COPYs into /vllm.
# Passing `.` instead would ship the cwd -- 17G from here (.env + ws).
CTX=$(mktemp -d)
trap 'rm -rf "${CTX}"' EXIT
cp "${ROOT}/38uniPerf.sh" "${ROOT}/qwen38-27B.py" "${CTX}/"

docker buildx build \
    --load \
    --progress=plain \
    -f "${ROOT}/Dockerfile.unitrace" \
    -t "${IMAGE_TAG}" \
    --build-arg http_proxy="${http_proxy}" \
    --build-arg https_proxy="${https_proxy}" \
    --build-arg no_proxy="${no_proxy}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg PTI_GPU_COMMIT=d669f3b931ca2a190bde11baa39f548eb364570d \
    --build-arg BUILD_WITH_MPI=1 \
    "${CTX}"
