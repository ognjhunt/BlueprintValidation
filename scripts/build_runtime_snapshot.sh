#!/usr/bin/env bash
set -euo pipefail

# Build a Docker runtime snapshot image for BlueprintValidation.
# Optional push:
#   PUSH=true DOCKERHUB_IMAGE=yourname/blueprint-validation bash scripts/build_runtime_snapshot.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="${DOCKERFILE:-docker/runpod.Dockerfile}"
DOCKERHUB_IMAGE="${DOCKERHUB_IMAGE:-}"
PUSH="${PUSH:-false}"
TAG="${TAG:-$(date +%Y%m%d)-$(git -C "$ROOT_DIR" rev-parse --short HEAD)}"
VENDOR_STRATEGY="${VENDOR_STRATEGY:-auto}" # auto|vendored|clone

DREAMDOJO_REPO_URL="${DREAMDOJO_REPO_URL:-https://github.com/NVIDIA/DreamDojo.git}"
DREAMDOJO_REF="${DREAMDOJO_REF:-7f3379bcb831147c0cc170e79ba08471ad186497}"
COSMOS_REPO_URL="${COSMOS_REPO_URL:-https://github.com/nvidia-cosmos/cosmos-transfer2.5.git}"
COSMOS_REF="${COSMOS_REF:-c9ad44b7283613618d57c1e4c9991916907d4f4b}"
OPENVLA_REPO_URL="${OPENVLA_REPO_URL:-https://github.com/moojink/openvla-oft.git}"
OPENVLA_REF="${OPENVLA_REF:-e4287e94541f459edc4feabc4e181f537cd569a8}"

if [[ -n "$DOCKERHUB_IMAGE" ]]; then
  IMAGE_REF="$DOCKERHUB_IMAGE:$TAG"
else
  IMAGE_REF="blueprint-validation:$TAG"
fi

case "$VENDOR_STRATEGY" in
  auto|vendored|clone) ;;
  *)
    echo "ERROR: VENDOR_STRATEGY must be one of: auto|vendored|clone (got '$VENDOR_STRATEGY')." >&2
    exit 1
    ;;
esac

if [[ "$VENDOR_STRATEGY" == "vendored" ]]; then
  missing=0
  for repo in DreamDojo cosmos-transfer openvla-oft; do
    if [[ ! -d "$ROOT_DIR/data/vendor/$repo" ]]; then
      echo "ERROR: missing vendored repo for VENDOR_STRATEGY=vendored: $ROOT_DIR/data/vendor/$repo" >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    echo "Either set VENDOR_STRATEGY=clone (or auto) or provision data/vendor before build." >&2
    exit 1
  fi
fi

echo "Building snapshot image:"
echo "  Dockerfile: $DOCKERFILE"
echo "  Image:      $IMAGE_REF"
echo "  Vendor strategy: $VENDOR_STRATEGY"

DOCKER_BUILDKIT=1 docker build \
  -f "$ROOT_DIR/$DOCKERFILE" \
  -t "$IMAGE_REF" \
  --build-arg VENDOR_STRATEGY="$VENDOR_STRATEGY" \
  --build-arg DREAMDOJO_REPO_URL="$DREAMDOJO_REPO_URL" \
  --build-arg DREAMDOJO_REF="$DREAMDOJO_REF" \
  --build-arg COSMOS_REPO_URL="$COSMOS_REPO_URL" \
  --build-arg COSMOS_REF="$COSMOS_REF" \
  --build-arg OPENVLA_REPO_URL="$OPENVLA_REPO_URL" \
  --build-arg OPENVLA_REF="$OPENVLA_REF" \
  "$ROOT_DIR"

if [[ "$PUSH" == "true" ]]; then
  if [[ -z "$DOCKERHUB_IMAGE" ]]; then
    echo "ERROR: Set DOCKERHUB_IMAGE when PUSH=true" >&2
    exit 1
  fi
  echo "Pushing $IMAGE_REF"
  docker push "$IMAGE_REF"
fi

echo "Done: $IMAGE_REF"
