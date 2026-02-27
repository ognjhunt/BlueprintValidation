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

if [[ -n "$DOCKERHUB_IMAGE" ]]; then
  IMAGE_REF="$DOCKERHUB_IMAGE:$TAG"
else
  IMAGE_REF="blueprint-validation:$TAG"
fi

echo "Building snapshot image:"
echo "  Dockerfile: $DOCKERFILE"
echo "  Image:      $IMAGE_REF"

docker build -f "$ROOT_DIR/$DOCKERFILE" -t "$IMAGE_REF" "$ROOT_DIR"

if [[ "$PUSH" == "true" ]]; then
  if [[ -z "$DOCKERHUB_IMAGE" ]]; then
    echo "ERROR: Set DOCKERHUB_IMAGE when PUSH=true" >&2
    exit 1
  fi
  echo "Pushing $IMAGE_REF"
  docker push "$IMAGE_REF"
fi

echo "Done: $IMAGE_REF"
