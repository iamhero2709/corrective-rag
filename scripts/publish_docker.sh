#!/bin/bash
# Build and publish Docker image
# Usage: ./scripts/publish_docker.sh [test|prod]

set -e

MODE=${1:-test}
IMAGE_NAME="randhir-kumar/corrective-rag"
VERSION="0.3.0"

echo "Building Docker image: $IMAGE_NAME:$VERSION"

# Build
docker build -t $IMAGE_NAME:$VERSION -t $IMAGE_NAME:latest .

echo ""
echo "Image built successfully:"
docker images | grep $IMAGE_NAME

if [ "$MODE" = "prod" ]; then
    echo ""
    echo "Pushing to Docker Hub..."
    docker push $IMAGE_NAME:$VERSION
    docker push $IMAGE_NAME:latest
    echo ""
    echo "✓ Published to Docker Hub!"
    echo "Pull with: docker pull $IMAGE_NAME:$VERSION"
else
    echo ""
    echo "Test mode - not pushing"
    echo ""
    echo "To push to Docker Hub:"
    echo "  docker push $IMAGE_NAME:$VERSION"
    echo "  docker push $IMAGE_NAME:latest"
    echo ""
    echo "To run locally:"
    echo "  docker run -p 8000:8000 $IMAGE_NAME:$VERSION"
fi
