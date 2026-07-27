#!/bin/bash
# Build and publish to PyPI
# Usage: ./scripts/publish_pypi.sh [test|prod]

set -e

MODE=${1:-test}

echo "Building Corrective RAG for PyPI..."

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build
echo "Building package..."
python -m build

echo ""
echo "Build artifacts:"
ls -la dist/

# Check with twine
echo ""
echo "Checking package..."
twine check dist/*

if [ "$MODE" = "prod" ]; then
    echo ""
    echo "Uploading to PyPI..."
    twine upload dist/*
    echo ""
    echo "✓ Published to PyPI!"
    echo "Install with: pip install corrective-rag"
else
    echo ""
    echo "Test mode - not uploading"
    echo ""
    echo "To upload to PyPI:"
    echo "  twine upload dist/*"
    echo ""
    echo "To upload to TestPyPI:"
    echo "  twine upload --repository testpypi dist/*"
fi
