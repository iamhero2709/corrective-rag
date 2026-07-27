#!/bin/bash
# Record demo with asciinema and convert to GIF
# Usage: ./scripts/record_demo.sh

set -e

echo "Recording demo..."
echo "This will run the demo script and record it."
echo "Press Ctrl+D when done."
echo ""

# Check if asciinema is installed
if ! command -v asciinema &> /dev/null; then
    echo "Error: asciinema not installed"
    echo "Install with: sudo apt install asciinema"
    echo "Or use: pip install asciinema"
    exit 1
fi

# Record
asciinema rec demo.cast -c "./scripts/demo.sh"

echo ""
echo "Recording saved to demo.cast"
echo ""
echo "To convert to GIF:"
echo "  1. Install agg: pip install agg"
echo "  2. Convert: agg demo.cast demo.gif --cols 80 --rows 24"
echo ""
echo "Or upload to asciinema.org:"
echo "  asciinema upload demo.cast"
