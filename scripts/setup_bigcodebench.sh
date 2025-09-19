#!/bin/bash

# BigCodeBench Dependencies Setup Script
# This script installs all dependencies needed for BigCodeBench evaluation
# Based on eyon's comprehensive dependency solution

set -e

echo "🔧 Setting up BigCodeBench dependencies..."
echo ""

# Check if we're in a virtual environment
if [[ -z "${VIRTUAL_ENV}" && -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "⚠️  Warning: No virtual environment detected."
    echo "   It's recommended to run this from an activated virtual environment."
    echo "   Example: source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "📦 Installing BigCodeBench dependencies..."
echo "   This includes 73 packages that BigCodeBench tests may require."
echo ""

# Install the project with BigCodeBench dependencies
uv pip install -e . --group bigcodebench

echo ""
echo "✅ BigCodeBench dependencies installed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Make sure to run evaluations from an activated virtual environment"
echo "   2. Run: python -m exploration_hacking.scripts.evaluate --config etc/bigcodebench/eval_bigcodebench.yaml"
echo ""
echo "💡 Why this works:"
echo "   BigCodeBench runs tests in subprocesses that inherit the parent's Python environment."
echo "   When you run evaluation from an activated venv, the subprocess can access all these packages."