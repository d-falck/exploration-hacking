#!/bin/bash
# Restores VCT data and frontier auditing outputs from the private data repo.
# Requires access to github.com/d-falck/exploration-hacking-private-data.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "Cloning private data repo..."
git clone --depth 1 https://github.com/d-falck/exploration-hacking-private-data.git "$TMPDIR/data"

echo "Restoring files..."
cp -r "$TMPDIR/data/data/vct" "$REPO_ROOT/data/vct"
cp "$TMPDIR/data/scripts/prepare_vct_search_data.py" "$REPO_ROOT/scripts/"
mkdir -p "$REPO_ROOT/other_experiments/frontier_auditing_discovery/outputs" \
         "$REPO_ROOT/other_experiments/frontier_auditing_discovery/artifacts"
cp -r "$TMPDIR/data/other_experiments/frontier_auditing_discovery/outputs/"* \
      "$REPO_ROOT/other_experiments/frontier_auditing_discovery/outputs/"
cp -r "$TMPDIR/data/other_experiments/frontier_auditing_discovery/artifacts/"* \
      "$REPO_ROOT/other_experiments/frontier_auditing_discovery/artifacts/"

echo "Done. All private data restored (gitignored — will not be committed)."
