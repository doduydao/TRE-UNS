#!/bin/bash
# Setup symlinks for TBD experiment
# Run this from the neupsl-ijcai23/ directory:
#   bash TBD/scripts/setup_symlinks.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SCRIPT_DIR = .../TBD/scripts
# BASE_DIR = .../TBD
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# PROJECT_DIR = .../neupsl-ijcai23
PROJECT_DIR="$(cd "$BASE_DIR/.." && pwd)"

echo "Setting up TBD symlinks..."
echo "  TBD dir: $BASE_DIR"
echo "  Project dir: $PROJECT_DIR"

# Clean up any incorrect nested TBD directory
rm -rf "$BASE_DIR/TBD"

# Create tre symlink in TBD/ pointing to ../MATRES/tre
ln -sf ../MATRES/tre "$BASE_DIR/tre"

# Create JAR symlink
ln -sf ../../MATRES/cli/psl-cli-2.4.0.jar "$BASE_DIR/cli/psl-cli-2.4.0.jar"

# Make run.sh executable
chmod +x "$BASE_DIR/cli/run.sh"

# Create data directory
mkdir -p "$BASE_DIR/data"

echo ""
echo "Verifying..."
echo "tre symlink:"
ls -la "$BASE_DIR/tre"
echo "JAR symlink:"
ls -la "$BASE_DIR/cli/psl-cli-2.4.0.jar"
echo "run.sh:"
ls -la "$BASE_DIR/cli/run.sh"
echo "data dir:"
ls -la "$BASE_DIR/data/"
echo ""
echo "Done!"
