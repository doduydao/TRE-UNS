#!/bin/bash
# Setup TDDMan directory structure for NeuPSL

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="${THIS_DIR}/.."

echo "=== Setting up TDDMan ==="

# Create directories
mkdir -p "${ROOT}/TDDMan/cli"
mkdir -p "${ROOT}/TDDMan/data"
mkdir -p "${ROOT}/TDDMan/scripts"

# Symlinks
ln -sf ../MATRES/tre "${ROOT}/TDDMan/tre"
ln -sf ../../MATRES/cli/psl-cli-2.4.0.jar "${ROOT}/TDDMan/cli/psl-cli-2.4.0.jar"
ln -sf ../../MATRES/scripts/neupsl-model.py "${ROOT}/TDDMan/scripts/neupsl-model.py"
ln -sf ../../MATRES/scripts/generate_psl_data_v2.py "${ROOT}/TDDMan/scripts/generate_psl_data_v2.py"
ln -sf ../../MATRES/scripts/evaluate_inferred.py "${ROOT}/TDDMan/scripts/evaluate_inferred.py"
ln -sf ../../MATRES/scripts/evaluate_test.py "${ROOT}/TDDMan/scripts/evaluate_test.py"

echo "Symlinks:"
ls -la "${ROOT}/TDDMan/tre" "${ROOT}/TDDMan/cli/psl-cli-2.4.0.jar" "${ROOT}/TDDMan/scripts/"

# Generate PSL data
echo ""
echo "=== Generating PSL data for TDDMan ==="
python "${ROOT}/TDDMan/scripts/generate_psl_data_v2.py" \
    --cache_dir "/data/ddao/TRE/pretrained_models/Reasoning/TDDMan/cache" \
    --output_dir "${ROOT}/TDDMan/data"

echo "=== Setup Complete ==="
