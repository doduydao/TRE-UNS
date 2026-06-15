#!/bin/bash
# Run neural-only inference (no PSL) for a given experiment
# Usage: ./neural_infer.sh <experiment> [cache_path]
# Example: ./neural_infer.sh MATRES
#          ./neural_infer.sh TBD /path/to/cache/

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly PROJ_ROOT="${THIS_DIR}/.."

if [[ $# -lt 1 ]]; then
    echo "USAGE: $0 <experiment> [cache_path]"
    echo "Example: $0 MATRES"
    echo "         $0 TBD /data/ddao/TRE/pretrained_models/Reasoning/TBD/cache/"
    exit 1
fi

readonly EXPERIMENT=$1
readonly CACHE_PATH=$2

# Run Python script
cd "${PROJ_ROOT}"
python "${THIS_DIR}/neural_infer.py" "${EXPERIMENT}" "${CACHE_PATH}"
