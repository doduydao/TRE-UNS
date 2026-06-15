#!/bin/bash
# Generate PSL data files (targets, truth, entity-data-map)
# Usage: ./gen_data.sh [MATRES|TDDMan]  (default: all)

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

declare -A CACHE_DIRS
CACHE_DIRS["MATRES"]="/data/ddao/TRE/pretrained_models/Reasoning/MATRES/cache"
CACHE_DIRS["TDDMan"]="/data/ddao/TRE/pretrained_models/Reasoning/TDDMan/cache"
CACHE_DIRS["TBD"]="/data/ddao/TRE/pretrained_models/Reasoning/TBD/cache"

function gen() {
    local name=$1
    local cache_dir="${CACHE_DIRS[$name]}"
    local output_dir="${THIS_DIR}/../${name}/data"
    local script="${THIS_DIR}/../${name}/scripts/generate_psl_data_v2.py"

    if [[ -f "${output_dir}/entity-data-map.txt" ]]; then
        echo "=== PSL Data already exists for ${name}, skipping. Use --force to regenerate. ==="
        return
    fi

    echo "=== Generating PSL Data: ${name} ==="
    python "${script}" --cache_dir "${cache_dir}" --output_dir "${output_dir}"
    echo "=== Done: ${name} ==="
}

if [[ $# -eq 0 ]]; then
    for name in "${!CACHE_DIRS[@]}"; do
        gen "$name"
    done
else
    gen "$1"
fi
