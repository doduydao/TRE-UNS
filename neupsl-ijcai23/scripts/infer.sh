#!/bin/bash
# Run NeuPSL Inference only (no learning) on test set using saved model
# Usage: ./infer.sh <experiment>  (e.g. MATRES, TDDMan)

if [[ $# -ne 1 ]]; then
    echo "USAGE: $0 <experiment>"
    exit 1
fi

readonly EXPERIMENT=$1
readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly RESULTS_DIR="${THIS_DIR}/../results/${EXPERIMENT}-infer"
readonly CLI_DIR="${THIS_DIR}/../${EXPERIMENT}/cli"
readonly CONFIG="${CLI_DIR}/${EXPERIMENT}-infer.json"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Config not found: ${CONFIG}"
    exit 1
fi

mkdir -p "${RESULTS_DIR}"
export NEUPSL_LOG_DIR="${RESULTS_DIR}"

echo "=== Running NeuPSL Inference: ${EXPERIMENT} ==="
echo "Config: ${CONFIG}"
echo "Results: ${RESULTS_DIR}"

pushd . > /dev/null
    cd "${CLI_DIR}"
    time java -jar psl-cli-2.4.0.jar --infer --config "${CONFIG}" --output inferred-predicates \
        > "${RESULTS_DIR}/out.txt" 2> "${RESULTS_DIR}/out.err"

    mv inferred-predicates "${RESULTS_DIR}/" 2>/dev/null
popd > /dev/null

echo "=== Inference Complete ==="
echo "Evaluating..."

python "${THIS_DIR}/../${EXPERIMENT}/scripts/evaluate_inferred.py" "${RESULTS_DIR}/inferred-predicates/REL.txt"
