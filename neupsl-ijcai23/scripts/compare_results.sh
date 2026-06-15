#!/bin/bash
# Quick comparison script - compare neural vs PSL vs neural+PSL results

readonly EXPERIMENT=${1:-MATRES}
readonly RESULTS_DIR="./results"

echo "================================================================================"
echo "COMPARISON: Neural-Only vs PSL with Different Iterations vs Neural in PSL Output"
echo "Experiment: $EXPERIMENT"
echo "================================================================================"
echo ""

# Find available result directories
echo "Available results:"
ls -d "$RESULTS_DIR/${EXPERIMENT}"* 2>/dev/null | while read dir; do
    basename "$dir"
done
echo ""

# Compare metrics files
echo "Accuracy comparison:"
echo "---"

# Neural-only from neural_infer.sh
if [[ -f "$RESULTS_DIR/${EXPERIMENT}-neural-infer/neural_evaluation_metrics.txt" ]]; then
    echo "Neural-Only (neural_infer.sh):"
    grep "Accuracy:" "$RESULTS_DIR/${EXPERIMENT}-neural-infer/neural_evaluation_metrics.txt"
fi

# Neural extracted from PSL NEURAL.txt
echo ""
echo "Evaluating NEURAL.txt from infer.sh..."
python scripts/evaluate_neural_from_psl.py "$EXPERIMENT" "$RESULTS_DIR/${EXPERIMENT}-infer/inferred-predicates/NEURAL.txt" 2>&1 | grep -A 5 "NEURAL PREDICTIONS FROM PSL"

# PSL+Neural full iterations
if [[ -f "$RESULTS_DIR/${EXPERIMENT}-infer/evaluation_metrics.txt" ]]; then
    echo ""
    echo "PSL+Neural (full 1000 iterations):"
    grep "Accuracy:" "$RESULTS_DIR/${EXPERIMENT}-infer/evaluation_metrics.txt"
fi

# PSL with 1 step if exists
if [[ -f "$RESULTS_DIR/${EXPERIMENT}-infer_1 step/evaluation_metrics.txt" ]]; then
    echo ""
    echo "PSL+Neural (1 iteration):"
    grep "Accuracy:" "$RESULTS_DIR/${EXPERIMENT}-infer_1 step/evaluation_metrics.txt"
fi

# PSL with 50 steps if exists
if [[ -f "$RESULTS_DIR/${EXPERIMENT}-infer_50 step/evaluation_metrics.txt" ]]; then
    echo ""
    echo "PSL+Neural (50 iterations):"
    grep "Accuracy:" "$RESULTS_DIR/${EXPERIMENT}-infer_50 step/evaluation_metrics.txt"
fi

echo ""
echo "================================================================================"
echo "INTERPRETATION:"
echo "================================================================================"
echo ""
echo "Neural-only       : Pure neural network, no PSL reasoning"
echo "PSL step=1        : PSL with only 1 ADMM iteration (usually worse - incomplete)"
echo "PSL step=50       : PSL with 50 ADMM iterations (improving convergence)"
echo "PSL step=1000     : PSL with 1000 ADMM iterations (best - fully converged)"
echo ""
echo "Expected pattern: step=1 < neural-only < step=50 < step=1000"
echo "(But sometimes step=1 can be worse than neural-only if PSL rules are misaligned)"
echo ""
echo "================================================================================"
