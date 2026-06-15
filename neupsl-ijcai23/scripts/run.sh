#!/bin/bash

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly RESULTS_DIR="${THIS_DIR}/../results"

function run() {
    local experiment=$1

    if [[ ! -d "${THIS_DIR}/../${experiment}" ]]; then
        echo "Experiment does not exist: ${experiment}"
        exit 1
    fi

    local cli_dir="${THIS_DIR}/../${experiment}/cli"
    local results_dir="${RESULTS_DIR}/${experiment}"

    mkdir -p "${results_dir}"

    # Regenerate PSL data files
    "${THIS_DIR}/gen_data.sh" "${experiment}"

    local log_out_path="${results_dir}/out.txt"
    local log_err_path="${results_dir}/out.err"

    # Export log dir so Python writes neupsl_python.log next to out.txt
    export NEUPSL_LOG_DIR="${results_dir}"

    echo "Running '${results_dir}'."

    pushd . > /dev/null
        cd "${cli_dir}"
        time bash run.sh > "${log_out_path}" 2> "${log_err_path}"

        # Move inferred predicates into results
        mv inferred-predicates "${results_dir}/" 2>/dev/null
    popd > /dev/null
}

function main() {
    if [[ $# -ne 1 ]]; then
        echo "USAGE: $0 <experiment>"
        exit 1
    fi

    trap exit SIGINT

    run $1
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
