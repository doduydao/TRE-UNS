# NeuPSL IJCAI 2023

This folder contains the reproducible experiments for the paper *NeuPSL: Neural Probabilistic Soft Logic*.

## Structure

```text
neupsl-ijcai23/
├── MATRES/      # MATRES NeuPSL dataset/scripts
├── TBD/         # TBD NeuPSL dataset/scripts
├── scripts/     # Experiment wrappers and utility scripts
├── results/     # Generated output
├── requirements.txt
└── README.md
```

## Installation

Requirements:

- Bash 4+
- Java 7+
- Python 3.7+
- POSIX system such as Linux or macOS

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run NeuPSL for an experiment:

```bash
./scripts/run.sh <experiment>
```

`<experiment>` currently supports:

- `citation`
- `mnist-addition`
- `vspc`

This script will:

- regenerate data if needed,
- call `./<experiment>/cli/run.sh`,
- download the PSL jar from Maven Central if needed,
- write results to `results/<experiment>/`.

## Baselines

Baseline scripts live inside each experiment folder, following this pattern:

```text
./<experiment>/other-methods/<baseline>/scripts
```

## Paper

```text
@article{pryor2023ijcai,
    title   = {NeuPSL: Neural Probabilistic Soft Logic},
    booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)},
    year    = {2023}
}
```
