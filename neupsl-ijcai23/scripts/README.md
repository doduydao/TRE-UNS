# scripts

This folder contains NeuPSL wrappers, data generation helpers, and result-analysis scripts.

## Structure

```text
scripts/
├── run.sh
├── gen_data.sh
├── neural_infer.sh
├── neural_infer.py
├── evaluate_test.py
├── generate_psl_data_v2.py
├── plot_scalability.py
└── README.md
```

## Installation

Same requirements as the parent `neupsl-ijcai23` project:

- Python 3.7+
- Bash 4+
- Java 7+

Install Python dependencies:

```bash
cd ..
python -m pip install -r requirements.txt
```

## Usage

Run an experiment:

```bash
./run.sh MATRES
```

Run neural-only inference:

```bash
./neural_infer.sh MATRES
```

## Notes

- `run.sh` is the main entrypoint.
- `gen_data.sh` generates PSL data.
- `parse-results.py` and `results-parser.py` are used for summarizing results.
