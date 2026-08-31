# Reasoning

This folder contains the TRE neural-symbolic reasoning system, along with analysis artifacts for MATRES, TBD, I2B2, and TDDMAN.

## Structure

```text
Reasoning/
├── tre_reasoner/      # Core source: model, reasoning, train, eval
├── script/            # Run scripts, runtime config, analysis tools
├── MATRES/            # Output and analysis for MATRES
├── TBD/               # Output and analysis for TBD
├── all_results/       # Aggregated results for comparison
├── analysis_csv/      # CSV files for analysis
├── analysis_prediction/
├── debug_model/
├── notebook/
└── rules/             # Logic rule files
```

## Installation

See the root [README](../README.md) for shared installation steps.

You will need at least:

- Python 3.9+
- PyTorch
- Transformers
- pandas, numpy, scikit-learn, tqdm
- spaCy + `spacy-alignments`
- `orjson`

## Quick Usage

Run training / evaluation through the wrapper:

```bash
cd Reasoning
bash script/run.sh MATRES train
bash script/run.sh MATRES evaluate
```

Or call the entrypoints directly:

```bash
python tre_reasoner/start_training.py --dataset MATRES --mode baseline_reasoning
python tre_reasoner/start_evaluation.py --dataset MATRES --mode baseline_reasoning --split test
```

## Notes

- `script/config/*.conf` is the main runtime config source.
- `MATRES/` and `TBD/` contain logs, prediction CSVs, and analysis files; they are generated outputs.
- `rules/` contains the logic rule files loaded by the reasoning layer.
