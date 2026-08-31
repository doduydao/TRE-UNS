# script

This folder contains experiment scripts, runtime configs, and a few result-analysis tools.

## Structure

```text
script/
├── run.sh                   # Train / predict / evaluate wrapper
├── analyze_reasoning_log.py  # Parse reasoning training logs
├── stats_rule_satisfaction_tbd.py
└── config/
    ├── MATRES.conf
    ├── TBD.conf
    ├── I2B2.conf
    └── TDDMAN.conf
```

## Installation

There are no dependencies beyond the `Reasoning/tre_reasoner` package.

These scripts assume you already installed:

- Python 3.9+
- PyTorch
- Transformers
- pandas / numpy / scikit-learn / tqdm
- spaCy + `spacy-alignments`

## Usage

### Run a dataset

```bash
cd Reasoning
bash script/run.sh MATRES evaluate
```

Supported run modes:

- `train`
- `predict`
- `evaluate`

Model modes:

- `baseline`
- `baseline_psl`
- `baseline_reasoning`
- `TRER`

### Analyze logs

```bash
python script/analyze_reasoning_log.py Reasoning/MATRES/log_train.txt
```

### Rule satisfaction stats for TBD

```bash
python script/stats_rule_satisfaction_tbd.py --split test
```

## Notes

- `run.sh` reads the matching config file from `config/` automatically.
- The cache paths in the config are absolute and may need to be changed for another environment.
- `run.sh` writes output to `Reasoning/<DATASET>/` and checkpoints to `Reasoning/<DATASET>/checkpoint/`.
