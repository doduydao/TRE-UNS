# TRE-UNS

This repo contains two main tracks for temporal relation extraction:

- `Reasoning/`: a TRE system built around neural + logic reasoning.
- `neupsl-ijcai23/`: reproducible NeuPSL experiments from the IJCAI 2023 paper.

Link to paper:  Unified Neural-Symbolic Method for Temporal Relation Extraction: https://rdcu.be/8Ydz3aPfFKGa


## Structure

```text
TRE-UNS/
├── Reasoning/              # Reasoning source code and analysis artifacts
├── neupsl-ijcai23/         # NeuPSL experiments
└── README.md
```

Folders such as `results/`, `analysis/`, `checkpoint/`, and `__pycache__/` are generated artifacts, not primary source code.

## Installation

Minimum requirements:

- Python 3.9+
- `pip`
- Bash
- Java 7+ if you run NeuPSL

Recommended: create a dedicated virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### NeuPSL

```bash
pip install -r neupsl-ijcai23/requirements.txt
```

### Reasoning TRE

There is no separate `requirements.txt` at the root of `Reasoning/`, so install the main packages used by the current code:

```bash
pip install torch transformers pandas numpy scikit-learn tqdm spacy spacy-alignments orjson
```

If you run the spaCy preprocessing step, install the English model:

```bash
python -m spacy download en_core_web_sm
```

## Quick Start

### NeuPSL

```bash
cd neupsl-ijcai23
./scripts/run.sh <experiment>
```

`<experiment>` can be `citation`, `mnist-addition`, or `vspc`.

### Reasoning

```bash
cd Reasoning
bash script/run.sh MATRES evaluate
```

The datasets supported by the current config are:

- `MATRES`
- `I2B2`
- `TDDMAN`
- `TBD`

## Notes

- Runtime config files live in `Reasoning/script/config/*.conf`.
- The cache paths in the config are absolute and may need to be adjusted for a different machine.
- Inference outputs, checkpoints, logs, and charts are written into the corresponding subfolders in each project.


## Contact
- Name: Duy Dao DO
- Email: duy-dao.do@univ-orleans.fr
