# TBD - NeuPSL

This folder contains the configuration and data for the TBD experiment in the NeuPSL suite.

## Structure

```text
TBD/
├── cli/        # PSL entrypoint and config
├── data/       # TBD splits and mappings
├── scripts/    # Train / evaluate / generate scripts
└── tre/        # Neural source, data loader, and model
```

## Installation

There are no separate dependencies beyond `neupsl-ijcai23/requirements.txt`.

From the repo root:

```bash
cd neupsl-ijcai23
python -m pip install -r requirements.txt
```

## Usage

Run from the experiment directory:

```bash
cd TBD/cli
./run.sh
```

Or use the wrapper at the root:

```bash
cd ..
./scripts/run.sh TBD
```

Results are written to `results/TBD/`.

## Notes

- `tre/model.py` is the neural model for TBD.
- `scripts/generate_psl_data_v2.py` and `scripts/evaluate_test.py` support the PSL pipeline.
- `scripts/setup_symlinks.sh` helps create local symlinks when needed.
