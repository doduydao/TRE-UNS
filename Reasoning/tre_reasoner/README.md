# tre_reasoner

This is the core package of the TRE system. It bundles:

- encoder BERT + pooling entity,
- baseline classifier,
- reasoning layer logic,
- train/eval entrypoint,
- API library for notebooks or external scripts.

## Structure

```text
tre_reasoner/
├── model.py            # model factory and model classes
├── reasoning_v2.py     # logic reasoning layer, parser, energy
├── train.py            # training loop and losses
├── evaluate.py         # metric, inference, consistency
├── eval_helpers.py     # helpers for energy/violation measurements
├── data.py             # dataset/cache loader
├── data_sampler.py     # document-level sampler
├── conf_loader.py      # .conf loader
├── config.py           # compatibility wrapper
├── start_training.py   # CLI train
├── start_evaluation.py # CLI evaluate
├── start_generation.py # CLI for exporting prediction CSVs
├── start_preprocessing.py
└── library/            # library API
```

## Installation

Use the dependencies listed at the repo root. If you run this package in isolation, you will need at least:

```bash
pip install torch transformers pandas numpy scikit-learn tqdm spacy spacy-alignments orjson
```

## Config

Runtime config files are read from:

```text
Reasoning/script/config/*.conf
```

These configs point to cached preprocessed data, the rule file, label mappings, and model parameters.

## Quick Usage

Train:

```bash
python start_training.py --dataset MATRES --mode baseline_reasoning
```

Evaluate:

```bash
python start_evaluation.py --dataset MATRES --mode baseline_reasoning --split test
```

Export prediction CSV:

```bash
python start_generation.py --dataset MATRES --mode baseline_reasoning --model_path <checkpoint>
```

## Library API

If you want to use it from a notebook or external code:

```python
from library import TRELibrary
from library.workflow import RuntimeOptions

opts = RuntimeOptions(dataset="MATRES", split="test", model_path="path/to/checkpoint.pt")
lib = TRELibrary(opts)
result = lib.evaluate_energy(energy_mode="pred", debug=True)
```

## Notes

- `baseline` runs only the encoder.
- `baseline_psl` uses the baseline model plus PSL regularization.
- `baseline_reasoning` and `TRER` use the logic reasoning layer.
- Some modes require an existing checkpoint and cached data.
