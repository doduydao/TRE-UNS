# TRE Reasoner Folder Structure (Library Root)

This folder is the library root (`tre_reasoner`) and contains only execution-related source code.

## Core files
- `model.py`: model architecture + factory (`create_model`)
- `reasoning_v2.py`: logic reasoning layer and rule energy computation
- `train.py`: training loops and losses
- `start_training.py`: training entrypoint
- `start_evaluation.py`: evaluation entrypoint (orchestration only)
- `eval_helpers.py`: evaluation helper logic (consistency, GT/pred energy)
- `config.py`: dataset/model runtime configuration

## Library API (recommended)
- `library/`: clean programmatic API for using `model` as a library.
  - `library/workflow.py`: `TRELibrary`, `RuntimeOptions`
  - `library/__init__.py`: public exports
  - `library/README.md`: quick examples

## Diagnostic / debug scripts
Keep ad-hoc scripts as diagnostics only:
- Moved outside `model/` to keep library core clean:
  - `../debug_model/`
  - `../analysis_csv/`
- Typical script names there:
  - `debug_*.py`
  - `verify_*.py`
  - `inspect_*.py`
  - `analyze_*.py`

## Runtime artifacts
Stored outside library root:
- checkpoints: `artifacts/tre_reasoner/checkpoints/`
- predictions: `artifacts/tre_reasoner/predictions/`
- logs/cache: `artifacts/tre_reasoner/`

## Usage
- Standard test evaluation:
  - `python tre_reasoner/start_evaluation.py --dataset MATRES --mode baseline_reasoning --split test --data_mode doc --model_path artifacts/tre_reasoner/checkpoints/MATRES_baseline_reasoning_model.pt`
- GT energy check:
  - `python tre_reasoner/start_evaluation.py --dataset MATRES --mode energy_gt --split test --data_mode doc --model_path artifacts/tre_reasoner/checkpoints/MATRES_baseline_reasoning_model.pt --debug_energy`
- Predicted energy check:
  - `python tre_reasoner/start_evaluation.py --dataset MATRES --mode energy_pred --split test --data_mode doc --model_path artifacts/tre_reasoner/checkpoints/MATRES_baseline_reasoning_model.pt --debug_energy`
