# Neural-Only Inference Script

This script runs neural network predictions **without PSL reasoning**. It loads a pre-trained neural model checkpoint and evaluates predictions directly on the test set.

## Usage

### Quick Start

```bash
# Run for MATRES
./scripts/neural_infer.sh MATRES

# Run for TBD
./scripts/neural_infer.sh TBD

# Run for TDDMan
./scripts/neural_infer.sh TDDMan
```

### With Custom Cache Path

```bash
python scripts/neural_infer.py MATRES /path/to/custom/cache/
```

### Direct Python Execution

```bash
cd /path/to/project
python scripts/neural_infer.py <EXPERIMENT> [CACHE_PATH]
```

## What It Does

1. **Loads Pre-trained Model**: Reads checkpoint from `results/<EXPERIMENT>/checkpoint.pt`
2. **Loads Test Data**: Reads encoded test samples from cache directory
3. **Forward Pass**: Runs only the BERT-based neural network (no PSL)
4. **Predictions**: Generates predictions via argmax over softmax probabilities
5. **Evaluation**: Computes accuracy, F1-micro, F1-weighted, F1-macro, and per-class metrics
6. **Saves Results**: Stores metrics and detailed logs in `results/<EXPERIMENT>-neural-infer/`

## Output

Results are saved to:
```
results/<EXPERIMENT>-neural-infer/
├── neural_evaluation_metrics.txt    # Detailed classification report
└── neural_infer.log                  # Full execution log
```

### Metrics File Example

```
======================================================================
  NEURAL-ONLY TEST RESULTS (n=709)
======================================================================
  Accuracy:     0.796897
  F1-micro:     0.796897
  F1-weighted:  0.798801
  F1-macro:     0.584807
======================================================================

              precision    recall  f1-score   support

           0     0.8117    0.7321    0.7698       265
           1     0.8495    0.8886    0.8686       413
           2     0.1053    0.1290    0.1159        31

    accuracy                         0.7969       709
   macro avg     0.5888    0.5832    0.5848       709
weighted avg     0.8029    0.7969    0.7988       709
```

## Configuration by Experiment

The script automatically configures for each experiment:

| Experiment | Cache Path | Num Classes | Skip Label | Notes |
|-----------|-----------|-------------|-----------|-------|
| MATRES | `/data/ddao/TRE/pretrained_models/Reasoning/MATRES/cache/` | 3 | 3 (VAGUE) | Binary + EQUAL relations |
| TBD | `/data/ddao/TRE/pretrained_models/Reasoning/TBD/cache/` | 5 | None | 5-way temporal relations |
| TDDMan | `/data/ddao/TRE/pretrained_models/Reasoning/TDDMan/cache/` | 5 | None | 5-way temporal relations |

## Requirements

- PyTorch with CUDA support
- Transformers library (BERT model)
- scikit-learn (for metrics)
- Pre-trained checkpoint must exist at `results/<EXPERIMENT>/checkpoint.pt`
- Cached data must exist at configured `cache_path`

## Key Differences from PSL Pipeline

| Aspect | Neural-Only | PSL+Neural |
|--------|-----------|-----------|
| **Speed** | Fast (single forward pass) | Slow (iterative reasoning) |
| **Reasoning** | None | Full logical reasoning |
| **Convergence** | Immediate | Depends on ADMM iterations |
| **Constraints** | Not enforced | Enforced via PSL rules |
| **Use Case** | Baseline, debugging | Production predictions |

## Troubleshooting

### "Checkpoint not found"
- Ensure the model has been trained first: `bash scripts/run.sh <EXPERIMENT>`
- Check that `results/<EXPERIMENT>/checkpoint.pt` exists

### "Cache path not found"
- Verify the cached data directory contains `train/`, `valid/`, `test/` subdirectories
- Each should have `token_cache_bert.pt` and `spacy_cache_bert.jsonl` files

### CUDA out of memory
- Reduce batch size: Edit `neural_infer.py` line with `batch_size=128` and change to a smaller value
- Or disable GPU: `CUDA_VISIBLE_DEVICES="" python scripts/neural_infer.py MATRES`

## Example: Compare Neural vs PSL

```bash
# Run PSL inference (includes neural)
bash scripts/infer.sh MATRES

# Run neural-only for comparison
bash scripts/neural_infer.sh MATRES

# Compare results
diff results/MATRES-infer/evaluation_metrics.txt results/MATRES-neural-infer/neural_evaluation_metrics.txt
```

The PSL+Neural should typically achieve better performance than neural-only due to logical constraint enforcement.
