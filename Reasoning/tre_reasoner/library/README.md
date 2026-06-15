# TRE Model Library API

This package exposes a small, stable API so `model/` can be treated as a library.

## Quick usage

```python
from library import TRELibrary
from library.workflow import RuntimeOptions

opts = RuntimeOptions(
    dataset="MATRES",
    split="test",
    data_mode="doc",
    model_path="/home/prof/ddao/daodd/phd-dao-do/TRE/notebooks/Reasoning/artifacts/tre_reasoner/checkpoints/MATRES_baseline_reasoning_model.pt",
)

lib = TRELibrary(opts)
result = lib.evaluate_energy(energy_mode="pred", debug=True)
print(result)
```

## Why this helps
- One entry point for loading config, data, model.
- Programmatic use in notebooks/scripts without depending on long CLI commands.
- Keeps old scripts intact while providing cleaner library-style usage.
