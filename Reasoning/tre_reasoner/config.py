"""Compatibility config module backed by .conf files.

This keeps the old API (`get_config`, `ModelConfig`, `CONFIGS`) working while
all runtime data is loaded from `script/config/*.conf`.
"""

from typing import Dict, Optional, Tuple

from conf_loader import (
    DatasetConfig,
    ModelConfig,
    available_datasets,
    load_runtime_config,
)


def _build_configs() -> Dict[str, DatasetConfig]:
    configs: Dict[str, DatasetConfig] = {}
    for dataset_name in available_datasets():
        dataset_cfg, _ = load_runtime_config(dataset_name)
        configs[dataset_name] = dataset_cfg
    return configs


CONFIGS: Dict[str, DatasetConfig] = _build_configs()


def get_config(dataset_name: str, conf_path: Optional[str] = None) -> DatasetConfig:
    dataset_name = dataset_name.upper()
    if conf_path is None and dataset_name not in CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not found. Available: {list(CONFIGS.keys())}")
    dataset_cfg, _ = load_runtime_config(dataset_name, conf_path)
    return dataset_cfg


def get_runtime_configs(dataset_name: str, conf_path: Optional[str] = None) -> Tuple[DatasetConfig, ModelConfig]:
    return load_runtime_config(dataset_name, conf_path)
