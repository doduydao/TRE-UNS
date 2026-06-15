import configparser
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DatasetConfig:
    name: str
    num_classes: int
    train_token_path: str
    train_jsonl_path: str
    val_token_path: str
    val_jsonl_path: str
    test_token_path: Optional[str] = None
    test_jsonl_path: Optional[str] = None
    rule_file: Optional[str] = None
    relation_map: Optional[Dict[str, int]] = None
    class_weights: Optional[List[float]] = None
    vague_label_id: Optional[int] = None


@dataclass
class ModelConfig:
    bert_model: str = "bert-base-uncased"
    freeze_bert: bool = False
    data_mode: str = 'pair'
    batch_size: int = 128
    epochs: int = 40
    lr: float = 2e-5
    max_grad_norm: float = 1.0
    use_amp: bool = True
    step_size: float = 1.0
    lambda_kl: float = 0.1
    smooth_tau: float = 0.05
    max_steps: int = 50
    tol: float = 1e-7
    use_deq: bool = False
    output_option: str = 'direct_q'
    rule_chunk_size: int = 1
    lambda_distill: float = 1.0
    lambda_energy: float = 1.0
    learn_rule_weights: bool = True
    initial_rule_weight: float = 0.0


REQUIRED_DATASET_KEYS = [
    "name",
    "num_classes",
    "train_token_path",
    "train_jsonl_path",
    "val_token_path",
    "val_jsonl_path",
    "test_token_path",
    "test_jsonl_path",
    "rule_file",
    "relation_map",
    "class_weights",
    "vague_label_id",
]

REQUIRED_MODEL_KEYS = [
    "bert_model",
    "freeze_bert",
    "data_mode",
    "batch_size",
    "epochs",
    "lr",
    "max_grad_norm",
    "use_amp",
    "step_size",
    "lambda_kl",
    "smooth_tau",
    "max_steps",
    "tol",
    "use_deq",
    "output_option",
    "rule_chunk_size",
    "lambda_distill",
    "lambda_energy",
    "learn_rule_weights",
    "initial_rule_weight",
]


def _project_root() -> Path:
    env_root = os.environ.get("TRE_REASONER_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).parent.parent


def _configs_root() -> Path:
    env_conf_root = os.environ.get("TRE_REASONER_CONFIG_ROOT")
    if env_conf_root:
        return Path(env_conf_root)
    return _project_root() / "script" / "config"


def available_datasets() -> List[str]:
    return sorted([p.stem.upper() for p in _configs_root().glob("*.conf")])


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _replace_root(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return value.replace("{ROOT}", str(_project_root()))


def _parse_json_or_none(value: Optional[str]):
    if value is None:
        return None
    s = value.strip().lower()
    if s in ("", "none", "null"):
        return None
    return json.loads(value)


def _require_keys(parser: configparser.ConfigParser, section: str, keys: List[str], conf_path: str):
    if section not in parser:
        raise ValueError(f"Missing [{section}] section in config: {conf_path}")
    missing = [k for k in keys if k not in parser[section]]
    if missing:
        raise ValueError(
            f"Missing keys in [{section}] of {conf_path}: {missing}. "
            f"Please define all parameters explicitly in .conf."
        )


def load_runtime_config(dataset_name: str, conf_path: Optional[str] = None):
    dataset_name = dataset_name.upper()
    if conf_path is None:
        conf_path = str(_configs_root() / f"{dataset_name}.conf")

    parser = configparser.ConfigParser()
    read_ok = parser.read(conf_path)
    if not read_ok:
        raise FileNotFoundError(f"Cannot read config file: {conf_path}")

    _require_keys(parser, "dataset", REQUIRED_DATASET_KEYS, conf_path)
    _require_keys(parser, "model", REQUIRED_MODEL_KEYS, conf_path)

    d = parser["dataset"]
    m = parser["model"]

    dataset_cfg = DatasetConfig(
        name=d.get("name", dataset_name),
        num_classes=d.getint("num_classes"),
        train_token_path=_replace_root(d.get("train_token_path")),
        train_jsonl_path=_replace_root(d.get("train_jsonl_path")),
        val_token_path=_replace_root(d.get("val_token_path")),
        val_jsonl_path=_replace_root(d.get("val_jsonl_path")),
        test_token_path=_replace_root(d.get("test_token_path", fallback=None)),
        test_jsonl_path=_replace_root(d.get("test_jsonl_path", fallback=None)),
        rule_file=_replace_root(d.get("rule_file", fallback=None)),
        relation_map=_parse_json_or_none(d.get("relation_map", fallback=None)),
        class_weights=_parse_json_or_none(d.get("class_weights", fallback=None)),
        vague_label_id=(None if d.get("vague_label_id", fallback="null").strip().lower() in ("none", "null", "") else d.getint("vague_label_id")),
    )

    model_cfg = ModelConfig(
        bert_model=m.get("bert_model"),
        freeze_bert=_as_bool(m.get("freeze_bert")),
        data_mode=m.get("data_mode"),
        batch_size=m.getint("batch_size"),
        epochs=m.getint("epochs"),
        lr=m.getfloat("lr"),
        max_grad_norm=m.getfloat("max_grad_norm"),
        use_amp=_as_bool(m.get("use_amp")),
        step_size=m.getfloat("step_size"),
        lambda_kl=m.getfloat("lambda_kl"),
        smooth_tau=m.getfloat("smooth_tau"),
        max_steps=m.getint("max_steps"),
        tol=m.getfloat("tol"),
        use_deq=_as_bool(m.get("use_deq")),
        output_option=m.get("output_option"),
        rule_chunk_size=m.getint("rule_chunk_size"),
        lambda_distill=m.getfloat("lambda_distill"),
        lambda_energy=m.getfloat("lambda_energy"),
        learn_rule_weights=_as_bool(m.get("learn_rule_weights")),
        initial_rule_weight=m.getfloat("initial_rule_weight"),
    )

    return dataset_cfg, model_cfg
