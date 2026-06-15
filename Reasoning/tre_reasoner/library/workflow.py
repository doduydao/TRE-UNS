from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, BertTokenizerFast

from ..config import get_config, ModelConfig
from ..data import TRECachedDataset, tre_collate_cached
from ..data_sampler import DocBatchSampler, SortedDocSampler
from ..model import create_model
from ..eval_helpers import calculate_gt_energy, calculate_pred_energy


@dataclass
class RuntimeOptions:
    dataset: str = "MATRES"
    split: str = "test"
    mode: str = "baseline_reasoning"
    data_mode: str = "doc"
    model_path: Optional[str] = None
    gpu: int = 0
    batch_size: Optional[int] = None


class TRELibrary:
    def __init__(self, options: RuntimeOptions):
        self.options = options
        self.dataset_cfg = get_config(options.dataset)
        self.model_cfg = ModelConfig()
        if options.data_mode:
            self.model_cfg.data_mode = options.data_mode
        self.device = torch.device(f"cuda:{options.gpu}" if torch.cuda.is_available() else "cpu")

    def build_dataloader(self):
        if self.options.split == "test":
            token_path = self.dataset_cfg.test_token_path
            jsonl_path = self.dataset_cfg.test_jsonl_path
        else:
            token_path = self.dataset_cfg.val_token_path
            jsonl_path = self.dataset_cfg.val_jsonl_path

        if (not token_path or not jsonl_path) and self.options.split == "test":
            token_path = self.dataset_cfg.val_token_path
            jsonl_path = self.dataset_cfg.val_jsonl_path

        dataset = TRECachedDataset(token_path, jsonl_path)
        batch_size = self.options.batch_size if self.options.batch_size is not None else self.model_cfg.batch_size

        if self.model_cfg.data_mode == "doc":
            sampler = DocBatchSampler(dataset.doc_ids, batch_size, shuffle=False)
            dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=tre_collate_cached)
        else:
            ordered_sampler = SortedDocSampler(dataset.doc_ids)
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=ordered_sampler, collate_fn=tre_collate_cached)

        return dataloader

    def build_model(self):
        bert_model_name = self.model_cfg.bert_model
        tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
        bert = AutoModel.from_pretrained(bert_model_name)
        tokenizer.add_special_tokens({"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]})
        bert.resize_token_embeddings(len(tokenizer))

        model = create_model(
            mode="baseline_reasoning",
            bert=bert,
            num_classes=self.dataset_cfg.num_classes,
            rule_file_path=self.dataset_cfg.rule_file,
            relation_map=self.dataset_cfg.relation_map,
            num_types=1,
            step_size=self.model_cfg.step_size,
            lambda_kl=self.model_cfg.lambda_kl,
            smooth_tau=self.model_cfg.smooth_tau,
            max_steps=self.model_cfg.max_steps,
            tol=self.model_cfg.tol,
            use_deq=self.model_cfg.use_deq,
            output_option=self.model_cfg.output_option,
            learn_rule_weights=self.model_cfg.learn_rule_weights,
            initial_rule_weight=self.model_cfg.initial_rule_weight,
            rule_chunk_size=self.model_cfg.rule_chunk_size,
            tokenizer=tokenizer,
        )

        model_path = self.options.model_path
        if model_path:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)

        model.to(self.device)
        model.eval()
        return model

    def evaluate_energy(self, energy_mode: str = "pred", debug: bool = False) -> Dict[str, Any]:
        dataloader = self.build_dataloader()
        model = self.build_model()

        if energy_mode == "gt":
            avg_energy = calculate_gt_energy(
                model=model,
                dataloader=dataloader,
                device=self.device,
                dataset_cfg=self.dataset_cfg,
                debug=debug,
            )
            return {"mode": "gt", "avg_energy": float(avg_energy)}

        avg_energy = calculate_pred_energy(
            model=model,
            dataloader=dataloader,
            device=self.device,
            dataset_cfg=self.dataset_cfg,
            debug=debug,
        )
        return {"mode": "pred", "avg_energy": float(avg_energy)}
