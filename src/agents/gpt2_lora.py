# src/agents/gpt2_lora.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import json

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model, TaskType, PeftModel


def _format_text(subject: str, body: str) -> str:
    subject = (subject or "").strip()
    body = (body or "").strip()
    if subject:
        return f"Subject: {subject}\nBody: {body}"
    return f"Body: {body}"


@dataclass(frozen=True)
class LoRAConfig:
    model_name: str = "distilgpt2"
    device: str | None = None  # "cpu" ou "cuda" (auto si None)
    max_length: int = 256

    # LoRA hyperparams
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("c_attn", "c_proj")


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str = "outputs/checkpoints/gpt2_lora"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    logging_steps: int = 50
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"  # transformers v5: eval_strategy (NOT evaluation_strategy)
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True
    report_to: str = "none"
    seed: int = 42


def _compute_accuracy(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def prepare_tokenized_splits(
    tokenizer,
    train_ds,
    val_ds,
    test_ds,
    label2id: dict[str, int],
    max_length: int,
):
    """
    Expects HF Datasets with columns at least: subject, body, queue
    Returns tokenized datasets with columns: input_ids, attention_mask, labels
    """

    def add_text_and_labels(ex):
        ex["text"] = _format_text(ex.get("subject", ""), ex.get("body", ""))
        ex["labels"] = int(label2id[ex["queue"]])  # REQUIRED so the model returns a loss
        return ex

    def tok_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=max_length)

    train_l = train_ds.map(add_text_and_labels)
    val_l = val_ds.map(add_text_and_labels)
    test_l = test_ds.map(add_text_and_labels)

    train_tok = train_l.map(tok_fn, batched=False)
    val_tok = val_l.map(tok_fn, batched=False)
    test_tok = test_l.map(tok_fn, batched=False)

    keep = {"input_ids", "attention_mask", "labels"}
    train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in keep])
    val_tok = val_tok.remove_columns([c for c in val_tok.column_names if c not in keep])
    test_tok = test_tok.remove_columns([c for c in test_tok.column_names if c not in keep])

    return train_tok, val_tok, test_tok


class GPT2LoRARouter:
    """
    GPT-2 + LoRA fine-tuning for sequence classification.

    Designed to be comparable with GPT2PromptingRouter:
      - same input format {"subject","body"}
      - same label order label_list -> label ids
      - same max_length default (256)
    """

    def __init__(self, label_list: list[str], cfg: LoRAConfig, init_adapter: bool = True):
        self.label_list = label_list
        self.cfg = cfg

        self.id2label = {i: lab for i, lab in enumerate(label_list)}
        self.label2id = {lab: i for i, lab in enumerate(label_list)}

        device = cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = None
        if init_adapter:
            self._build_base_and_lora()

    @classmethod
    def from_adapter(
        cls,
        label_list: list[str],
        cfg: LoRAConfig,
        adapter_dir: str | Path,
    ) -> "GPT2LoRARouter":
        """
        Construct a router and load a LoRA adapter in one step.
        This avoids double base-model loading (and duplicate load reports).
        """
        self = cls(label_list, cfg, init_adapter=False)
        self.load(adapter_dir)
        return self

    def _build_base_and_lora(self) -> None:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id,
        )
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

        lora = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.cfg.r,
            lora_alpha=self.cfg.alpha,
            lora_dropout=self.cfg.dropout,
            target_modules=list(self.cfg.target_modules),
            bias="none",
        )
        self.model = get_peft_model(base_model, lora)
        self.model.to(self.device)
        self.model.eval()

    def train(
        self,
        train_ds,
        val_ds,
        test_ds,
        train_cfg: TrainConfig,
        label2id: dict[str, int],
    ) -> dict[str, Any]:
        if self.model is None:
            self._build_base_and_lora()
        train_tok, val_tok, _ = prepare_tokenized_splits(
            tokenizer=self.tokenizer,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            label2id=label2id,
            max_length=self.cfg.max_length,
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        args = TrainingArguments(
            output_dir=str(train_cfg.output_dir),
            eval_strategy=train_cfg.eval_strategy,
            save_strategy=train_cfg.save_strategy,
            logging_strategy="steps",
            logging_steps=train_cfg.logging_steps,
            num_train_epochs=train_cfg.num_train_epochs,
            per_device_train_batch_size=train_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            fp16=torch.cuda.is_available() and self.device.type == "cuda",
            report_to=train_cfg.report_to,
            load_best_model_at_end=train_cfg.load_best_model_at_end,
            metric_for_best_model=train_cfg.metric_for_best_model,
            greater_is_better=train_cfg.greater_is_better,
            seed=train_cfg.seed,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            data_collator=data_collator,
            compute_metrics=_compute_accuracy,
        )

        tr = trainer.train()
        ev = trainer.evaluate()
        best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)

        best_adapter_dir: str | None = None
        if best_ckpt:
            out_best = Path(train_cfg.output_dir) / "best_adapter"
            out_best.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(out_best)
            self.tokenizer.save_pretrained(out_best)
            best_adapter_dir = str(out_best)

        return {
            "best_checkpoint": best_ckpt,
            "train_metrics": getattr(tr, "metrics", {}) if tr is not None else {},
            "eval_metrics": ev,
            "out_dir": str(train_cfg.output_dir),
            "best_adapter_dir": best_adapter_dir,
        }

    @torch.inference_mode()
    def predict_batch(self, items: list[dict[str, str]], batch_size: int = 16) -> list[int]:
        if self.model is None:
            raise RuntimeError("Model not initialized. Call load() or create with init_adapter=True.")
        preds: list[int] = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            texts = [_format_text(x.get("subject", ""), x.get("body", "")) for x in batch]
            tok = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.cfg.max_length,
                padding=True,
                return_tensors="pt",
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}
            out = self.model(**tok)
            logits = out.logits.detach().cpu().numpy()
            preds.extend(np.argmax(logits, axis=-1).tolist())
        return [int(p) for p in preds]

    def save(self, out_dir: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model not initialized. Call load() or create with init_adapter=True.")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)


    def load(self, adapter_dir: str | Path) -> None:
        adapter_dir = Path(adapter_dir)

        # Optionally reload tokenizer if present in the adapter directory.
        if (adapter_dir / "tokenizer.json").exists():
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pick base model from adapter_config.json if available.
        base_name = self.cfg.model_name
        cfg_path = adapter_dir / "adapter_config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                base_name = json.load(f).get("base_model_name_or_path", base_name)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id,
        )
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.to(self.device)
        self.model.eval()
