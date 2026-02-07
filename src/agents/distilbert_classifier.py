# src/agents/distilbert_classifier.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def _format_text(subject: str, body: str) -> str:
    """
    Standardize text formatting across agents for fair comparison.
    """
    subject = (subject or "").strip()
    body = (body or "").strip()
    if subject:
        return f"Subject: {subject}\nBody: {body}"
    return f"Body: {body}"


@dataclass
class DistilBertConfig:
    model_name: str = "distilbert-base-uncased"
    device: str = "cpu"  # "cpu" or "cuda"
    max_length: int = 256


class DistilBertClassifierRouter:
    """
    DistilBERT-based supervised classifier.
    This class supports:
      - loading base model or fine-tuned checkpoint
      - batch inference (predict_batch)
      - save/load
    Training is expected to be done in the notebook using Hugging Face Trainer.
    """

    def __init__(
        self,
        label_list: list[str],
        cfg: DistilBertConfig,
        checkpoint_dir: str | Path | None = None,
    ):
        self.label_list = label_list
        self.cfg = cfg

        self.id2label = {i: lab for i, lab in enumerate(label_list)}
        self.label2id = {lab: i for i, lab in enumerate(label_list)}

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

        # Decide device
        use_cuda = (cfg.device == "cuda") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # Load model
        # If checkpoint_dir is provided, we load from it (fine-tuned weights)
        model_source = str(checkpoint_dir) if checkpoint_dir is not None else cfg.model_name

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=len(label_list),
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_batch(self, items: list[dict[str, str]], batch_size: int = 32) -> list[int]:
        """
        items: [{"subject": "...", "body": "..."}, ...]
        returns: list[int] predicted label ids aligned with label_list order
        """
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

        return preds

    def save(self, out_dir: str | Path) -> None:
        """
        Save tokenizer + model in Hugging Face format.
        This is compatible with Trainer checkpoints and from_pretrained().
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)

    def load(self, ckpt_dir: str | Path) -> None:
        """
        Reload a fine-tuned checkpoint into this router.
        """
        ckpt_dir = Path(ckpt_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir))
        self.model.to(self.device)
        self.model.eval()
