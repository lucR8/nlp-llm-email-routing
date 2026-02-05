from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_PROMPT_TEMPLATE = """You are an email routing assistant.
Your task is to classify the email into one of the following categories:
{labels}

Email:
Subject: {subject}
Body: {body}

Answer with the department name only.
"""


@dataclass(frozen=True)
class PromptingConfig:
    model_name: str = "distilgpt2"   # faster baseline; you can also use "gpt2"
    max_new_tokens: int = 8
    temperature: float = 0.0         # deterministic
    do_sample: bool = False
    device: str | None = None        # "cuda" or "cpu" (auto if None)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def build_prompt(subject: str, body: str, label_list: list[str], template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    return template.format(
        labels=", ".join(label_list),
        subject=subject or "",
        body=body or "",
    )


def parse_department(text: str, label_list: list[str]) -> int | None:
    """
    Return label id if any label is detected in text, else None.
    We match labels case-insensitively.
    """
    out = _normalize(text)
    for i, lab in enumerate(label_list):
        if _normalize(lab) in out:
            return i
    return None


class GPT2PromptingRouter:
    def __init__(self, label_list: list[str], cfg: PromptingConfig | None = None):
        self.label_list = label_list
        self.cfg = cfg or PromptingConfig()

        self.device = self.cfg.device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        # GPT-2 has no pad token by default; set to eos for batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, examples: list[dict]) -> list[int]:
        """
        examples: list of dicts with keys: subject, body
        returns: list of predicted label ids
        """
        prompts = [
            build_prompt(ex.get("subject", ""), ex.get("body", ""), self.label_list)
            for ex in examples
        ]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        gen = self.model.generate(
            **enc,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only the generated continuation (after prompt)
        preds: list[int] = []
        for i in range(gen.shape[0]):
            prompt_len = enc["input_ids"][i].shape[0]
            # safer: decode full and parse
            text = self.tokenizer.decode(gen[i], skip_special_tokens=True)
            label_id = parse_department(text, self.label_list)
            preds.append(label_id if label_id is not None else -1)  # -1 = unknown/unparsed
        return preds
