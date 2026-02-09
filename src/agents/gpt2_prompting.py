from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_PROMPT_TEMPLATE = """You are an email router.
Classify the ticket into exactly ONE of these queues:
{labels}

Return ONLY the queue label, nothing else.

Subject: {subject}
Body: {body}

Label:
"""


def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s\-]", " ", s)   # keep words + hyphen
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_prompt(subject: str, body: str, label_list: list[str], template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    return template.format(
        labels=" | ".join(label_list),
        subject=(subject or "").strip(),
        body=(body or "").strip(),
    )

def parse_department(text: str, label_list: list[str]) -> int | None:
    """
    Return label id if a label is predicted exactly (robust parsing).
    IMPORTANT: apply to the model generated answer only, not the full prompt.
    """
    if not text:
        return None

    t = text.strip().lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()

    for i, lab in enumerate(label_list):
        if t == lab.lower().strip():
            return i
    return None


@dataclass(frozen=True)
class PromptingConfig:
    model_name: str = "distilgpt2"
    device: str | None = None                 # "cuda" or "cpu" (auto if None)

    # Prompting
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    max_length: int = 256                      # prompt truncation
    max_new_tokens: int = 8                    # only for method="generate"
    temperature: float = 0.0
    do_sample: bool = False

    # Method
    method: Literal["score", "generate"] = "score"

    # Batch / safety
    batch_size: int = 8
    fallback_label: str | None = None          # if None -> label_list[0]


class GPT2PromptingRouter:
    """
    GPT-2 prompting baseline.

    Two modes:
      - method="generate": generate a label string and parse it (more fragile)
      - method="score":   compute log-likelihood for each candidate label and pick best (recommended)
    """

    def __init__(self, label_list: list[str], cfg: PromptingConfig | None = None):
        self.label_list = label_list
        self.cfg = cfg or PromptingConfig()

        device = self.cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name)
        self.model.to(self.device)
        self.model.eval()

        self._fallback_id = self._resolve_fallback_id()

        # Pre-tokenize label strings for scoring
        self._label_token_ids: list[list[int]] = []
        for lab in self.label_list:
            # prepend a space so tokenization matches natural continuation
            ids = self.tokenizer.encode(" " + lab, add_special_tokens=False)
            self._label_token_ids.append(ids)

    def _resolve_fallback_id(self) -> int:
        if self.cfg.fallback_label is None:
            return 0
        for i, lab in enumerate(self.label_list):
            if _normalize(lab) == _normalize(self.cfg.fallback_label):
                return i
        return 0

    def _make_prompts(self, items: list[dict]) -> list[str]:
        return [
            build_prompt(
                subject=it.get("subject", ""),
                body=it.get("body", ""),
                label_list=self.label_list,
                template=self.cfg.prompt_template,
            )
            for it in items
        ]

    @torch.inference_mode()
    def _predict_generate(self, items: list[dict]) -> list[int]:
        prompts = self._make_prompts(items)

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        ).to(self.device)

        gen = self.model.generate(
            **enc,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        input_lens = enc["attention_mask"].sum(dim=1).tolist()
        preds: list[int] = []

        for i in range(len(items)):
            prompt_len = int(input_lens[i])
            gen_ids = gen[i, prompt_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            text = text.strip().splitlines()[0].strip()
            label_id = parse_department(text, self.label_list)
            preds.append(int(label_id if label_id is not None else self._fallback_id))

        return preds

    @torch.inference_mode()
    def _predict_score(self, items: list[dict]) -> list[int]:
        """
        For each item, we score each label by log-likelihood of label tokens given the prompt.
        This avoids parsing failures and is more reproducible.
        """
        prompts = self._make_prompts(items)

        # Tokenize prompts once
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        batch_size = enc["input_ids"].shape[0]
        scores = torch.zeros((batch_size, len(self.label_list)), device=self.device)

        prompt_lens = enc["attention_mask"].sum(dim=1)  # (batch,)

        for j, lab_ids in enumerate(self._label_token_ids):
            combined_ids = []
            combined_attn = []
            max_len = 0

            for i in range(batch_size):
                p_len = int(prompt_lens[i].item())
                p_ids = enc["input_ids"][i, :p_len].tolist()
                ids = p_ids + lab_ids
                max_len = max(max_len, len(ids))
                combined_ids.append(ids)

            pad_id = int(self.tokenizer.pad_token_id)
            for ids in combined_ids:
                attn = [1] * len(ids) + [0] * (max_len - len(ids))
                ids.extend([pad_id] * (max_len - len(ids)))
                combined_attn.append(attn)

            input_ids = torch.tensor(combined_ids, device=self.device)
            attention_mask = torch.tensor(combined_attn, device=self.device)

            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits  # (batch, seq, vocab)
            logp = F.log_softmax(logits, dim=-1)

            for i in range(batch_size):
                p_len = int(prompt_lens[i].item())
                ll = 0.0
                for k, tok_id in enumerate(lab_ids):
                    pred_pos = p_len - 1 + k
                    ll += float(logp[i, pred_pos, tok_id].item())
                scores[i, j] = ll

        preds = torch.argmax(scores, dim=1).detach().cpu().tolist()
        return [int(p) for p in preds]

    @torch.inference_mode()
    def predict_batch(self, items: list[dict]) -> list[int]:
        if not items:
            return []

        preds: list[int] = []
        bs = max(1, int(self.cfg.batch_size))
        for i in range(0, len(items), bs):
            chunk = items[i : i + bs]
            if self.cfg.method == "generate":
                preds.extend(self._predict_generate(chunk))
            else:
                preds.extend(self._predict_score(chunk))
        return preds

    @torch.inference_mode()
    def generate_raw(self, items: list[dict]) -> list[str]:
        """
        Generate raw GPT-2 continuations for each prompt (debug/sanity checks).
        Returns only the newly generated text, without parsing.
        """
        if not items:
            return []

        outputs: list[str] = []
        bs = max(1, int(self.cfg.batch_size))
        for i in range(0, len(items), bs):
            chunk = items[i : i + bs]
            prompts = self._make_prompts(chunk)

            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
            ).to(self.device)

            gen = self.model.generate(
                **enc,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            input_lens = enc["attention_mask"].sum(dim=1).tolist()
            for j in range(len(chunk)):
                prompt_len = int(input_lens[j])
                gen_ids = gen[j, prompt_len:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                outputs.append(text)

        return outputs
