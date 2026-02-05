from __future__ import annotations

import os
import time
from typing import Callable, Any

import psutil
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def ram_mb() -> float:
    """Process resident set size (RAM) in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)


def timed_predict(predict_fn: Callable[[list[Any]], list[int]], items: list[Any]) -> dict:
    """
    Run predict_fn(items) while measuring total time.
    predict_fn must return a list of predicted label ids (ints), length == len(items).
    """
    t0 = time.perf_counter()
    preds = predict_fn(items)
    t1 = time.perf_counter()

    total = t1 - t0
    per_item = total / max(1, len(items))
    return {
        "preds": preds,
        "total_sec": float(total),
        "per_item_sec": float(per_item),
        "n_items": int(len(items)),
    }


def eval_classification(y_true: list[int], y_pred: list[int], label_list: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=label_list, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": float(acc),
        "report": report,
        "confusion_matrix": cm,
    }
