from __future__ import annotations
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, recall_score,
)
from src.data.dataset import GENDER_TO_IDX


def compute_metrics(y_true, y_pred) -> dict:
    male_idx = GENDER_TO_IDX["male"]
    female_idx = GENDER_TO_IDX["female"]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall_male": recall_score(y_true, y_pred, pos_label=male_idx, average="binary",
                                     zero_division=0),
        "recall_female": recall_score(y_true, y_pred, pos_label=female_idx, average="binary",
                                       zero_division=0),
    }
