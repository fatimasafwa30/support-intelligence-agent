"""Evaluate confidence abstention thresholds for the Intent Classifier.

Computes classification metrics across candidate abstention thresholds on:
1. Validation Split (DEV data: pure specific, full DEV, and realistic 10% unclear blend)
2. Final Benchmark (Human Golden Set: 250 samples)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intents.classifier import IntentClassifier, PredictionResult
from src.intents.silver_labeler import CANONICAL_INTENTS, SilverLabeler

CANDIDATE_THRESHOLDS: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def evaluate_dataset_at_thresholds(
    texts: Sequence[str],
    y_true: Sequence[str],
    clf: IntentClassifier,
    thresholds: Sequence[float | None] = (None, *CANDIDATE_THRESHOLDS),
    classes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a dataset across candidate confidence abstention thresholds."""
    if classes is None:
        target_classes = list(clf.classes_)
        if "other_unclear" not in target_classes:
            target_classes.append("other_unclear")
    else:
        target_classes = list(classes)

    raw_preds: list[PredictionResult] = clf.predict(texts)
    confs = [p.confidence for p in raw_preds]
    raw_intents = [p.intent for p in raw_preds]

    results: list[dict[str, Any]] = []
    unc_idx = target_classes.index("other_unclear")

    for t in thresholds:
        y_pred = [
            "other_unclear" if (t is not None and c < t) else i
            for c, i in zip(confs, raw_intents)
        ]
        acc = float(accuracy_score(y_true, y_pred))
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=target_classes, average="macro", zero_division=0
        )
        p_per, r_per, f1_per, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=target_classes, zero_division=0
        )
        unc_p, unc_r, unc_f1 = float(p_per[unc_idx]), float(r_per[unc_idx]), float(f1_per[unc_idx])

        abstained_count = sum(1 for yp in y_pred if yp == "other_unclear")
        coverage = float((len(y_pred) - abstained_count) / len(y_pred)) if y_pred else 0.0

        results.append({
            "threshold": t,
            "threshold_label": "None" if t is None else f"{t:.2f}",
            "accuracy": acc,
            "macro_f1": float(macro_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "other_unclear_precision": unc_p,
            "other_unclear_recall": unc_r,
            "other_unclear_f1": unc_f1,
            "coverage": coverage,
            "abstained_count": abstained_count,
            "total_samples": len(y_pred),
        })

    return results


def format_table(results: list[dict[str, Any]], title: str) -> str:
    """Format evaluation results into a clear markdown-style table."""
    lines = [
        "=" * 105,
        title,
        "=" * 105,
        f"{'Threshold':<11} | {'Accuracy':<9} | {'Macro-F1':<9} | {'Unclear P':<10} | {'Unclear R':<10} | {'Unclear F1':<11} | {'Coverage':<9} | {'Abstained':<9}",
        "-" * 105,
    ]
    for r in results:
        lines.append(
            f"{r['threshold_label']:<11} | "
            f"{r['accuracy']*100:6.2f}%   | "
            f"{r['macro_f1']*100:6.2f}%   | "
            f"{r['other_unclear_precision']*100:7.2f}%   | "
            f"{r['other_unclear_recall']*100:7.2f}%   | "
            f"{r['other_unclear_f1']*100:7.2f}%    | "
            f"{r['coverage']*100:6.2f}%   | "
            f"{r['abstained_count']:>5d} / {r['total_samples']}"
        )
    lines.append("=" * 105)
    return "\n".join(lines)


def main() -> None:
    clf = IntentClassifier.load(PROJECT_ROOT / "data" / "processed" / "models" / "intent_classifier_baseline.joblib")
    labeler = SilverLabeler()

    # 1. Load DEV split conversations and label
    with open(PROJECT_ROOT / "data" / "processed" / "conversation_splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    dev_cids = set(splits["dev_conversation_ids"])

    specific_dev: list[tuple[str, str]] = []
    unclear_dev: list[tuple[str, str]] = []

    with open(PROJECT_ROOT / "data" / "processed" / "apple_conversations.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["conversation_id"] in dev_cids:
                for tw in c.get("tweets", []):
                    if tw.get("inbound"):
                        txt = str(tw.get("text") or "").strip()
                        if txt:
                            lbl = labeler.label(txt)["intent"]
                            if lbl != "other_unclear":
                                specific_dev.append((txt, lbl))
                            else:
                                unclear_dev.append((txt, lbl))

    # Balanced DEV validation blend matching the ~10% unclear ratio in real evaluation traffic
    random.seed(42)
    sample_size = min(len(unclear_dev), int(len(specific_dev) * 0.10))
    sampled_unclear = random.sample(unclear_dev, sample_size)
    blend_dev = specific_dev + sampled_unclear

    # 2. Evaluate on Validation DEV Blend
    blend_results = evaluate_dataset_at_thresholds(
        texts=[s[0] for s in blend_dev],
        y_true=[s[1] for s in blend_dev],
        clf=clf,
    )
    print("\n" + format_table(blend_results, f"VALIDATION DATA: DEV SPLIT REPRESENTATIVE BLEND (N={len(blend_dev)}: {len(specific_dev)} Specific + {sample_size} Unclear)"))

    # 3. Evaluate on Human Golden Set (250 examples)
    golden_df = pd.read_csv(PROJECT_ROOT / "data" / "golden" / "golden_annotation.csv")
    golden_results = evaluate_dataset_at_thresholds(
        texts=golden_df["text"].tolist(),
        y_true=golden_df["gold_intent"].tolist(),
        clf=clf,
    )
    print("\n" + format_table(golden_results, f"TEST BENCHMARK: HUMAN GOLDEN SET (N={len(golden_df)}: 225 Specific + 25 Unclear)"))


if __name__ == "__main__":
    main()
