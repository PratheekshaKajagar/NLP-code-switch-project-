"""
evaluation.py
-------------
Shared metrics + plots (confusion matrix, multi-class ROC) used by both
the TF-IDF models (train_models.py) and BERT (bert_model.py).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize

CLASSES = ["KN", "HI", "EN"]


def evaluate(name, y_true, y_pred, y_proba, out_dir, classes=CLASSES):
    """y_proba columns must follow the order of `classes`."""
    os.makedirs(out_dir, exist_ok=True)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="macro", zero_division=0)
    y_bin = label_binarize(y_true, classes=classes)
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "roc_auc_macro_ovr": float(roc_auc_score(y_bin, y_proba, average="macro")),
        "roc_auc_per_class": {
            c: float(roc_auc_score(y_bin[:, i], y_proba[:, i]))
            for i, c in enumerate(classes)
        },
    }
    report = classification_report(y_true, y_pred, labels=classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    safe = name.lower().replace(" ", "_")
    plot_confusion_matrix(cm, classes, name, os.path.join(out_dir, f"confusion_matrix_{safe}.png"))
    plot_roc(y_bin, y_proba, classes, name, os.path.join(out_dir, f"roc_curve_{safe}.png"))
    with open(os.path.join(out_dir, f"report_{safe}.txt"), "w", encoding="utf-8") as f:
        f.write(f"{name}\n{'=' * len(name)}\n{report}\nConfusion matrix (rows=true, cols=pred, order={classes})\n{cm}\n")
    with open(os.path.join(out_dir, f"metrics_{safe}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics, report, cm


def plot_confusion_matrix(cm, classes, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    for ax, data, fmt, sub in ((axes[0], cm, "d", "Counts"),
                               (axes[1], cm_norm, ".2f", "Row-normalised")):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(sub)
        thr = data.max() / 2
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > thr else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Confusion matrix - {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_bin, y_proba, classes, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC = {auc(fpr, tpr):.3f})")
    # macro-average
    grid = np.linspace(0, 1, 200)
    mean_tpr = np.mean([np.interp(grid, *roc_curve(y_bin[:, i], y_proba[:, i])[:2])
                        for i in range(len(classes))], axis=0)
    ax.plot(grid, mean_tpr, "k--", label=f"macro-avg (AUC = {auc(grid, mean_tpr):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC (one-vs-rest) - {title}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(all_metrics, path):
    names = [m["model"] for m in all_metrics]
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro_ovr"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(names)); w = 0.16
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (k, l) in enumerate(zip(keys, labels)):
        ax.bar(x + (i - 2) * w, [m[k] for m in all_metrics], w, label=l)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0.5, 1.02); ax.set_ylabel("Score (test set)")
    ax.set_title("Model comparison")
    ax.legend(ncol=5, loc="lower center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_overlay(curves, classes, path):
    """curves: {model_name: (y_bin, y_proba)} -> macro-average ROC of every model."""
    fig, ax = plt.subplots(figsize=(6, 5.2))
    grid = np.linspace(0, 1, 300)
    for name, (y_bin, y_proba) in curves.items():
        mean_tpr = np.mean([np.interp(grid, *roc_curve(y_bin[:, i], y_proba[:, i])[:2])
                            for i in range(len(classes))], axis=0)
        ax.plot(grid, mean_tpr, label=f"{name} (AUC = {auc(grid, mean_tpr):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Macro-average ROC - all models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
