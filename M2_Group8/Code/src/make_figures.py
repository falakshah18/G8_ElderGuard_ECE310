#!/usr/bin/env python3
"""
ElderGuard - figure generation (M2)
ECE 310 Group 8

Reads Results/metrics.json (and, when present, Results/features.npz) and
writes the figures used in the report:

    model_comparison.png
    confusion_matrix_<model>.png      (one per model)
    feature_importance.png
    amp_variance_vs_people.png        (needs features.npz)

Keeping the figures derived from metrics.json means the report and the plots
can never drift apart.
"""

import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_ORDER = ["Threshold (amp-variance)", "Random Forest", "MLP"]
CLASS_NAMES = ["0_people", "1_person", "2_people"]


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_model_comparison(results, out_dir):
    models = [m for m in MODEL_ORDER if m in results] or \
             [k for k in results if not k.startswith("_")]
    acc = [results[m]["accuracy"] for m in models]
    f1 = [results[m]["macro_f1"] for m in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, acc, 0.4, label="Accuracy")
    ax.bar(x + 0.2, f1, 0.4, label="Macro F1")
    ax.axhline(1 / 3, ls="--", lw=1, color="grey", label="3-class chance")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison (leave-one-session-out)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "model_comparison.png"), dpi=150)
    plt.close(fig)


def plot_confusion(name, cm, out_dir):
    cm = np.array(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), labels=CLASS_NAMES, rotation=15)
    ax.set_yticks(range(3), labels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix - {name}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fname = f"confusion_matrix_{_slug(name)}.png"
    # Keep the exact report filename for the threshold baseline.
    if name == "Threshold (amp-variance)":
        fname = "confusion_matrix_threshold_amp-variance.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)


def plot_feature_importance(results, out_dir):
    imp = results.get("_feature_importance")
    names = results.get("_feature_names")
    if not imp or not names:
        return
    order = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(len(imp)), np.array(imp)[order])
    ax.set_xticks(np.arange(len(imp)))
    ax.set_xticklabels([names[i] for i in order], rotation=60, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)


def plot_amp_variance_vs_people(features_path, out_dir):
    if not os.path.exists(features_path):
        return False
    data = np.load(features_path)
    X, y = data["X"], data["y"]
    col = X[:, 0]  # amp_var_mean
    groups = [col[y == c] for c in (0, 1, 2)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(groups, tick_labels=CLASS_NAMES, showfliers=True)
    ax.set_ylabel("Mean amplitude variance (feature 0)")
    ax.set_title("CSI amplitude variance vs. number of people")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "amp_variance_vs_people.png"), dpi=150)
    plt.close(fig)
    return True


def make_all_figures(metrics_path, out_dir):
    with open(metrics_path) as fh:
        results = json.load(fh)
    plot_model_comparison(results, out_dir)
    for name in MODEL_ORDER:
        if name in results and "confusion_matrix" in results[name]:
            plot_confusion(name, results[name]["confusion_matrix"], out_dir)
    plot_feature_importance(results, out_dir)
    plot_amp_variance_vs_people(os.path.join(out_dir, "features.npz"), out_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="../../Results/metrics.json")
    ap.add_argument("--out", default="../../Results")
    a = ap.parse_args()
    make_all_figures(a.metrics, a.out)
    print("Figures regenerated.")
