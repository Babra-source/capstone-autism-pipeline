"""
Evaluation Module
=================
Computes all metrics required for the thesis:
  - Accuracy
  - Per-class Precision, Recall, F1-Score
  - Macro-averaged F1
  - Confusion Matrix (saved as PNG)
  - Full classification report (saved as CSV)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model,
                   test_loader,
                   device,
                   class_names: list,
                   mode:        str = 'face',
                   save_dir:    str = 'results',
                   experiment:  str = 'experiment') -> dict:
    """
    Run full evaluation on test_loader.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1_macro,
                    per_class (DataFrame), confusion_matrix (ndarray)
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            face  = batch['face'].to(device)
            eye   = batch['eye'].to(device)
            label = batch['label'].to(device)

            if mode == 'face':
                logits = model(face)
            elif mode == 'eye':
                logits = model(eye)
            elif mode in ('early_fusion', 'late_fusion'):
                logits = model(face, eye)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            probs = F.softmax(logits, dim=1) if mode != 'late_fusion' \
                    else torch.exp(logits)         # late fusion returns log-proba

            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro  = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class report
    report_dict = classification_report(
        y_true, 
        y_pred,
        labels=list(range(len(class_names))),  # ensures labels match the number of classes
        target_names=class_names,             # names for each class
        output_dict=True,                      # return results as a dict instead of string
        zero_division=0                        # prevents divide-by-zero errors
    )
    per_class_df = pd.DataFrame(report_dict).T.round(4)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # ── Save artefacts ────────────────────────────────────────────────────

    _save_confusion_matrix(cm, class_names, save_dir, experiment)
    _save_classification_report(per_class_df, save_dir, experiment)

    results = {
        'experiment': experiment,
        'accuracy':   round(acc,       4),
        'precision':  round(precision, 4),
        'recall':     round(recall,    4),
        'f1_macro':   round(f1_macro,  4),
        'per_class':  per_class_df,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_proba': np.array(all_probs),
    }

    # Print summary
    print(f"\n{'─'*50}")
    print(f"  {experiment}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"{'─'*50}\n")
    print(per_class_df[['precision', 'recall', 'f1-score', 'support']].to_string())

    # Save JSON summary
    summary = {k: v for k, v in results.items()
               if k not in ('per_class', 'confusion_matrix',
                             'y_true', 'y_pred', 'y_proba')}
    with open(os.path.join(save_dir, f"{experiment}_metrics.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return results


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _save_confusion_matrix(cm, class_names, save_dir, experiment):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label',      fontsize=12)
    ax.set_title(f'Confusion Matrix — {experiment}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, f"{experiment}_confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {path}")


def _save_classification_report(df, save_dir, experiment):
    path = os.path.join(save_dir, f"{experiment}_report.csv")
    df.to_csv(path)
    print(f"  Classification report saved → {path}")


def plot_training_curves(history: dict,
                         save_dir: str,
                         experiment: str):
    """Plot and save loss + accuracy curves from training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'],   label='Val',   linewidth=2, linestyle='--')
    axes[0].set_title('Loss', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'],   label='Val',   linewidth=2, linestyle='--')
    axes[1].set_title('Accuracy', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(experiment, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{experiment}_training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Training curves saved → {path}")


def compare_experiments(results_list: list,
                        save_dir:     str,
                        filename:     str = 'experiment_comparison.png'):
    """
    Bar chart comparing accuracy and F1 across all experiments.
    results_list: list of dicts with 'experiment', 'accuracy', 'f1_macro'
    """
    names    = [r['experiment'] for r in results_list]
    accs     = [r['accuracy']   for r in results_list]
    f1s      = [r['f1_macro']   for r in results_list]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, accs, w, label='Accuracy', color='#2E86AB', alpha=0.85)
    bars2 = ax.bar(x + w/2, f1s,  w, label='F1 Macro', color='#A23B72', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Experiment Comparison — Accuracy & Macro F1',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f'{h:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Comparison chart saved → {path}")
