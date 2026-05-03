"""
Eye-Tracking vs FER — Accuracy Improvement Analysis
=========================================================

Research Question:
    How can eye-tracking data (gaze direction, pupil movement) improve
    emotion recognition accuracy compared to facial expression recognition alone?

This module provides a structured comparison that answers the research question by:

1.  Head-to-head accuracy table
        For each backbone (ResNet-18, ResNet-50, ViT), compare the same
        architecture trained on face vs eye input. This isolates the
        contribution of the modality (not the model capacity).

        | Backbone     | Face (FER) Acc | Eye Acc | Δ Acc  | Eye F1 | Face F1 |
        |--------------|----------------|---------|--------|--------|---------|
        | ResNet-18    |                |         |        |        |         |
        | ResNet-50    |                |         |        |        |         |
        | ViT-B/16     |                |         |        |        |         |
        | EyeNet (CNN) |      —         |         |   —    |        |         |

2.  Improvement bar chart
        Signed Δ accuracy (eye − face) per backbone → shows when and how
        much eye data helps vs. hurts compared to FER alone.

3.  Per-class improvement heatmap
        Δ F1 score per (backbone, emotion class) — identifies which
        emotions are better detected by gaze data vs. facial expressions.

4.  Fusion lift analysis
        Compares eye-only → FER-only → fusion (early) → fusion (late).
        Shows the incremental lift from adding each modality.

5.  Statistical significance (McNemar's test)
        Test whether eye-only vs face-only prediction differences are
        statistically significant per backbone.

Outputs saved to  results/eye_vs_face/ :
    accuracy_table.csv
    accuracy_table.png
    improvement_bars.png
    per_class_delta_heatmap.png
    fusion_lift.png
    mcnemar_tests.csv
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Accuracy comparison table
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(
        face_results: dict,   # {backbone_name: metrics_dict}
        eye_results:  dict,   # {backbone_name: metrics_dict}
        save_dir:     str,
) -> pd.DataFrame:
    """
    Build a paired comparison table:
        backbone | face_acc | eye_acc | delta_acc | face_f1 | eye_f1 | delta_f1

    face_results and eye_results are dicts of the form returned by
    evaluate_model(), keyed by backbone label (e.g. 'ResNet-18').
    """
    rows = []
    all_backbones = sorted(set(list(face_results.keys()) + list(eye_results.keys())))

    for bb in all_backbones:
        face = face_results.get(bb)
        eye  = eye_results.get(bb)

        face_acc = face['accuracy']  if face else None
        eye_acc  = eye['accuracy']   if eye  else None
        face_f1  = face['f1_macro']  if face else None
        eye_f1   = eye['f1_macro']   if eye  else None

        delta_acc = round(eye_acc  - face_acc,  4) if (eye_acc  is not None and face_acc  is not None) else None
        delta_f1  = round(eye_f1   - face_f1,   4) if (eye_f1   is not None and face_f1   is not None) else None

        rows.append({
            'backbone':  bb,
            'face_acc':  face_acc,
            'eye_acc':   eye_acc,
            'Δ_acc':     delta_acc,
            'face_f1':   face_f1,
            'eye_f1':    eye_f1,
            'Δ_f1':      delta_f1,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, 'accuracy_table.csv')
    df.to_csv(csv_path, index=False)
    print(f'  Accuracy table saved → {csv_path}')

    # Pretty-print
    print(f'\n  {"Backbone":<18} {"Face Acc":>9} {"Eye Acc":>9} '
          f'{"Δ Acc":>8} {"Face F1":>9} {"Eye F1":>8} {"Δ F1":>7}')
    print('  ' + '─' * 72)
    for _, row in df.iterrows():
        def fmt(v): return f'{v:.4f}' if v is not None else '  —   '
        print(f'  {row["backbone"]:<18} {fmt(row["face_acc"])} '
              f'{fmt(row["eye_acc"])} {fmt(row["Δ_acc"])} '
              f'{fmt(row["face_f1"])} {fmt(row["eye_f1"])} {fmt(row["Δ_f1"])}')

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Improvement bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_improvement_bars(df: pd.DataFrame, save_dir: str):
    """
    Signed Δ Accuracy (eye − face) and Δ F1 (eye − face) per backbone.
    Positive = eye-tracking improves over FER alone.
    """
    sub = df.dropna(subset=['Δ_acc', 'Δ_f1'])
    if sub.empty:
        print('  No paired results — skipping improvement bars.')
        return

    x = np.arange(len(sub))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(sub) * 2), 5))

    b1 = ax.bar(x - w / 2, sub['Δ_acc'], w,
                color=['#2ecc71' if v >= 0 else '#e74c3c' for v in sub['Δ_acc']],
                alpha=0.85, label='Δ Accuracy')
    b2 = ax.bar(x + w / 2, sub['Δ_f1'],  w,
                color=['#3498db' if v >= 0 else '#e67e22' for v in sub['Δ_f1']],
                alpha=0.85, label='Δ F1 Macro')

    ax.axhline(0, color='black', linewidth=0.9, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(sub['backbone'], fontsize=11)
    ax.set_ylabel('Eye − Face (Δ score)', fontsize=12)
    ax.set_title(
        'Eye-Tracking Improvement over FER Alone\n'
        '(positive = eye data helps, negative = eye data hurts)',
        fontsize=13, fontweight='bold',
    )
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for bar_group in [b1, b2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.annotate(
                f'{h:+.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4 if h >= 0 else -12),
                textcoords='offset points',
                ha='center', fontsize=8,
            )

    plt.tight_layout()
    path = os.path.join(save_dir, 'improvement_bars.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Improvement bar chart saved → {path}')


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-class Δ F1 heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_delta_heatmap(
        face_results: dict,
        eye_results:  dict,
        class_names:  list,
        save_dir:     str,
):
    """
    Δ F1 = eye_F1(class) − face_F1(class) for each (backbone, emotion class).
    Shows which emotions benefit most from gaze data.
    """
    shared = sorted(set(face_results.keys()) & set(eye_results.keys()))
    if not shared:
        print('  No shared backbones — skipping per-class heatmap.')
        return

    delta_matrix = np.zeros((len(shared), len(class_names)))

    for i, bb in enumerate(shared):
        face_pc = face_results[bb].get('per_class')
        eye_pc  = eye_results[bb].get('per_class')
        if face_pc is None or eye_pc is None:
            continue
        for j, cn in enumerate(class_names):
            f_f1 = face_pc.loc[cn, 'f1-score'] if cn in face_pc.index else 0.0
            e_f1 = eye_pc.loc[cn,  'f1-score'] if cn in eye_pc.index  else 0.0
            delta_matrix[i, j] = round(e_f1 - f_f1, 4)

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 1.8), max(4, len(shared) * 0.9)))
    sns.heatmap(
        delta_matrix,
        xticklabels=class_names,
        yticklabels=shared,
        cmap='RdYlGn', center=0, vmin=-0.3, vmax=0.3,
        annot=True, fmt='.3f', linewidths=0.5, ax=ax,
    )
    ax.set_title(
        'Per-Class Δ F1 (Eye − Face) by Backbone\n'
        'Green = eye-tracking helps; Red = FER alone is better',
        fontsize=12, fontweight='bold',
    )
    ax.set_xlabel('Emotion Class', fontsize=11)
    ax.set_ylabel('Backbone', fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, 'per_class_delta_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Per-class Δ F1 heatmap saved → {path}')


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Fusion lift chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_fusion_lift(
        results_ordered: list,
        labels_ordered:  list,
        save_dir:        str,
):
    """
    Line plot showing incremental lift across the pipeline:
        Eye-only → FER-only → Early Fusion → Late Fusion

    results_ordered : list of metrics dicts in that order
    labels_ordered  : list of string labels
    """
    accs = [r['accuracy'] for r in results_ordered]
    f1s  = [r['f1_macro'] for r in results_ordered]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(labels_ordered, accs, '-o', linewidth=2.5, markersize=8,
            color='#4C8BBF', label='Accuracy')
    ax.plot(labels_ordered, f1s,  '-s', linewidth=2.5, markersize=8,
            color='#E67E3B', label='F1 Macro', linestyle='--')

    for x_pos, (a, f) in enumerate(zip(accs, f1s)):
        ax.annotate(f'{a:.3f}', (x_pos, a), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=9, color='#4C8BBF')
        ax.annotate(f'{f:.3f}', (x_pos, f), textcoords='offset points',
                    xytext=(0, -14), ha='center', fontsize=9, color='#E67E3B')

    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        'Fusion Lift — Eye-Only → FER-Only → Fusion',
        fontsize=13, fontweight='bold',
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, 'fusion_lift.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Fusion lift chart saved → {path}')


# ─────────────────────────────────────────────────────────────────────────────
# 5.  McNemar's test (eye vs face statistical significance)
# ─────────────────────────────────────────────────────────────────────────────

def run_mcnemar_tests(
        face_results: dict,
        eye_results:  dict,
        save_dir:     str,
) -> pd.DataFrame:
    """
    McNemar's test on paired predictions:
        H0: Eye model and Face model make the same errors.
        H1: Their error distributions differ significantly.

    A significant result (p < 0.05) supports the research question — eye-tracking captures
    different (complementary) information to facial expressions.
    """
    rows = []
    shared = sorted(set(face_results.keys()) & set(eye_results.keys()))

    for bb in shared:
        y_true = face_results[bb].get('y_true')
        yf     = face_results[bb].get('y_pred')
        ye     = eye_results[bb].get('y_pred')

        if y_true is None or yf is None or ye is None:
            continue

        correct_face = (yf == y_true)
        correct_eye  = (ye == y_true)

        # McNemar contingency:
        #   b = face wrong, eye right
        #   c = face right, eye wrong
        b = np.sum(~correct_face &  correct_eye)
        c = np.sum( correct_face & ~correct_eye)
        n = b + c

        # Exact McNemar (chi2 with continuity correction if n > 25)
        if n == 0:
            chi2, p = 0.0, 1.0
        elif n >= 25:
            chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
            from scipy.stats import chi2 as chi2_dist
            p = 1 - chi2_dist.cdf(chi2, df=1)
        else:
            from scipy.stats import binomtest
            p = min(1.0, 2 * binomtest(min(b, c), n, 0.5).pvalue)
            chi2 = None

        rows.append({
            'backbone':            bb,
            'face_correct':        int(correct_face.sum()),
            'eye_correct':         int(correct_eye.sum()),
            'both_wrong':          int((~correct_face & ~correct_eye).sum()),
            'face_right_eye_wrong': int(c),
            'face_wrong_eye_right': int(b),
            'chi2':                round(chi2, 4) if chi2 is not None else None,
            'p_value':             round(float(p), 5),
            'significant':         p < 0.05,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        csv_path = os.path.join(save_dir, 'mcnemar_tests.csv')
        df.to_csv(csv_path, index=False)
        print(f'  McNemar tests saved → {csv_path}')
        print('\n  McNemar results:')
        for _, row in df.iterrows():
            sig = '✓ significant' if row['significant'] else '✗ not significant'
            print(f'    {row["backbone"]:<18}  p={row["p_value"]:.5f}  {sig}')

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Master RQ2 runner
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(
        face_results:     dict,
        eye_results:      dict,
        class_names:      list,
        fusion_pipeline:  list = None,
        fusion_labels:    list = None,
        save_dir:         str  = 'results/eye_vs_face',
):
    """
    Full RQ2 pipeline.

    Parameters
    ----------
    face_results    : {backbone_label: evaluate_model() dict}
    eye_results     : {backbone_label: evaluate_model() dict}
    class_names     : list of emotion class names
    fusion_pipeline : ordered list of metrics dicts for fusion lift chart
    fusion_labels   : ordered list of string labels for fusion lift chart
    save_dir        : output directory
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f'\n{"="*60}')
    print('  Eye-Tracking vs FER Comparison')
    print(f'{"="*60}')

    df = build_comparison_table(face_results, eye_results, save_dir)
    plot_improvement_bars(df, save_dir)
    plot_per_class_delta_heatmap(face_results, eye_results, class_names, save_dir)
    run_mcnemar_tests(face_results, eye_results, save_dir)

    if fusion_pipeline and fusion_labels:
        plot_fusion_lift(fusion_pipeline, fusion_labels, save_dir)

    print(f'\n  Analysis complete. Outputs → {save_dir}')
