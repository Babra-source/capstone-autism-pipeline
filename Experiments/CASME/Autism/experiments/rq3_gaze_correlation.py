"""
RQ3: Gaze Pattern and Emotional State Correlation Analysis


1.  Feature extraction
        Run each trained eye model over the test set → collect (N, D) matrix
  

2.  Top gaze features per emotion class
        Rank feature dimensions by absolute Pearson r magnitude.
        Bar charts of the top-20 most emotion-discriminative gaze dims.


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
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler



# 1.  Feature extraction


def extract_gaze_features(
        eye_model: nn.Module,
        test_loader,
        device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run eye_model over test_loader and collect feature vectors + labels.

    Parameters
    ----------
    eye_model   : any eye model with a get_features(eye_tensor) method
    test_loader : DataLoader yielding batches with keys 'eye' and 'label'
    device      : torch.device

    Returns
    -------
    features : (N, D) ndarray – gaze feature matrix
    labels   : (N,)   ndarray – integer emotion labels
    """
    eye_model.eval()
    all_feats, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            eye   = batch['eye'].to(device)
            label = batch['label']
            feat  = eye_model.get_features(eye)   # (B, D)
            all_feats.append(feat.cpu().numpy())
            all_labels.extend(label.numpy())

    features = np.concatenate(all_feats, axis=0)   # (N, D)
    labels   = np.array(all_labels)                # (N,)
    return features, labels



def compute_correlations(
        features:    np.ndarray,
        labels:      np.ndarray,
        class_names: list,
        save_dir:    str,
        model_name:  str,
) -> pd.DataFrame:
    N, D = features.shape
    C = len(class_names)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)  # (N, D)

    records = []
    pearson_matrix = np.zeros((D, C))

    for c, cname in enumerate(class_names):
        y = (labels == c).astype(float)  # one-vs-rest binary vector

        for d in range(D):
            # Check for constant input
            if np.std(X[:, d]) == 0 or np.std(y) == 0:
                pr = pp = sr = sp = np.nan
            else:
                pr, pp = pearsonr(X[:, d], y)
                sr, sp = spearmanr(X[:, d], y)

            pearson_matrix[d, c] = pr if not np.isnan(pr) else 0

            records.append({
                'class': cname,
                'feature_dim': d,
                'pearson_r': round(pr, 5) if not np.isnan(pr) else np.nan,
                'pearson_p': round(pp, 5) if not np.isnan(pp) else np.nan,
                'spearman_r': round(sr, 5) if not np.isnan(sr) else np.nan,
                'spearman_p': round(sp, 5) if not np.isnan(sp) else np.nan,
            })

    df = pd.DataFrame(records)

    # Save full correlation table
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f'gaze_correlation_summary_{model_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f'  Correlation table saved → {csv_path}')

    # Pearson heatmap — top-50 features by max absolute correlation
    max_abs = np.nan_to_num(np.abs(pearson_matrix)).max(axis=1)
    top_idx = np.argsort(max_abs)[::-1][:50]

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(8, C * 1.5), 14))
    sns.heatmap(
        pearson_matrix[top_idx],
        xticklabels=class_names,
        yticklabels=[f'feat_{i}' for i in top_idx],
        cmap='RdBu_r', center=0, vmin=-0.6, vmax=0.6,
        linewidths=0.3, annot=False, ax=ax,
    )
    ax.set_title(
        f'Pearson Correlation: Gaze Features × Emotion Class\n'
        f'(Top-50 features by max |r|) — {model_name}',
        fontsize=12, fontweight='bold',
    )
    ax.set_xlabel('Emotion Class', fontsize=11)
    ax.set_ylabel('Gaze Feature Dimension', fontsize=11)
    plt.tight_layout()
    hmap_path = os.path.join(save_dir, f'gaze_pearson_heatmap_{model_name}.png')
    fig.savefig(hmap_path, dpi=150)
    plt.close(fig)
    print(f'  Pearson heatmap saved → {hmap_path}')

    return df


def plot_top_features_per_class(
        df:          pd.DataFrame,
        class_names: list,
        save_dir:    str,
        model_name:  str,
        top_n:       int = 20,
):
    """
    For each emotion class, bar-chart the top-N gaze feature dimensions
    ranked by |Pearson r|.  Positive r → feature increases with emotion;
    negative r → feature suppressed.
    """
    fig, axes = plt.subplots(
        1, len(class_names),
        figsize=(5 * len(class_names), 6),
        sharey=False,
    )
    if len(class_names) == 1:
        axes = [axes]

    colours = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for ax, cname, colour in zip(axes, class_names, colours):
        sub = df[df['class'] == cname].copy()
        sub['abs_r'] = sub['pearson_r'].abs()
        sub = sub.nlargest(top_n, 'abs_r')
        sub = sub.sort_values('pearson_r')

        ax.barh(
            [f'f{d}' for d in sub['feature_dim']],
            sub['pearson_r'],
            color=[colour if r >= 0 else '#e74c3c' for r in sub['pearson_r']],
            alpha=0.85,
        )
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(cname, fontsize=11, fontweight='bold')
        ax.set_xlabel('Pearson r', fontsize=9)
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle(
        f'Top-{top_n} Gaze Feature Correlations per Emotion Class\n{model_name}',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    path = os.path.join(save_dir, f'gaze_top_features_{model_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Top-feature bar charts saved → {path}')







def run_rq3_analysis(
        eye_model:   nn.Module,
        model_name:  str,
        test_loader,
        device,
        class_names: list,
        save_dir:    str = 'results/rq3_gaze_correlation',
) -> dict:
    """
    Full RQ3 pipeline for one eye model.
    Call once per trained eye model (EyeNet, ResNet18Eye, ResNet50Eye, ViTEye).

    Parameters
    ----------
    eye_model   : trained eye model with get_features() method
    model_name  : string label used in output filenames
    test_loader : DataLoader for test split
    device      : torch.device
    class_names : list of emotion class name strings
    save_dir    : output directory

    Returns
    -------
    summary dict with correlation statistics
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f'\n{"="*60}')
    print(f'  RQ3 Gaze Correlation Analysis — {model_name}')
    print(f'{"="*60}')

    # 1. Extract features
    features, labels = extract_gaze_features(eye_model, test_loader, device)
    print(f'  Features extracted: {features.shape}  Labels: {labels.shape}')

    # 2. Pearson + Spearman correlations + heatmap
    corr_df = compute_correlations(features, labels, class_names,
                                   save_dir, model_name)

    # 3. Top features per class
    plot_top_features_per_class(corr_df, class_names, save_dir, model_name)


    # Save summary stats
    summary = {
        'model':       model_name,
        'n_samples':   int(len(features)),
        'feature_dim': int(features.shape[1]),
        'n_classes':   len(class_names),
        'mean_abs_pearson_per_class': {
            cname: round(float(
                corr_df[corr_df['class'] == cname]['pearson_r'].abs().mean()
            ), 5)
            for cname in class_names
        },
        'mean_abs_spearman_per_class': {
            cname: round(float(
                corr_df[corr_df['class'] == cname]['spearman_r'].abs().mean()
            ), 5)
            for cname in class_names
        },
    }
    json_path = os.path.join(save_dir, f'rq3_summary_{model_name}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  RQ3 summary saved → {json_path}')
    print(f'\n  Analysis complete. Outputs → {save_dir}')

    return summary