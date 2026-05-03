"""
Late Fusion Weight Search
=========================
Grid-searches FER_WEIGHT over [0.1, 0.2, …, 0.9] on the validation set
to find the optimal weighting between the face-FER stream and the
eye-tracking stream.

Outputs:
  - results/late_fusion_weight_search.json
  - results/late_fusion_weight_search.png  (accuracy vs weight plot)
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def weight_search(face_model,
                  eye_model,
                  val_loader,
                  device,
                  num_classes: int = 5,
                  save_dir:    str = 'results') -> float:
    """
    Returns the optimal fer_weight (float).
    """
    os.makedirs(save_dir, exist_ok=True)

    face_model.eval()
    eye_model.eval()

    # Collect all probabilities once (avoid redundant forward passes)
    all_fer_proba, all_eye_proba, all_labels = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            face  = batch['face'].to(device)
            eye   = batch['eye'].to(device)
            label = batch['label']

            fer_logits = face_model(face)
            eye_logits = eye_model(eye)

            all_fer_proba.append(F.softmax(fer_logits, dim=1).cpu().numpy())
            all_eye_proba.append(F.softmax(eye_logits, dim=1).cpu().numpy())
            all_labels.extend(label.numpy())

    fer_p = np.concatenate(all_fer_proba, axis=0)   # (N, C)
    eye_p = np.concatenate(all_eye_proba, axis=0)   # (N, C)
    y_true = np.array(all_labels)                   # (N,)

    weights   = np.arange(0.1, 1.0, 0.1)
    accs      = []
    best_w    = 0.5
    best_acc  = 0.0

    for w in weights:
        p_used = w * fer_p + (1 - w) * eye_p
        preds  = p_used.argmax(axis=1)
        acc    = (preds == y_true).mean()
        accs.append(acc)
        print(f"  fer_weight={w:.1f}  eye_weight={1-w:.1f}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_w   = round(float(w), 1)

    print(f"\n   Optimal fer_weight = {best_w}  (val_acc = {best_acc:.4f})")

    # Save results
    result = {
        'fer_weights':  weights.tolist(),
        'val_accs':     [round(a, 4) for a in accs],
        'best_fer_weight': best_w,
        'best_val_acc':    round(best_acc, 4),
    }
    with open(os.path.join(save_dir, 'late_fusion_weight_search.json'), 'w') as f:
        json.dump(result, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(weights, accs, marker='o', linewidth=2, color='#4C8BBF')
    ax.axvline(best_w, color='#E67E3B', linestyle='--',
               label=f'Optimal w={best_w}')
    ax.set_xlabel('FER_WEIGHT', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Late Fusion — Weight Search', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'late_fusion_weight_search.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Weight search plot saved → {path}")

    return best_w
