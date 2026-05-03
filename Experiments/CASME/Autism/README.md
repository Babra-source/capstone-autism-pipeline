# FER + Eye-Tracking Thesis — Emotion Recognition in Minimal Verbal Autistic Students

**Thesis title:** Evaluating the Effectiveness of Facial Expression Recognition and Eye-Tracking Techniques in Detecting Emotional Cues of Minimal Verbal Autistic Students.

---

## Research Questions & Experiment Mapping

| RQ | Question | Answered by |
|----|----------|-------------|
| **RQ1** | How effective is the integration of FER and eye-tracking in detecting autism emotional cues? | Exp 3 (Early Fusion) + Exp 4 (Late Fusion) vs. single-stream baselines → `rq1_all_experiments_comparison.png` |
| **RQ2** | How can eye-tracking data improve emotion recognition accuracy compared to FER alone? | Exp 2b/2c/2d (eye models on same backbones as FER) → `rq2_eye_vs_face/` |
| **RQ3** | What is the correlation between gaze patterns and emotional states? | Pearson/Spearman correlations, t-SNE, spatial heatmaps, radar charts → `rq3_gaze_correlation/` |

---

## Project Structure

```
fer_project/
├── data/
│   └── casme2_dataset.py          CASME2 loader (AVI → frames → tensors)
├── models/
│   └── architectures.py           All model classes (see table below)
├── experiments/
│   ├── trainer.py                 Training engine (early stopping, class weights)
│   ├── evaluate.py                Accuracy, F1, confusion matrix, comparison charts
│   ├── weight_search.py           Late fusion weight grid search
│   ├── rq2_eye_vs_face.py         RQ2 eye vs face comparison analysis  ← NEW
│   └── rq3_gaze_correlation.py    RQ3 gaze–emotion correlation analysis ← NEW
├── gradcam/
│   └── gradcam.py                 Grad-CAM for CNN and ViT models
├── results/                       Auto-created — all outputs saved here
├── run_all.py                     Master script — runs all experiments + RQ analyses
└── requirements.txt
```

---

## Models

### Experiment 1 — FER-only (face stream)

| ID   | Class             | Backbone         | Notes                           |
|------|-------------------|------------------|---------------------------------|
| 1a   | `ResNet18FER`     | ResNet-18        | ImageNet pretrained             |
| 1b   | `ResNet50FER`     | ResNet-50        | ImageNet pretrained             |
| 1c   | `ViTFER`          | ViT-B/16         | Patch attention on face         |
| 1d   | `MultiScaleViTFER`| ViT-B/16 × 2    | **NEW** — dual-resolution cross-attention fusion. Two ViTs process global (224px) and local centre-cropped (112→224px) face. Captures subtle micro-expressions missed by single-scale models. |

### Experiment 2 — Eye-only (eye stream, RQ2)

| ID   | Class           | Backbone      | Notes                                              |
|------|-----------------|---------------|----------------------------------------------------|
| 2a   | `EyeNet`        | Custom CNN    | Baseline lightweight model, feature dim: 256       |
| 2b   | `ResNet18Eye`   | ResNet-18     | **NEW** — same backbone as 1a for fair comparison  |
| 2c   | `ResNet50Eye`   | ResNet-50     | **NEW** — same backbone as 1b for fair comparison  |
| 2d   | `ViTEye`        | ViT-B/16      | **NEW** — attention across gaze patches, feature dim: 768 |

> **Why RQ2 needs controlled backbone comparison:** If EyeNet (custom CNN) under-performs ViTFER, it could be model capacity, not the modality. By running ResNet18Eye vs ResNet18FER (same backbone, different input), any accuracy delta is attributable purely to whether face or eye information is more discriminative for each emotion class.

### Experiments 3 & 4 — Fusion (RQ1)

| ID   | Class              | Inputs                   | Strategy |
|------|--------------------|--------------------------|----------|
| 3    | `EarlyFusionViT`   | ViT face + EyeNet eye    | Feature concat → joint head |
| 4    | `LateFusionModel`  | Any face + any eye model | Weighted softmax combination; weight found by grid search |

---

## MultiScaleViT Architecture Detail (Exp 1d)

```
Input face (224×224)
    │
    ├──→ ViT-global encoder ──→ CLS_global (768-d)
    │                                   │
    └──→ centre-crop (112px) ──→ resize ──→ ViT-local encoder ──→ CLS_local (768-d)
                                                                          │
                                               Cross-Attention: global queries local
                                                                          │
                                                          CLS_fused (768-d) + residual
                                                                          │
                                              concat [CLS_fused ‖ CLS_local] (1536-d)
                                                                          │
                                                             Classifier head → logits
```
Rationale: Autistic micro-expressions are spatially subtle. The local stream
zooms 2× on the perioral/periorbital region — the most information-dense zone
for micro-expression recognition.

---

## RQ2 Outputs (`results/rq2_eye_vs_face/`)

| File | Description |
|------|-------------|
| `rq2_accuracy_table.csv` | Paired face vs eye accuracy per backbone |
| `rq2_improvement_bars.png` | Signed Δ accuracy (eye − face) bar chart |
| `rq2_per_class_delta_heatmap.png` | Δ F1 per (backbone, emotion class) |
| `rq2_fusion_lift.png` | Eye-only → FER-only → Fusion progression |
| `rq2_mcnemar_tests.csv` | Statistical significance of face vs eye differences |

---

## RQ3 Outputs (`results/rq3_gaze_correlation/`)

| File | Description |
|------|-------------|
| `gaze_tsne_<model>.png` | t-SNE of gaze features coloured by emotion class |
| `gaze_pearson_heatmap_<model>.png` | Top-50 gaze dims × emotion class Pearson r matrix |
| `gaze_top_features_<model>.png` | Top-20 feature correlations per emotion class |
| `gaze_class_profiles_<model>.png` | Radar chart of per-class gaze feature profiles |
| `gaze_spatial_heatmap_<model>.png` | Spatial conv activations per class (CNN models) |
| `gaze_correlation_summary_<model>.csv` | Full Pearson + Spearman table |
| `rq3_model_comparison.png` | Mean |r| per class across all eye models |

---

## Setup

```bash
pip install -r requirements.txt

# Dataset:
# CASME2_compressed/   sub01/ … sub26/   (AVI clips at 200fps)
# CASME2-ObjectiveClasses.xlsx           (label file)
```

---

## Running

```bash
# Full run (all experiments + all RQ analyses)
python run_all.py \
    --data_dir   /path/to/CASME2_compressed \
    --label_file /path/to/CASME2-ObjectiveClasses.xlsx \
    --num_classes 5 --epochs 50 --batch_size 16

# Smoke test (2 epochs, no Grad-CAM, no RQ3)
python run_all.py \
    --data_dir   /path/to/CASME2_compressed \
    --label_file /path/to/CASME2-ObjectiveClasses.xlsx \
    --epochs 2 --skip_gradcam --skip_rq3

# Skip slow exp2 backbone models (keeps EyeNet, skips ResNet18/50/ViT eye)
python run_all.py ... --skip_exp2_extra
```

---

## Dataset: CASME2

- 26 subjects, 247 micro-expression video clips at 200 fps
- 5-class: disgust, happiness, repression, surprise, others
- 4-class: positive, negative, surprise, others
- Split: train = sub01–sub23, val = sub24–sub25, test = sub26

> **Thesis note:** CASME2 contains neurotypical adult subjects. For a final autism-specific deployment, data collected from autistic students (with appropriate ethics approval) should be used for fine-tuning. The CASME2 experiments establish a general-purpose benchmark and demonstrate the technical pipeline's effectiveness.

---

## Citation

Yan, W.J. et al. (2014). CASME II: An Improved Spontaneous Micro-Expression Database and the Baseline Evaluation. *PLOS ONE*.
