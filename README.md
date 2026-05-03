

## Project Overview
**Evaluating the Effectiveness of Facial Expression Recognition and Eye Tracking in Detecting Emotional Cues of Minimal Verbal Autistic Students**

This project implements a multimodal emotion detection system that combines **Facial Expression Recognition (FER)** and **Eye Tracking** to detect emotional cues in minimally verbal autistic students. The system evaluates two fusion strategies: early fusion and late fusion , across two datasets: the Mendeley Facial Emotion Recognition Dataset and the CASME II micro-expression dataset.

---

## Repository Structure

```
capstone-autism-pipeline/
├── Experiments/
│   ├── CASME/Autism/
│   │   ├── .ipynb_checkpoints/
│   │   ├── experiments/
│   │   ├── models/
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   └── run_all.py
│   └── MEDLEY/
│       ├── Early_Fusion.ipynb
│       ├── FinalFER (2).ipynb
│       ├── eye_tracking (2).ipynb
│       └── late_fusion (1).ipynb
└── .gitignore
```

---

## Datasets

| Dataset | Type | Description |
|--------|------|-------------|
| [Mendeley FER Dataset](https://data.mendeley.com/datasets/b33pf78h62/1) | Images | Facial emotion images for children with Autism. 6 labels: Happy, Sad, Neutral, Joy, Anger, Fear |
| [CASME II](https://doi.org/10.1371/journal.pone.0086041) | Videos | Spontaneous micro-expression video clips. 5 labels: Disgust, Happiness, Repression, Surprise, Others |

> **Note:** Datasets are not included in this repository. Download them separately from the links above.

---

## Requirements

**Python:** 3.12  
**Environment:** Virtual environment (`python-env`)  
**CUDA:** Supported (cuda-toolkit 13.0.2)

Install the required packages:

```bash
pip install torch torchvision
pip install mediapipe
pip install scikit-learn xgboost
pip install opencv-python opencv-contrib-python
pip install matplotlib seaborn
pip install pandas numpy
pip install timm
pip install grad-cam
pip install tqdm pillow scipy
```

### Key Packages

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.11.0 | Deep learning framework |
| torchvision | 0.26.0 | Image transforms and pretrained models |
| mediapipe | 0.10.33 | Facial landmark detection and eye region extraction |
| scikit-learn | 1.8.0 | Machine learning classifiers (Random Forest, SVM) |
| xgboost | 3.2.0 | XGBoost classifier for eye tracking pipeline |
| opencv-python | 4.13.0.92 | Image and video processing |
| numpy | 2.4.4 | Numerical computation |
| pandas | 3.0.2 | Data handling and analysis |
| matplotlib | 3.10.8 | Plotting and visualisation |
| seaborn | 0.13.2 | Statistical visualisation |
| timm | 1.0.26 | Vision Transformer and MultiScale-ViT models |
| grad-cam | 1.5.5 | Gradient-weighted class activation mapping |
| scipy | 1.17.1 | Statistical analysis |
| pillow | 12.2.0 | Image loading and preprocessing |
| tqdm | 4.67.3 | Training progress bars |

---

## How to Run

**CASME II Dataset**
```bash
python run_all.py
```

**Mendeley Dataset**  
Open the notebooks inside `Experiments/MEDLEY/` in Jupyter and run them using the **Run button** in this order:
1. `FinalFER (2).ipynb`
2. `eye_tracking (2).ipynb`
3. `Early_Fusion.ipynb`
4. `late_fusion (1).ipynb`


## AI Tool Usage

This project used AI assistance (Claude AI — Sonnet 4.6, January–April 2026) for code restructuring, fusion implementation and parameter tuning. All AI-generated outputs were reviewed, tested and adapted by the author. Full details are documented in the thesis appendix.

---

## License
 Not licensed for commercial use.
