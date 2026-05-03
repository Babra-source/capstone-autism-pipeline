"""
run_all.py — Master Experiment Script


Experiment Structure
--------------------
Exp 1  - FER-only models (face stream):
    1a. ResNet-18
    1b. ResNet-50
    1c. ViT-B/16
    1d. MultiScaleViT

Exp 2  - Eye-only models (RQ2):
    2a. ResNet-18 on eye crops
    2b. ResNet-50 on eye crops
    2c. ViT-B/16 on eye crops

Exp 3  - Early Fusion (RQ1):
    3a. ResNet-18 face + ResNet-18 eye -> concat head

Exp 4  - Late Fusion (RQ1):
    4a. Weight search (val set)
    4b. Best FER + best eye model -> weighted probabilities

RQ2 analysis:  results/eye_vs_face/
RQ3 analysis:  results/rq3_gaze_correlation/


"""

import os
import sys
import argparse
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.casme2_dataset         import get_dataloaders, FOUR_CLASS_NAMES, FIVE_CLASS_NAMES
from models.architectures         import (
    ResNet18FER, ResNet50FER, ViTFER, MultiScaleViTFER,
    ResNet18Eye, ResNet50Eye, ViTEye,
    EarlyFusionResNet, LateFusionModel,
)
from experiments.trainer          import Trainer
from experiments.evaluate         import (
    evaluate_model, plot_training_curves, compare_experiments,
)
from experiments.weight_search    import weight_search
from experiments.rq2_eye_vs_face  import run_analysis
from experiments.rq3_gaze_correlation import run_rq3_analysis


DEFAULT_DATA_DIR   = "Autism/data/CASME2_Compressed_video/CASME2_compressed"
DEFAULT_LABEL_FILE = "Autism/data/CASME2-ObjectiveClasses.xlsx"




def parse_args():
    p = argparse.ArgumentParser(
        description='FER + Eye-Tracking Thesis — All Experiments'
    )
    p.add_argument('--data_dir',    default=DEFAULT_DATA_DIR)
    p.add_argument('--label_file',  default=DEFAULT_LABEL_FILE)
    p.add_argument('--results_dir', default='results')
    p.add_argument('--num_classes', type=int, default=5, choices=[4, 5])
    p.add_argument('--epochs',      type=int, default=50)
    p.add_argument('--batch_size',  type=int, default=16)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--patience',    type=int, default=10)
    p.add_argument('--val_subjects',  nargs='+', type=int, default=[24, 25])
    p.add_argument('--test_subjects', nargs='+', type=int, default=[26])
    p.add_argument('--skip_exp1',       action='store_true',
                   help='Skip Exp 1 FER models')
    p.add_argument('--skip_exp2_extra', action='store_true',
                   help='Skip Exp 2b/2c (ResNet50Eye, ViTEye)')
    p.add_argument('--skip_rq3',        action='store_true',
                   help='Skip RQ3 gaze correlation analysis (saves time)')
    return p.parse_args()



def run_experiment(
        model,
        exp_name:    str,
        mode:        str,
        loaders:     dict,
        class_names: list,
        device,
        args,
        exp_dir:     str,
        lr=1e-4
) -> dict:
    """
    Train and evaluate a single model. Returns the evaluate_model() dict.
    mode: 'face' | 'eye' | 'early_fusion' | 'late_fusion'
    """
    os.makedirs(exp_dir, exist_ok=True)

    trainer = Trainer(
        model        = model,
        train_loader = loaders['train'],
        val_loader   = loaders['val'],
        num_classes  = args.num_classes,
        device       = device,
        save_dir     = exp_dir,
        experiment   = exp_name,
        mode         = mode,
        lr           = lr,
        epochs       = args.epochs,
        patience     = args.patience,
    )
    history = trainer.train()
    plot_training_curves(history, exp_dir, exp_name)

    results = evaluate_model(
        model       = model,
        test_loader = loaders['test'],
        device      = device,
        class_names = class_names,
        mode        = mode,
        save_dir    = exp_dir,
        experiment  = exp_name,
    )
    return results


def _load_ckpt(model, ckpt_path: str, device, label: str):
    """Load a checkpoint if it exists, otherwise warn."""
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f'  Loaded {label} weights from {ckpt_path}')
    else:
        print(f'  WARNING: {label} checkpoint not found at {ckpt_path} — using random init.')




def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device: {device}')

    class_names = FOUR_CLASS_NAMES if args.num_classes == 4 else FIVE_CLASS_NAMES
    R = args.results_dir
    os.makedirs(R, exist_ok=True)

    # ── DataLoaders 
    print('\n  Loading CASME2 dataset ...')
    loaders = get_dataloaders(
        root_dir      = args.data_dir,
        label_file    = args.label_file,
        num_classes   = args.num_classes,
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
        val_subjects  = args.val_subjects,
        test_subjects = args.test_subjects,
    )

    all_results  = []   # all experiments (for global comparison chart)
    face_results = {}   # backbone -> metrics  (for RQ2)
    eye_results  = {}   # backbone -> metrics  (for RQ2)
    
    exp1_models = [
        ('exp1a_resnet18',  ResNet18FER,      'ResNet-18',      1e-3),
        ('exp1b_resnet50',  ResNet50FER,      'ResNet-50',      5e-4),
        ('exp1c_vit',       ViTFER,           'ViT-B/16',       5e-5),
        ('exp1d_ms_vit',    MultiScaleViTFER, 'MultiScale-ViT', 2e-5),
    ]
    
    for exp_name, model_cls, bb_label, lr in exp1_models:  # unpack lr
        model = model_cls(num_classes=args.num_classes).to(device)
        r = run_experiment(
            model, exp_name, 'face', loaders, class_names,
            device, args, os.path.join(R, 'exp1_fer', exp_name),
            lr=lr,
    )
    # 
    # EXPERIMENT 1 — FER-only models  (face stream)
    # 
    print('\n' + '=' * 64)
    print('  EXPERIMENT 1 — FER Pipeline  (face-only models)')
    print('=' * 64)

    exp1_models = [
        ('exp1a_resnet18',  ResNet18FER,      'ResNet-18',      1e-3),
        ('exp1b_resnet50',  ResNet50FER,      'ResNet-50',      5e-4),
        ('exp1c_vit',       ViTFER,           'ViT-B/16',       5e-5),
        ('exp1d_ms_vit',    MultiScaleViTFER, 'MultiScale-ViT', 2e-5),
    ]
    
    if args.skip_exp1:
        print('  (skipped via --skip_exp1)')
    else:
        for exp_name, model_cls, bb_label, lr in exp1_models:
            print(f'\n  -> {exp_name}  [{bb_label}]')
            model = model_cls(num_classes=args.num_classes).to(device)
            r = run_experiment(
                model, exp_name, 'face', loaders, class_names, device,
                args, os.path.join(R, 'exp1_fer', exp_name),
                lr=lr,
            )
            all_results.append(r)
            face_results[bb_label] = r

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2 — Eye-only models  (eye stream)
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 64)
    print('  EXPERIMENT 2 — Eye-Tracking Pipeline  (eye-only models)')
    print('=' * 64)

    # 2a: ResNet18Eye baseline — always runs, needed for fusion
    print('\n  -> exp2a_resnet18_eye  [ResNet-18Eye]')
    resnet18_eye = ResNet18Eye(num_classes=args.num_classes).to(device)
    r_resnet18_eye = run_experiment(
        resnet18_eye, 'exp2a_resnet18_eye', 'eye', loaders, class_names, device,
        args, os.path.join(R, 'exp2_eye', 'exp2a_resnet18_eye'),
        lr=1e-3,
    )
    all_results.append(r_resnet18_eye)
    eye_results['ResNet-18'] = r_resnet18_eye
    trained_eye_models = {'ResNet-18': resnet18_eye}
    
    eye_extra_models = [
        ('exp2b_resnet50_eye', ResNet50Eye, 'ResNet-50', 5e-4),
        ('exp2c_vit_eye',      ViTEye,      'ViT-B/16',  5e-5),
    ]
    
    if args.skip_exp2_extra:
        print('  (exp2b/2c skipped via --skip_exp2_extra)')
    else:
        for exp_name, model_cls, bb_label, lr in eye_extra_models:
            print(f'\n  -> {exp_name}  [{bb_label}]')
            model = model_cls(num_classes=args.num_classes).to(device)
            r = run_experiment(
                model, exp_name, 'eye', loaders, class_names,
                device, args, os.path.join(R, 'exp2_eye', exp_name),
                lr=lr,
            )
            all_results.append(r)
            eye_results[bb_label] = r
            trained_eye_models[bb_label] = model
        # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 3 — Early Fusion  (RQ1)
    # ══════════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 64)
    print('  EXPERIMENT 3 — Early Fusion  (ResNet-18 face + ResNet-18 eye -> concat head)')
    print('=' * 64)
    
    early_model = EarlyFusionResNet(num_classes=args.num_classes).to(device)
    r_early = run_experiment(
        early_model, 'exp3_early_fusion', 'early_fusion',
        loaders, class_names, device,
        args, os.path.join(R, 'exp3_early'),
        lr=5e-4,
    )
   
    # EXPERIMENT 4 — Late Fusion  (RQ1)

    print('\n' + '=' * 64)
    print('  EXPERIMENT 4 — Late Fusion  (best FER + best eye, optimal weights)')
    print('=' * 64)

    # Load best FER model checkpoint (ViT-B/16 from Exp 1c)
    face_vit = ViTFER(num_classes=args.num_classes).to(device)
    _load_ckpt(
        face_vit,
        os.path.join(R, 'exp1_fer', 'exp1c_vit', 'exp1c_vit_best.pth'),
        device, 'ViT-FER',
    )

    # Load best eye model checkpoint (ResNet18Eye from Exp 2a)
    best_eye = ResNet18Eye(num_classes=args.num_classes).to(device)
    _load_ckpt(
        best_eye,
        os.path.join(R, 'exp2_eye', 'exp2a_resnet18_eye', 'exp2a_resnet18_eye_best.pth'),
        device, 'ResNet18Eye',
    )

    exp4_dir = os.path.join(R, 'exp4_late')
    best_fer_w = weight_search(
        face_model  = face_vit,
        eye_model   = best_eye,
        val_loader  = loaders['val'],
        device      = device,
        num_classes = args.num_classes,
        save_dir    = exp4_dir,
    )

    late_model = LateFusionModel(
        face_model  = face_vit,
        eye_model   = best_eye,
        num_classes = args.num_classes,
        fer_weight  = best_fer_w,
    ).to(device)

    os.makedirs(exp4_dir, exist_ok=True)
    r_late = evaluate_model(
        model       = late_model,
        test_loader = loaders['test'],
        device      = device,
        class_names = class_names,
        mode        = 'late_fusion',
        save_dir    = exp4_dir,
        experiment  = 'exp4_late_fusion',
    )
    all_results.append(r_late)

    # RQ1 — Global comparison chart

    print('\n' + '=' * 64)
    print('  RQ1 SUMMARY — All Experiments')
    print('=' * 64)

    print(f'\n  {"Experiment":<35} {"Accuracy":>10} {"F1 Macro":>10}')
    print('  ' + '-' * 57)
    for r in all_results:
        print(f'  {r["experiment"]:<35} {r["accuracy"]:>10.4f} {r["f1_macro"]:>10.4f}')

    compare_experiments(
        results_list = all_results,
        save_dir     = R,
        filename     = 'rq1_all_experiments_comparison.png',
    )

    
    # RQ2 — Eye-tracking vs FER comparison

    print('\n' + '=' * 64)
    print('  RQ2 — Eye-Tracking vs FER Comparison')
    print('=' * 64)

    # Fusion lift: ResNet18Eye -> ViT FER -> Early Fusion -> Late Fusion
    fusion_pipeline = [
        eye_results['ResNet-18'],
        face_results.get('ViT-B/16', face_results.get('ResNet-18')),
        r_early,
        r_late,
    ]
    fusion_labels = [
        'Eye-Only\n(ResNet-18)',
        'FER-Only\n(ViT)',
        'Early\nFusion',
        'Late\nFusion',
    ]

    run_analysis(
        face_results    = face_results,
        eye_results     = eye_results,
        class_names     = class_names,
        fusion_pipeline = fusion_pipeline,
        fusion_labels   = fusion_labels,
        save_dir        = os.path.join(R, 'eye_vs_face'),
    )


    # RQ3 — Gaze pattern <-> emotion correlation
   
    if not args.skip_rq3:
        print('\n' + '=' * 64)
        print('  RQ3 — Gaze Pattern <-> Emotion Correlation Analysis')
        print('=' * 64)

        rq3_dir       = os.path.join(R, 'rq3_gaze_correlation')
        rq3_summaries = []

        for model_label, eye_m in trained_eye_models.items():
            safe_name = model_label.replace('/', '_').replace(' ', '_').lower()
            summary = run_rq3_analysis(
                eye_model   = eye_m,
                model_name  = safe_name,
                test_loader = loaders['test'],
                device      = device,
                class_names = class_names,
                save_dir    = rq3_dir,
            )
            rq3_summaries.append(summary)
    else:
        print('\n  RQ3 analysis skipped (--skip_rq3).')

    # Master summary JSON

    summary_data = [
        {k: v for k, v in r.items()
         if k not in ('per_class', 'confusion_matrix', 'y_true', 'y_pred', 'y_proba')}
        for r in all_results
    ]
    with open(os.path.join(R, 'master_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f'\n  Master summary -> {R}/master_summary.json')
    print('\n  All experiments complete.\n')


if __name__ == '__main__':
    main()