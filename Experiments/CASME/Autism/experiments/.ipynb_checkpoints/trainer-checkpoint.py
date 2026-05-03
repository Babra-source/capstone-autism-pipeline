"""
Training Engine
===============
Handles training, validation, and checkpointing for all experiments.
Supports both single-stream (face-only / eye-only) and fusion models.
"""

import os
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight


# ─────────────────────────────────────────────────────────────────────────────

def get_class_weights(dataset, num_classes: int, device) -> torch.Tensor:
    """Compute inverse-frequency class weights to handle class imbalance."""
    labels = [s['label'] for s in dataset.samples]
    classes = np.arange(num_classes)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def get_optimizer(model: nn.Module, lr: float = 1e-4,
                  weight_decay: float = 0.01) -> optim.Optimizer:
    """AdamW with differential LR: backbone gets lower LR than head."""
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if 'classifier' in name or 'head' in name or 'fc' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    return optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1},
        {'params': head_params,     'lr': lr},
    ], weight_decay=weight_decay)


# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    General-purpose trainer for single-stream and fusion models.

    Args:
        model:         nn.Module to train
        train_loader:  DataLoader for training set
        val_loader:    DataLoader for validation set
        num_classes:   number of emotion classes
        device:        torch device
        save_dir:      directory to save checkpoints and logs
        experiment:    string identifier (e.g. 'exp1_resnet18')
        mode:          'face' | 'eye' | 'early_fusion' | 'late_fusion'
        lr:            initial learning rate
        epochs:        maximum training epochs
        patience:      early stopping patience
    """

    def __init__(self,
                 model,
                 train_loader: DataLoader,
                 val_loader:   DataLoader,
                 num_classes:  int = 5,
                 device=None,
                 save_dir:     str = 'results',
                 experiment:   str = 'experiment',
                 mode:         str = 'face',
                 lr:           float = 1e-4,
                 epochs:       int = 50,
                 patience:     int = 10):

        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.num_classes  = num_classes
        self.device       = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir     = save_dir
        self.experiment   = experiment
        self.mode         = mode
        self.epochs       = epochs
        self.patience     = patience

        os.makedirs(save_dir, exist_ok=True)
        self.model.to(self.device)

        # Class-weighted loss
        try:
            cw = get_class_weights(train_loader.dataset, num_classes, self.device)
        except Exception:
            cw = None

        if mode == 'late_fusion':
            self.criterion = nn.NLLLoss(weight=cw)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=cw)

        self.optimizer = get_optimizer(model, lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        self.history = {'train_loss': [], 'train_acc': [],
                        'val_loss':   [], 'val_acc':   []}
        self.best_val_acc  = 0.0
        self.best_weights  = None
        self.patience_ctr  = 0



    def _unpack(self, batch):
        face  = batch['face'].to(self.device)
        eye   = batch['eye'].to(self.device)
        label = batch['label'].to(self.device)
        return face, eye, label

    def _forward(self, face, eye):
        if self.mode == 'face':
            return self.model(face)
        if self.mode == 'eye':
            return self.model(eye)
        if self.mode in ('early_fusion', 'late_fusion'):
            return self.model(face, eye)
        raise ValueError(f"Unknown mode: {self.mode}")

    # ── One epoch ────────────────────────────────────────────────────────

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for batch in loader:
                face, eye, label = self._unpack(batch)
                logits = self._forward(face, eye)
                loss   = self.criterion(logits, label)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                preds = logits.argmax(dim=1)
                total_loss += loss.item() * label.size(0)
                correct    += (preds == label).sum().item()
                total      += label.size(0)

        return total_loss / total, correct / total

    # ── Main train loop ──────────────────────────────────────────────────

    def train(self):
        print(f"\n{'='*60}")
        print(f"  Experiment : {self.experiment}")
        print(f"  Mode       : {self.mode}")
        print(f"  Device     : {self.device}")
        print(f"  Epochs     : {self.epochs}   Patience: {self.patience}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._run_epoch(self.train_loader, train=True)
            va_loss, va_acc = self._run_epoch(self.val_loader,   train=False)
            self.scheduler.step()

            self.history['train_loss'].append(tr_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_loss'].append(va_loss)
            self.history['val_acc'].append(va_acc)

            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{self.epochs}  "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                  f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  "
                  f"({elapsed:.1f}s)")

            # Checkpoint
            if va_acc > self.best_val_acc:
                self.best_val_acc = va_acc
                self.best_weights = copy.deepcopy(self.model.state_dict())
                self.patience_ctr = 0
                ckpt_path = os.path.join(self.save_dir,
                                         f"{self.experiment}_best.pth")
                torch.save(self.best_weights, ckpt_path)
                print(f"  ✓ New best val_acc={va_acc:.4f}  saved → {ckpt_path}")
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= self.patience:
                    print(f"\n  Early stopping at epoch {epoch}.")
                    break

        # Restore best weights
        if self.best_weights:
            self.model.load_state_dict(self.best_weights)

        # Save history
        hist_path = os.path.join(self.save_dir, f"{self.experiment}_history.json")
        with open(hist_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n  History saved → {hist_path}")
        print(f"  Best val_acc = {self.best_val_acc:.4f}\n")

        return self.history
