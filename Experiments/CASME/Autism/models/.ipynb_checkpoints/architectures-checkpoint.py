"""
Model Architectures
===================

Experiment 1 – FER-only models (face stream):
    1a. ResNet-18
    1b. ResNet-50
    1c. ViT-B/16
    1d. MultiScaleViT  ← NEW: dual-resolution face ViT with cross-attention fusion

Experiment 2 – Eye-Tracking models (eye stream, RQ2):
    2a. EyeNet          – lightweight custom CNN baseline
    2b. ResNet18Eye     ← NEW: ResNet-18 fine-tuned on eye crops
    2c. ResNet50Eye     ← NEW: ResNet-50 fine-tuned on eye crops
    2d. ViTEye          ← NEW: ViT-B/16 fine-tuned on eye crops

Experiment 3 – Early Fusion (RQ1):
    3a. EarlyFusionViT  – ViT face + EyeNet eye concatenated

Experiment 4 – Late Fusion (RQ1):
    4.  LateFusionModel – weighted probability sum, any face+eye pair

RQ3 support:
    GazeFeatureExtractor – thin wrapper to pull raw gaze feature vectors
                           for correlation & t-SNE analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    vit_b_16, ViT_B_16_Weights,
)


# ───────────────────────────────────────────────────────────────────────────────
# Shared helper
# ───────────────────────────────────────────────────────────────────────────────

def _vit_cls_features(vit_model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract the [CLS] token embedding from any ViT whose head may be Identity."""
    x   = vit_model._process_input(x)
    n   = x.shape[0]
    cls = vit_model.class_token.expand(n, -1, -1)
    x   = torch.cat([cls, x], dim=1)
    x   = vit_model.encoder(x)
    return x[:, 0]   # (B, 768)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1  –  FER-only models  (face stream)
# ═══════════════════════════════════════════════════════════════════════════════

class ResNet18FER(nn.Module):
    """ResNet-18 fine-tuned for FER. Feature dim: 512."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.4):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone.fc.in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def get_features(self, x):
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x); x = b.layer2(x); x = b.layer3(x); x = b.layer4(x)
        return torch.flatten(b.avgpool(x), 1)   # (B, 512)


class ResNet50FER(nn.Module):
    """ResNet-50 fine-tuned for FER. Feature dim: 2048."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.4):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone.fc.in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def get_features(self, x):
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x); x = b.layer2(x); x = b.layer3(x); x = b.layer4(x)
        return torch.flatten(b.avgpool(x), 1)   # (B, 2048)


class ViTFER(nn.Module):
    """ViT-B/16 fine-tuned for FER. Feature dim: 768."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.vit.heads.head.in_features, num_classes),
        )

    def forward(self, x):
        return self.vit(x)

    def get_features(self, x):
        return _vit_cls_features(self.vit, x)   # (B, 768)


class MultiScaleViTFER(nn.Module):
    """
    Multi-Scale ViT for FER  (Exp 1d — NEW).

    Motivation
    ----------
    Autistic micro-expressions are often spatially subtle — a slight brow
    raise or lip corner tension — rather than full-face expressions. A
    single-scale ViT may miss these because the 16×16 patches average over
    large regions.  This model processes the same face at two scales:

        • Global (224×224) — captures overall expression context.
        • Local  (centre-cropped 112×112 → upsampled to 224×224) — zooms
          in on the perioral / periorbital region, the most informative
          zone for micro-expression recognition.

    The two 768-d [CLS] tokens are fused with a cross-attention layer
    (global queries local), then concatenated (1536-d) → classifier head.

    Architecture
    ------------
        x ──→ ViT-global ──→ CLS_g (768)─────────────────────────────┐
        x ──→ centre-crop + resize ──→ ViT-local ──→ CLS_l (768)─→ CrossAttn ──→ [CLS_fused ‖ CLS_l] (1536) ──→ head
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()

        # Two ViT encoders — separate weights allow each to specialise
        self.vit_global = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit_global.heads.head = nn.Identity()

        self.vit_local  = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit_local.heads.head  = nn.Identity()

        hidden = 768

        # Cross-attention: global queries local context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=8, dropout=dropout, batch_first=True,
        )
        self.ln_fusion = nn.LayerNorm(hidden)

        # Final classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _centre_crop_zoom(x: torch.Tensor) -> torch.Tensor:
        """Centre-crop inner 50 % → resize to 224, simulating 2× zoom."""
        _, _, H, W = x.shape
        t, l = H // 4, W // 4
        cropped = x[:, :, t: t + H // 2, l: l + W // 2]
        return F.interpolate(cropped, size=(H, W), mode='bilinear',
                             align_corners=False)

    def _fused_features(self, x: torch.Tensor):
        x_local    = self._centre_crop_zoom(x)
        cls_global = _vit_cls_features(self.vit_global, x)       # (B, 768)
        cls_local  = _vit_cls_features(self.vit_local,  x_local) # (B, 768)

        # Cross-attention: global attends to local
        q  = cls_global.unsqueeze(1)   # (B, 1, 768)
        kv = cls_local.unsqueeze(1)    # (B, 1, 768)
        attn_out, _ = self.cross_attn(q, kv, kv)
        attn_out = self.ln_fusion(attn_out.squeeze(1) + cls_global)  # residual

        return torch.cat([attn_out, cls_local], dim=1)   # (B, 1536)

    def forward(self, x: torch.Tensor):
        return self.classifier(self._fused_features(x))

    def get_features(self, x: torch.Tensor):
        return self._fused_features(x)   # (B, 1536)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2  –  Eye-Tracking models  (eye stream, RQ2)
# ═══════════════════════════════════════════════════════════════════════════════

class EyeNet(nn.Module):
    """
    Lightweight custom CNN for eye-region feature extraction (baseline).
    Now handles video sequences: input (B, T, C, H, W) → processes all frames
    and aggregates temporal information via mean pooling.
    Features: processes each frame through CNN, then averages across time.
    Feature dim: 256.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.embed = nn.Sequential(
            nn.Linear(128 * 4 * 8, 512), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.classifier = nn.Linear(256, num_classes)

    def _process_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Process either (B, C, H, W) or (B, T, C, H, W)."""
        if x.dim() == 4:
            # Single frame (B, C, H, W) - backward compatible
            x = self.features(x)
            return self.embed(x.view(x.size(0), -1))
        elif x.dim() == 5:
            # Video sequence (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = self.features(x)
            x = self.embed(x.view(B * T, -1))
            x = x.view(B, T, -1)
            # Temporal aggregation: mean across time dimension
            x = x.mean(dim=1)  # (B, 256)
            return x
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

    def forward(self, x):
        return self.classifier(self._process_frames(x))

    def get_features(self, x):
        return self._process_frames(x)   # (B, 256)


class ResNet18Eye(nn.Module):
    """
    ResNet-18 fine-tuned on eye-region crops (RQ2 — NEW).
    Now handles video sequences: (B, T, C, H, W) → processes all frames
    independently and aggregates temporal features via mean pooling.
    
    Using the same backbone as ResNet18FER means any accuracy difference
    between face-only and eye-only experiments is due to the modality
    (what the input encodes) rather than model capacity. This is the
    correct controlled comparison for RQ2.
    Feature dim: 512.
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.4):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone.fc.in_features, num_classes),
        )
        self.backbone = backbone

    @staticmethod
    def _resize(x: torch.Tensor) -> torch.Tensor:
        """Eye strip is 112×224; ResNet expects square input → resize."""
        if x.shape[-2] == 224:
            return x
        return F.interpolate(x, size=(224, 224), mode='bilinear',
                             align_corners=False)

    def _process_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Handle both (B, C, H, W) single frames and (B, T, C, H, W) videos."""
        if x.dim() == 4:
            # Single frame (B, C, H, W)
            return self._forward_2d(self._resize(x))
        elif x.dim() == 5:
            # Video sequence (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = self._resize(x)
            features = self._forward_2d(x)  # (B*T, 512)
            features = features.view(B, T, -1)
            # Temporal aggregation: mean across time
            return features.mean(dim=1)  # (B, 512)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of 2D images."""
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x); x = b.layer2(x); x = b.layer3(x); x = b.layer4(x)
        return torch.flatten(b.avgpool(x), 1)   # (B, 512)

    def forward(self, x):
        return self.backbone.fc(self._process_frames(x))

    def get_features(self, x):
        return self._process_frames(x)   # (B, 512)


class ResNet50Eye(nn.Module):
    """
    ResNet-50 fine-tuned on eye-region crops (RQ2 — NEW).
    Now handles video sequences: (B, T, C, H, W) → processes all frames
    independently and aggregates via mean pooling.
    Feature dim: 2048.
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.4):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone.fc.in_features, num_classes),
        )
        self.backbone = backbone

    @staticmethod
    def _resize(x: torch.Tensor) -> torch.Tensor:
        """Eye strip is 112×224; ResNet expects square input → resize."""
        if x.shape[-2] == 224:
            return x
        return F.interpolate(x, size=(224, 224), mode='bilinear',
                             align_corners=False)

    def _process_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Handle both (B, C, H, W) single frames and (B, T, C, H, W) videos."""
        if x.dim() == 4:
            # Single frame (B, C, H, W)
            return self._forward_2d(self._resize(x))
        elif x.dim() == 5:
            # Video sequence (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = self._resize(x)
            features = self._forward_2d(x)  # (B*T, 2048)
            features = features.view(B, T, -1)
            # Temporal aggregation: mean across time
            return features.mean(dim=1)  # (B, 2048)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of 2D images."""
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x); x = b.layer2(x); x = b.layer3(x); x = b.layer4(x)
        return torch.flatten(b.avgpool(x), 1)   # (B, 2048)

    def forward(self, x):
        return self.backbone.fc(self._process_frames(x))

    def get_features(self, x):
        return self._process_frames(x)   # (B, 2048)


class ViTEye(nn.Module):
    """
    ViT-B/16 fine-tuned on eye-region crops (RQ2 — NEW).
    Now handles video sequences: (B, T, C, H, W) → processes all frames
    independently and aggregates via mean pooling at feature level.
    Captures long-range gaze-direction cues via self-attention across patches.
    Feature dim: 768.
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.vit.heads.head.in_features, num_classes),
        )

    @staticmethod
    def _resize(x: torch.Tensor) -> torch.Tensor:
        """Eye strip is 112×224; ViT expects square input → resize."""
        if x.shape[-2] == 224:
            return x
        return F.interpolate(x, size=(224, 224), mode='bilinear',
                             align_corners=False)

    def _process_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Handle both (B, C, H, W) single frames and (B, T, C, H, W) videos."""
        if x.dim() == 4:
            # Single frame (B, C, H, W)
            return _vit_cls_features(self.vit, self._resize(x))
        elif x.dim() == 5:
            # Video sequence (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = self._resize(x)
            features = _vit_cls_features(self.vit, x)  # (B*T, 768)
            features = features.view(B, T, -1)
            # Temporal aggregation: mean across time
            return features.mean(dim=1)  # (B, 768)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

    def forward(self, x):
        return self.vit.heads.head(self._process_frames(x))

    def get_features(self, x):
        return self._process_frames(x)   # (B, 768)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3  –  Early Fusion  (RQ1)
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyFusionViT(nn.Module):
    """
    Early Fusion: ViT face (768) + EyeNet eye (256) → concat (1024) → head.
    Both streams jointly fine-tuned end-to-end.
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.face_vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.face_vit.heads.head = nn.Identity()
        self.eye_net  = EyeNet(num_classes=num_classes)

        self.classifier = nn.Sequential(
            nn.LayerNorm(768 + 256),
            nn.Dropout(dropout),
            nn.Linear(768 + 256, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, face, eye):
        face_feat = _vit_cls_features(self.face_vit, face)   # (B, 768)
        eye_feat  = self.eye_net.get_features(eye)            # (B, 256)
        return self.classifier(torch.cat([face_feat, eye_feat], dim=1))


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4  –  Late Fusion  (RQ1)
# ═══════════════════════════════════════════════════════════════════════════════

class LateFusionModel(nn.Module):
    """
    Late Fusion: independently trained face and eye models whose softmax
    outputs are combined:
        P = (fer_weight × P_face) + (eye_weight × P_eye)
    Optimal fer_weight is determined by grid search on the validation set.
    Accepts any (face_model, eye_model) pair — enabling cross-model fusion
    comparisons (e.g. ResNet50FER + ViTEye) as required for RQ2.
    """

    def __init__(self,
                 face_model: nn.Module,
                 eye_model:  nn.Module,
                 num_classes: int   = 5,
                 fer_weight:  float = 0.7,
                 learnable_weights: bool = False):
        super().__init__()
        self.face_model  = face_model
        self.eye_model   = eye_model
        self.num_classes = num_classes

        if learnable_weights:
            self.w_logit   = nn.Parameter(torch.tensor([1.0]))
            self.learnable = True
        else:
            self.fer_weight = fer_weight
            self.learnable  = False

    def get_weights(self):
        if self.learnable:
            w = torch.sigmoid(self.w_logit)
            return w, 1.0 - w
        return torch.tensor(self.fer_weight), torch.tensor(1.0 - self.fer_weight)

    def forward(self, face, eye):
        fer_p = F.softmax(self.face_model(face), dim=1)
        eye_p = F.softmax(self.eye_model(eye),   dim=1)
        fw, ew = self.get_weights()
        fw, ew = fw.to(face.device), ew.to(face.device)
        return torch.log(fw * fer_p + ew * eye_p + 1e-8)   # NLLLoss-compatible

    def predict_proba(self, face, eye):
        fer_p = F.softmax(self.face_model(face), dim=1)
        eye_p = F.softmax(self.eye_model(eye),   dim=1)
        fw, ew = self.get_weights()
        fw, ew = fw.to(face.device), ew.to(face.device)
        return fw * fer_p + ew * eye_p


# ═══════════════════════════════════════════════════════════════════════════════
# RQ3 SUPPORT  –  Gaze feature extractor
# ═══════════════════════════════════════════════════════════════════════════════

class GazeFeatureExtractor(nn.Module):
    """
    Thin wrapper that calls get_features() on any eye model.
    Used by experiments/rq3_gaze_correlation.py to build the (N, D) feature
    matrix for Pearson/Spearman correlation, t-SNE, and per-class gaze
    activation heatmaps — all required outputs for RQ3.
    """

    def __init__(self, eye_model: nn.Module):
        super().__init__()
        self.eye_model = eye_model

    @torch.no_grad()
    def forward(self, eye: torch.Tensor) -> torch.Tensor:
        self.eye_model.eval()
        return self.eye_model.get_features(eye)


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(name: str, num_classes: int = 5, **kwargs) -> nn.Module:
    """
    Instantiate a model by name.

    FER (face stream):
        'resnet18'        ResNet18FER
        'resnet50'        ResNet50FER
        'vit'             ViTFER
        'multiscale_vit'  MultiScaleViTFER          ← NEW

    Eye stream:
        'eyenet'          EyeNet
        'resnet18_eye'    ResNet18Eye               ← NEW
        'resnet50_eye'    ResNet50Eye               ← NEW
        'vit_eye'         ViTEye                    ← NEW

    Fusion:
        'early_fusion'    EarlyFusionViT
        'late_fusion'     LateFusionModel (ViT + EyeNet default pair)
    """
    n = name.lower()
    if n == 'resnet18':        return ResNet18FER(num_classes, **kwargs)
    if n == 'resnet50':        return ResNet50FER(num_classes, **kwargs)
    if n == 'vit':             return ViTFER(num_classes, **kwargs)
    if n == 'multiscale_vit':  return MultiScaleViTFER(num_classes, **kwargs)
    if n == 'eyenet':          return EyeNet(num_classes)
    if n == 'resnet18_eye':    return ResNet18Eye(num_classes, **kwargs)
    if n == 'resnet50_eye':    return ResNet50Eye(num_classes, **kwargs)
    if n == 'vit_eye':         return ViTEye(num_classes, **kwargs)
    if n == 'early_fusion':    return EarlyFusionViT(num_classes, **kwargs)
    if n == 'late_fusion':
        return LateFusionModel(
            ViTFER(num_classes), EyeNet(num_classes), num_classes, **kwargs)
    raise ValueError(f"Unknown model: '{name}'. See build_model docstring.")
