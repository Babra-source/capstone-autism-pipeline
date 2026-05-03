"""
Model Architectures
===================

Experiment 1 – FER-only models (face stream):
    1a. ResNet-18
    1b. ResNet-50
    1c. ViT-B/16
    1d. MultiScaleViT  ← dual-resolution face ViT with cross-attention fusion

Experiment 2 – Eye-Tracking models (eye stream, RQ2):
    2a. ResNet18Eye     - ResNet-18 fine-tuned on eye crops
    2b. ResNet50Eye     - ResNet-50 fine-tuned on eye crops
    2c. ViTEye          - ViT-B/16 fine-tuned on eye crops
    2d. MultiViTEye

Experiment 3 – Early Fusion (RQ1):
    3a. EarlyFusionResNet – ResNet-18 face (512) + ResNet-18 eye (512) concatenated → shared head

Experiment 4 – Late Fusion (RQ1):
    4.  LateFusionModel – weighted probability sum of best FER + best eye model
                          fer_weight determined by grid search on validation set

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
    Multi-Scale ViT for FER  (Exp 1d).

    Processes the same face at two scales:
        • Global (224×224) — captures overall expression context.
        • Local  (centre-cropped 112×112 → upsampled to 224×224) — zooms
          in on the perioral / periorbital region.

    The two 768-d [CLS] tokens are fused with a cross-attention layer
    (global queries local), then concatenated (1536-d) → classifier head.
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.vit_global = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit_global.heads.head = nn.Identity()

        self.vit_local  = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit_local.heads.head  = nn.Identity()

        hidden = 768

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=8, dropout=dropout, batch_first=True,
        )
        self.ln_fusion = nn.LayerNorm(hidden)

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

        q  = cls_global.unsqueeze(1)
        kv = cls_local.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        attn_out = self.ln_fusion(attn_out.squeeze(1) + cls_global)

        return torch.cat([attn_out, cls_local], dim=1)   # (B, 1536)

    def forward(self, x: torch.Tensor):
        return self.classifier(self._fused_features(x))

    def get_features(self, x: torch.Tensor):
        return self._fused_features(x)   # (B, 1536)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2  –  Eye-Tracking models  (eye stream, RQ2)
# ═══════════════════════════════════════════════════════════════════════════════


class ResNet18Eye(nn.Module):
    """
    ResNet-18 fine-tuned on eye-region crops.
    Handles video sequences: (B, T, C, H, W) → mean pooling over time.
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
        if x.shape[-2] == 224:
            return x
        return F.interpolate(x, size=(224, 224), mode='bilinear',
                             align_corners=False)

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x); x = b.layer2(x); x = b.layer3(x); x = b.layer4(x)
        return torch.flatten(b.avgpool(x), 1)   # (B, 512)

    def _process_frames(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return self._forward_2d(self._resize(x))
        elif x.dim() == 5:
            B, T, C, H, W = x.shape
            x = self._resize(x.view(B * T, C, H, W))
            return self._forward_2d(x).view(B, T, -1).mean(dim=1)  # (B, 512)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

    def forward(self, x):
        return self.backbone.fc(self._process_frames(x))

    def get_features(self, x):
        return self._process_frames(x)   # (B, 512)


class ResNet50Eye(nn.Module):
    """
    ResNet-50 fine-tuned on eye-region crops.
    Handles video sequences: (B, T, C, H, W) → mean pooling over time.
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
        if x.shape[-2] == 224:
            return x
        return F.interpolate(x, size=(224, 224), mode='bilinear',
                             align_corners=False)

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x); x = b.layer2(x); x = b.layer3(x); x = b.layer4(x)
        return torch.flatten(b.avgpool(x), 1)   # (B, 2048)

    def _process_frames(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return self._forward_2d(self._resize(x))
        elif x.dim() == 5:
            B, T, C, H, W = x.shape
            x = self._resize(x.view(B * T, C, H, W))
            return self._forward_2d(x).view(B, T, -1).mean(dim=1)  # (B, 2048)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

    def forward(self, x):
        return self.backbone.fc(self._process_frames(x))

    def get_features(self, x):
        return self._process_frames(x)   # (B, 2048)


class ViTEye(nn.Module):
    """
    ViT-B/16 fine-tuned on eye-region crops.
    Handles video sequences: (B, T, C, H, W) → mean pooling over time.
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
        if x.shape[-2] == 224:
            return x
        return F.interpolate(x, size=(224, 224), mode='bilinear',
                             align_corners=False)

    def _process_frames(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return _vit_cls_features(self.vit, self._resize(x))
        elif x.dim() == 5:
            B, T, C, H, W = x.shape
            x = self._resize(x.view(B * T, C, H, W))
            return _vit_cls_features(self.vit, x).view(B, T, -1).mean(dim=1)  # (B, 768)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

    def forward(self, x):
        return self.vit.heads.head(self._process_frames(x))

    def get_features(self, x):
        return self._process_frames(x)   # (B, 768)


class MultiViTEye(nn.Module):
    """
    Multi-Scale ViT for eye-region crops (Exp 2e).

    Mirrors MultiScaleViTFER but applied to the eye stream.
    The eye strip is processed at two scales:
        • Global (224×224) — full eye strip, captures both eyes together
          and their relative position/gaze direction.
        • Local  (centre-cropped 50% → upsampled to 224×224) — zooms into
          the inner eye region (pupils, eyelid tension, sclera exposure),
          the most informative zone for micro-expression eye cues.

    The two 768-d [CLS] tokens are fused with cross-attention
    (global queries local), then concatenated (1536-d) → classifier head.

    Handles video sequences: (B, T, C, H, W) → per-frame features →
    mean pooling over time → (B, 1536).

    Feature dim: 1536.
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()

        self.vit_global = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit_global.heads.head = nn.Identity()

        self.vit_local  = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit_local.heads.head  = nn.Identity()

        hidden = 768

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=8, dropout=dropout, batch_first=True,
        )
        self.ln_fusion = nn.LayerNorm(hidden)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _resize(x: torch.Tensor) -> torch.Tensor:
        """Eye strip is 112×224; ViT expects square input → resize."""
        if x.shape[-2] == 224:
            return x
        return F.interpolate(x, size=(224, 224), mode='bilinear',
                             align_corners=False)

    @staticmethod
    def _centre_crop_zoom(x: torch.Tensor) -> torch.Tensor:
        """Centre-crop inner 50% → resize to 224, simulating 2× zoom."""
        _, _, H, W = x.shape
        t, l = H // 4, W // 4
        cropped = x[:, :, t: t + H // 2, l: l + W // 2]
        return F.interpolate(cropped, size=(H, W), mode='bilinear',
                             align_corners=False)

    def _fused_features_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Compute fused features for a batch of 2D frames (B, C, H, W)."""
        x       = self._resize(x)
        x_local = self._centre_crop_zoom(x)

        cls_global = _vit_cls_features(self.vit_global, x)        # (B, 768)
        cls_local  = _vit_cls_features(self.vit_local,  x_local)  # (B, 768)

        q  = cls_global.unsqueeze(1)   # (B, 1, 768)
        kv = cls_local.unsqueeze(1)    # (B, 1, 768)
        attn_out, _ = self.cross_attn(q, kv, kv)
        attn_out = self.ln_fusion(attn_out.squeeze(1) + cls_global)  # residual

        return torch.cat([attn_out, cls_local], dim=1)   # (B, 1536)

    def _process_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Handle both (B, C, H, W) single frames and (B, T, C, H, W) videos."""
        if x.dim() == 4:
            return self._fused_features_2d(x)
        elif x.dim() == 5:
            B, T, C, H, W = x.shape
            # Process all frames independently then mean-pool over time
            feats = self._fused_features_2d(x.view(B * T, C, H, W))  # (B*T, 1536)
            return feats.view(B, T, -1).mean(dim=1)                    # (B, 1536)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

    def forward(self, x: torch.Tensor):
        return self.classifier(self._process_frames(x))

    def get_features(self, x: torch.Tensor):
        return self._process_frames(x)   # (B, 1536)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3  –  Early Fusion  (RQ1)
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyFusionResNet(nn.Module):
    """
    Early Fusion: ResNet-18 face (512) + ResNet-18 eye (512) → concat (1024) → head.

    Both streams use the same ResNet-18 backbone pretrained on ImageNet,
    ensuring any accuracy difference between face and eye is due to the
    modality (input content) rather than model capacity — consistent with
    the controlled comparison design for RQ2.

    The face stream receives the full 224×224 face image.
    The eye stream receives the MediaPipe-cropped eye region (resized to 224×224).
    Both backbones are fine-tuned end-to-end with differential LR.
    LayerNorm is applied to each stream before concatenation to align
    feature scales before the shared classification head.

    Feature dims: face 512 + eye 512 → concat 1024 → 512 → num_classes.
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.4):
        super().__init__()

        # Face stream — ResNet-18 backbone, fc replaced with Identity
        face_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        face_feat_dim = face_backbone.fc.in_features   # 512
        face_backbone.fc = nn.Identity()
        self.face_resnet = face_backbone

        # Eye stream — ResNet-18 backbone, fc replaced with Identity
        eye_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        eye_feat_dim = eye_backbone.fc.in_features     # 512
        eye_backbone.fc = nn.Identity()
        self.eye_resnet = eye_backbone

        fused_dim = face_feat_dim + eye_feat_dim       # 1024

        # Shared classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _resize(x: torch.Tensor) -> torch.Tensor:
        """Eye strip may be 112×224 — ResNet-18 requires square input."""
        if x.shape[-2] == 224 and x.shape[-1] == 224:
            return x
        return F.interpolate(x, size=(224, 224), mode='bilinear',
                             align_corners=False)

    def _eye_features(self, eye: torch.Tensor) -> torch.Tensor:
        """Handle both (B, C, H, W) and (B, T, C, H, W) eye inputs."""
        if eye.dim() == 4:
            return self.eye_resnet(self._resize(eye))          # (B, 512)
        elif eye.dim() == 5:
            B, T, C, H, W = eye.shape
            x = self._resize(eye.view(B * T, C, H, W))
            feats = self.eye_resnet(x)                         # (B*T, 512)
            return feats.view(B, T, -1).mean(dim=1)            # (B, 512)
        else:
            raise ValueError(f"Expected 4D or 5D eye input, got {eye.dim()}D")

    def forward(self, face: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
        face_feat = self.face_resnet(face)       # (B, 512)
        eye_feat  = self._eye_features(eye)      # (B, 512)
        fused     = torch.cat([face_feat, eye_feat], dim=1)   # (B, 1024)
        return self.classifier(fused)

    def get_features(self, face: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
        """Return the 1024-d fused representation before the classifier head."""
        face_feat = self.face_resnet(face)
        eye_feat  = self._eye_features(eye)
        return torch.cat([face_feat, eye_feat], dim=1)         # (B, 1024)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4  –  Late Fusion  (RQ1)
# ═══════════════════════════════════════════════════════════════════════════════

class LateFusionModel(nn.Module):
    """
    Late Fusion: takes the best pretrained FER model and the best pretrained
    eye model (both frozen), and combines their softmax outputs:

        P = (fer_weight × P_face) + (eye_weight × P_eye)

    IMPORTANT: both face_model and eye_model must be loaded from their best
    saved checkpoints BEFORE passing them here. This class does NOT train
    either model — it only combines their predictions.

    fer_weight is determined by grid search on the validation set via
    find_best_fer_weight(). The default 0.5 is a neutral starting point
    meaning equal trust — the real value must come from grid search.

    Args:
        face_model:        best trained FER model (e.g. ViTFER) — frozen
        eye_model:         best trained eye model (e.g. ResNet18Eye) — frozen
        num_classes:       number of output classes
        fer_weight:        weight for face model (0.0–1.0); eye gets 1-fer_weight
                           set via find_best_fer_weight(), not manually
        learnable_weights: if True, learns the weight via backprop instead
                           of grid search (useful for ablation)
    """

    def __init__(self,
                 face_model: nn.Module,
                 eye_model:  nn.Module,
                 num_classes: int   = 5,
                 fer_weight:  float = 0.5,   # neutral default — set by grid search
                 learnable_weights: bool = False):
        super().__init__()

        # Freeze both models — late fusion never retrains them
        self.face_model = face_model
        self.eye_model  = eye_model
        for p in self.face_model.parameters():
            p.requires_grad = False
        for p in self.eye_model.parameters():
            p.requires_grad = False

        self.num_classes = num_classes

        if learnable_weights:
            # sigmoid(0.0) = 0.5 → starts at equal weighting
            self.w_logit   = nn.Parameter(torch.tensor([0.0]))
            self.learnable = True
        else:
            self.register_buffer('_fer_weight',
                                 torch.tensor([fer_weight], dtype=torch.float32))
            self.learnable = False

    def get_weights(self):
        if self.learnable:
            w = torch.sigmoid(self.w_logit)
            return w, 1.0 - w
        return self._fer_weight, 1.0 - self._fer_weight

    def set_fer_weight(self, w: float):
        """Update fer_weight after grid search — call before test evaluation."""
        if self.learnable:
            raise ValueError("Model uses learnable weights — cannot set manually.")
        self._fer_weight.fill_(w)

    def forward(self, face, eye):
        self.face_model.eval()
        self.eye_model.eval()
        fer_p = F.softmax(self.face_model(face), dim=1)
        eye_p = F.softmax(self.eye_model(eye),   dim=1)
        fw, ew = self.get_weights()
        fw = fw.to(face.device)
        ew = ew.to(face.device)
        return torch.log(fw * fer_p + ew * eye_p + 1e-8)   # NLLLoss-compatible

    def predict_proba(self, face, eye):
        self.face_model.eval()
        self.eye_model.eval()
        fer_p = F.softmax(self.face_model(face), dim=1)
        eye_p = F.softmax(self.eye_model(eye),   dim=1)
        fw, ew = self.get_weights()
        fw = fw.to(face.device)
        ew = ew.to(face.device)
        return fw * fer_p + ew * eye_p


def find_best_fer_weight(face_model:  nn.Module,
                          eye_model:   nn.Module,
                          val_loader,
                          device:      str,
                          num_classes: int = 5) -> float:
    """
    Grid search over fer_weight on the validation set to find the optimal
    weighting between face and eye model predictions.

    Call this AFTER Exp1 and Exp2 are complete and best checkpoints loaded.
    Pass the result to LateFusionModel.set_fer_weight() before test eval.

    Returns:
        best_weight: float in [0.1, 0.9] that maximises val accuracy
    """
    best_weight, best_acc = 0.5, 0.0

    for w in [round(x * 0.1, 1) for x in range(1, 10)]:  # 0.1 to 0.9
        model = LateFusionModel(face_model, eye_model,
                                num_classes=num_classes,
                                fer_weight=w)
        model.eval().to(device)

        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                face  = batch['face'].to(device)
                eye   = batch['eye'].to(device)
                label = batch['label'].to(device)
                preds = model(face, eye).argmax(dim=1)
                correct += (preds == label).sum().item()
                total   += label.size(0)

        acc = correct / total
        print(f"  fer_weight={w:.1f}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc    = acc
            best_weight = w

    print(f"\n  Best fer_weight={best_weight:.1f}  val_acc={best_acc:.4f}")
    return best_weight


def build_late_fusion(best_fer_ckpt:  str,
                       best_eye_ckpt:  str,
                       face_model:     nn.Module,
                       eye_model:      nn.Module,
                       val_loader,
                       device:         str,
                       num_classes:    int = 5) -> LateFusionModel:
    """
    Load best checkpoints into face and eye models, run grid search,
    and return a ready-to-evaluate LateFusionModel.

    Usage in run_all.py:
        fusion = build_late_fusion(
            best_fer_ckpt = 'results/best_fer_model.pth',
            best_eye_ckpt = 'results/best_eye_model.pth',
            face_model    = ViTFER(num_classes),
            eye_model     = ResNet18Eye(num_classes),
            val_loader    = loaders['val'],
            device        = device,
        )
    """
    # Load trained weights — not random init
    face_model.load_state_dict(torch.load(best_fer_ckpt, map_location=device))
    eye_model.load_state_dict(torch.load(best_eye_ckpt,  map_location=device))
    face_model.eval()
    eye_model.eval()

    # Find optimal weight from validation set
    best_w = find_best_fer_weight(face_model, eye_model,
                                   val_loader, device, num_classes)

    # Build final model with optimal weight
    fusion = LateFusionModel(face_model, eye_model,
                              num_classes=num_classes,
                              fer_weight=best_w)
    return fusion


# ═══════════════════════════════════════════════════════════════════════════════
# RQ3 SUPPORT  –  Gaze feature extractor
# ═══════════════════════════════════════════════════════════════════════════════

class GazeFeatureExtractor(nn.Module):
    """
    Thin wrapper that calls get_features() on any eye model.
    Used by experiments/rq3_gaze_correlation.py to build the (N, D) feature
    matrix for Pearson/Spearman correlation, t-SNE, and per-class gaze
    activation heatmaps.
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
        'multiscale_vit'  MultiScaleViTFER

    Eye stream:
        'resnet18_eye'    ResNet18Eye
        'resnet50_eye'    ResNet50Eye
        'vit_eye'         ViTEye
        'multivit_eye'    MultiViTEye

    Fusion:
        'early_fusion'    EarlyFusionResNet
        'late_fusion'     use build_late_fusion() instead — requires trained
                          checkpoints and val_loader for grid search
    """
    n = name.lower()
    if n == 'resnet18':        return ResNet18FER(num_classes, **kwargs)
    if n == 'resnet50':        return ResNet50FER(num_classes, **kwargs)
    if n == 'vit':             return ViTFER(num_classes, **kwargs)
    if n == 'multiscale_vit':  return MultiScaleViTFER(num_classes, **kwargs)
    if n == 'resnet18_eye':    return ResNet18Eye(num_classes, **kwargs)
    if n == 'resnet50_eye':    return ResNet50Eye(num_classes, **kwargs)
    if n == 'vit_eye':         return ViTEye(num_classes, **kwargs)
    if n == 'multivit_eye':    return MultiViTEye(num_classes, **kwargs)
    if n == 'early_fusion':    return EarlyFusionResNet(num_classes, **kwargs)
    if n == 'late_fusion':
        raise ValueError(
            "'late_fusion' requires trained checkpoints. "
            "Use build_late_fusion(best_fer_ckpt, best_eye_ckpt, "
            "face_model, eye_model, val_loader, device) instead."
        )
    raise ValueError(f"Unknown model: '{name}'. See build_model docstring.")