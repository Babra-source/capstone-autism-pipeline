from io import BytesIO
from pathlib import Path
from typing import List, Optional
import os
import urllib.request

import numpy as np
import joblib
import timm
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

BACKEND_DIR = Path(__file__).parent
FER_MODEL_PATH = BACKEND_DIR / "multiscale_vit_best.pth"
EYE_MODEL_PATH = BACKEND_DIR / "eye_best.pkl"
LANDMARKER_PATH = BACKEND_DIR / "face_landmarker.task"
EMOTIONS = ["anger", "fear", "joy", "Natural", "sadness", "surprise"]

FER_WEIGHT = float(os.getenv("FER_WEIGHT", 0.6))
EYE_WEIGHT = float(os.getenv("EYE_WEIGHT", 0.4))
TOTAL_WEIGHT = FER_WEIGHT + EYE_WEIGHT if (FER_WEIGHT + EYE_WEIGHT) > 0 else 1.0
FER_WEIGHT /= TOTAL_WEIGHT
EYE_WEIGHT /= TOTAL_WEIGHT

app = FastAPI(title="Autism FER Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fer_model: Optional[torch.nn.Module] = None
eye_model = None
eye_scaler = None
mp_landmarker = None


class EmotionProb(BaseModel):
    label: str
    value: float


class PredictionResponse(BaseModel):
    top_emotion: str
    confidence: float
    emotions: List[EmotionProb]


class MultiScaleViT(nn.Module):
    def __init__(self, num_classes=6, in_chans=6, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            img_size=224,
        )
        embed_dim = self.backbone.embed_dim
        self.down2 = nn.AdaptiveAvgPool2d((112, 112))
        self.down4 = nn.AdaptiveAvgPool2d((56, 56))
        self.up_from_112 = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)
        self.up_from_56 = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)
        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward_one_scale(self, x):
        return self.backbone(x)

    def forward(self, x):
        feat_224 = self.forward_one_scale(x)
        feat_112 = self.forward_one_scale(self.up_from_112(self.down2(x)))
        feat_56 = self.forward_one_scale(self.up_from_56(self.down4(x)))
        return self.fusion(torch.cat([feat_224, feat_112, feat_56], dim=1))


def build_fer_model() -> torch.nn.Module:
    model = MultiScaleViT(num_classes=len(EMOTIONS), in_chans=6, pretrained=False)
    return model


def download_landmarker_model() -> None:
    if LANDMARKER_PATH.exists():
        return
    LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )
    urllib.request.urlretrieve(url, str(LANDMARKER_PATH))


def load_face_landmarker() -> Optional[face_landmarker.FaceLandmarker]:
    download_landmarker_model()
    base_options = python.BaseOptions(model_asset_path=str(LANDMARKER_PATH))
    options = face_landmarker.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionTaskRunningMode.IMAGE,
    )
    return face_landmarker.FaceLandmarker.create_from_options(options)


def load_models() -> None:
    global fer_model, eye_model, eye_scaler, mp_landmarker

    if FER_MODEL_PATH.exists():
        fer_model = build_fer_model()
        fer_model.load_state_dict(torch.load(FER_MODEL_PATH, map_location="cpu"))
        fer_model.eval()
    else:
        print(f"Missing FER model: {FER_MODEL_PATH}")

    if EYE_MODEL_PATH.exists():
        eye_model = joblib.load(str(EYE_MODEL_PATH))
    else:
        print(f"Missing Eye Tracking model: {EYE_MODEL_PATH}")

    mp_landmarker = load_face_landmarker()
    if mp_landmarker is None:
        print("Failed to initialize MediaPipe face landmarker.")


LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE_VERTICAL_1 = [159, 145]
LEFT_EYE_VERTICAL_2 = [158, 153]
LEFT_EYE_HORIZONTAL = [33, 133]
RIGHT_EYE_VERTICAL_1 = [386, 374]
RIGHT_EYE_VERTICAL_2 = [385, 380]
RIGHT_EYE_HORIZONTAL = [362, 263]
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
GAZE_DIRS = ["centre", "left", "right", "up", "down"]

EYE_FEATURES = [
    "left_ear",
    "right_ear",
    "avg_ear",
    "eye_openness_pct",
    "left_relative_pupil_x",
    "left_relative_pupil_y",
    "right_relative_pupil_x",
    "right_relative_pupil_y",
    "left_eye_width",
    "left_eye_height",
    "right_eye_width",
    "right_eye_height",
    "gaze_centre",
    "gaze_left",
    "gaze_right",
    "gaze_up",
    "gaze_down",
    "ear_asymmetry",
    "ear_asymmetry_abs",
    "ear_ratio_lr",
    "left_eye_aspect",
    "right_eye_aspect",
    "eye_aspect_asymmetry",
    "eye_width_asymmetry",
    "eye_height_asymmetry",
    "avg_gaze_x",
    "avg_gaze_y",
    "gaze_magnitude",
    "pupil_x_divergence",
    "pupil_y_divergence",
    "ear_x_gaze_mag",
    "ear_x_gaze_y",
    "is_wide_open",
    "is_squinting",
]


def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_ear(v1, v2, h):
    return (calculate_distance(*v1) + calculate_distance(*v2)) / (2.0 * calculate_distance(*h) + 1e-6)


def get_eye_metrics(landmarks, image_width, image_height, eye_indices, v1_idx, v2_idx, h_idx, iris_indices=None):
    pts = np.array([
        (int(landmarks[i].x * image_width), int(landmarks[i].y * image_height))
        for i in eye_indices
    ])
    eye_left, eye_right = pts[:, 0].min(), pts[:, 0].max()
    eye_top, eye_bottom = pts[:, 1].min(), pts[:, 1].max()
    eye_width = eye_right - eye_left
    eye_height = eye_bottom - eye_top
    eye_center_x = (eye_left + eye_right) / 2
    eye_center_y = (eye_top + eye_bottom) / 2

    def lm(index):
        return (landmarks[index].x * image_width, landmarks[index].y * image_height)

    ear = calculate_ear([lm(v1_idx[0]), lm(v1_idx[1])], [lm(v2_idx[0]), lm(v2_idx[1])], [lm(h_idx[0]), lm(h_idx[1])])
    pupil_x, pupil_y = eye_center_x, eye_center_y
    if iris_indices and len(landmarks) > max(iris_indices):
        pupil_x = landmarks[iris_indices[0]].x * image_width
        pupil_y = landmarks[iris_indices[0]].y * image_height

    return {
        "pupil_x": pupil_x,
        "pupil_y": pupil_y,
        "relative_pupil_x": (pupil_x - eye_center_x) / (eye_width / 2 + 1e-6),
        "relative_pupil_y": (pupil_y - eye_center_y) / (eye_height / 2 + 1e-6),
        "eye_center_x": eye_center_x,
        "eye_center_y": eye_center_y,
        "eye_width": eye_width,
        "eye_height": eye_height,
        "eye_aspect_ratio": ear,
    }


def derive_gaze(left_m, right_m):
    avg_x = (left_m["relative_pupil_x"] + right_m["relative_pupil_x"]) / 2
    avg_y = (left_m["relative_pupil_y"] + right_m["relative_pupil_y"]) / 2
    if abs(avg_x) < 0.15 and abs(avg_y) < 0.15:
        return "centre"
    if abs(avg_x) >= abs(avg_y):
        return "right" if avg_x > 0 else "left"
    return "down" if avg_y > 0 else "up"


class FacialRegionOcclusion:
    REGION_INDICES = {
        "left_eye": [33, 160, 158, 133, 153, 144],
        "right_eye": [362, 385, 387, 263, 373, 380],
        "nose": [1, 2, 98, 327, 168, 195],
    }

    def __init__(self, regions=None, occlusion_value=0):
        self.regions = regions or ["left_eye", "right_eye", "nose"]
        self.occlusion_value = occlusion_value
        base_options = python.BaseOptions(model_asset_path=str(LANDMARKER_PATH))
        options = face_landmarker.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionTaskRunningMode.IMAGE,
        )
        self.landmarker = face_landmarker.FaceLandmarker.create_from_options(options)

    def get_landmarks(self, image_np):
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        results = self.landmarker.detect(mp_image)
        if not results.face_landmarks:
            return None
        lms = results.face_landmarks[0]
        return [(int(lm.x * image_np.shape[1]), int(lm.y * image_np.shape[0])) for lm in lms]

    def occlude_region(self, image_np, landmarks, region_name):
        indices = self.REGION_INDICES[region_name]
        points = np.array([landmarks[i] for i in indices], dtype=np.int32)
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))
        h, w = image_np.shape[:2]
        box_h, box_w = int(h * 0.25), int(w * 0.25)
        top = max(0, center_y - box_h // 2)
        left = max(0, center_x - box_w // 2)
        bottom = min(h, top + box_h)
        right = min(w, left + box_w)
        image_np[top:bottom, left:right] = int(self.occlusion_value * 255)
        return image_np

    def __call__(self, pil_image: Image.Image) -> Image.Image:
        image_np = np.array(pil_image)
        landmarks = self.get_landmarks(image_np)
        if landmarks is None:
            return pil_image
        occluded = image_np.copy()
        for region in self.regions:
            occluded = self.occlude_region(occluded, landmarks, region)
        return Image.fromarray(occluded)


occluder = FacialRegionOcclusion()


def fer_transform_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = image.resize((224, 224))
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


def build_fer_input(image: Image.Image) -> torch.Tensor:
    original = fer_transform_tensor(image)
    occluded_image = occluder(image)
    occluded = fer_transform_tensor(occluded_image)
    return torch.cat([original, occluded], dim=0).unsqueeze(0)


def extract_eye_features(image: Image.Image) -> Optional[np.ndarray]:
    if mp_landmarker is None:
        return None
    rgb = np.array(image.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = mp_landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    left_metrics = get_eye_metrics(
        landmarks,
        rgb.shape[1],
        rgb.shape[0],
        LEFT_EYE_INDICES,
        LEFT_EYE_VERTICAL_1,
        LEFT_EYE_VERTICAL_2,
        LEFT_EYE_HORIZONTAL,
        LEFT_IRIS_INDICES,
    )
    right_metrics = get_eye_metrics(
        landmarks,
        rgb.shape[1],
        rgb.shape[0],
        RIGHT_EYE_INDICES,
        RIGHT_EYE_VERTICAL_1,
        RIGHT_EYE_VERTICAL_2,
        RIGHT_EYE_HORIZONTAL,
        RIGHT_IRIS_INDICES,
    )

    avg_ear = (left_metrics["eye_aspect_ratio"] + right_metrics["eye_aspect_ratio"]) / 2.0
    gaze_dir = derive_gaze(left_metrics, right_metrics)
    row = {
        "left_ear": left_metrics["eye_aspect_ratio"],
        "right_ear": right_metrics["eye_aspect_ratio"],
        "avg_ear": avg_ear,
        "eye_openness_pct": min(100.0, (avg_ear / 0.30) * 100.0),
        "left_relative_pupil_x": left_metrics["relative_pupil_x"],
        "left_relative_pupil_y": left_metrics["relative_pupil_y"],
        "right_relative_pupil_x": right_metrics["relative_pupil_x"],
        "right_relative_pupil_y": right_metrics["relative_pupil_y"],
        "left_eye_width": left_metrics["eye_width"],
        "left_eye_height": left_metrics["eye_height"],
        "right_eye_width": right_metrics["eye_width"],
        "right_eye_height": right_metrics["eye_height"],
    }
    for d in GAZE_DIRS:
        row[f"gaze_{d}"] = 1.0 if gaze_dir == d else 0.0
    row["ear_asymmetry"] = row["left_ear"] - row["right_ear"]
    row["ear_asymmetry_abs"] = abs(row["ear_asymmetry"])
    row["ear_ratio_lr"] = row["left_ear"] / (row["right_ear"] + 1e-6)
    row["left_eye_aspect"] = row["left_eye_height"] / (row["left_eye_width"] + 1e-6)
    row["right_eye_aspect"] = row["right_eye_height"] / (row["right_eye_width"] + 1e-6)
    row["eye_aspect_asymmetry"] = row["left_eye_aspect"] - row["right_eye_aspect"]
    row["eye_width_asymmetry"] = row["left_eye_width"] - row["right_eye_width"]
    row["eye_height_asymmetry"] = row["left_eye_height"] - row["right_eye_height"]
    row["avg_gaze_x"] = (row["left_relative_pupil_x"] + row["right_relative_pupil_x"]) / 2.0
    row["avg_gaze_y"] = (row["left_relative_pupil_y"] + row["right_relative_pupil_y"]) / 2.0
    row["gaze_magnitude"] = np.sqrt(row["avg_gaze_x"] ** 2 + row["avg_gaze_y"] ** 2)
    row["pupil_x_divergence"] = row["left_relative_pupil_x"] - row["right_relative_pupil_x"]
    row["pupil_y_divergence"] = row["left_relative_pupil_y"] - row["right_relative_pupil_y"]
    row["ear_x_gaze_mag"] = row["avg_ear"] * row["gaze_magnitude"]
    row["ear_x_gaze_y"] = row["avg_ear"] * row["avg_gaze_y"]
    row["is_wide_open"] = 1.0 if row["avg_ear"] > 0.30 else 0.0
    row["is_squinting"] = 1.0 if row["avg_ear"] < 0.22 else 0.0

    return np.array([row.get(f, 0.0) for f in EYE_FEATURES], dtype=np.float32)


def predict_eye_proba(feature_vector: np.ndarray) -> Optional[np.ndarray]:
    if eye_model is None:
        return None
    X = feature_vector.reshape(1, -1)
    if eye_scaler is not None:
        X = eye_scaler.transform(X)
    if hasattr(eye_model, "predict_proba"):
        return eye_model.predict_proba(X)[0].astype(np.float32)
    if hasattr(eye_model, "decision_function"):
        scores = eye_model.decision_function(X)
        scores = np.asarray(scores).reshape(-1)
        exp = np.exp(scores - np.max(scores))
        return exp / exp.sum()
    return None


def predict_fer_proba(image: Image.Image) -> Optional[np.ndarray]:
    if fer_model is None:
        return None
    tensor = build_fer_input(image)
    with torch.no_grad():
        logits = fer_model(tensor)
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return proba.astype(np.float32)


@app.on_event("startup")
def startup_event():
    load_models()
    if fer_model is None:
        print("Warning: FER model not loaded.")
    if eye_model is None:
        print("Warning: Eye Tracking model not loaded.")


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}")

    fer_proba = predict_fer_proba(image)
    eye_proba = extract_eye_features(image)
    eye_proba = predict_eye_proba(eye_proba) if eye_proba is not None else None

    if fer_proba is None and eye_proba is None:
        raise HTTPException(status_code=503, detail="No model outputs available.")

    if fer_proba is not None and eye_proba is not None:
        fused = (FER_WEIGHT * fer_proba) + (EYE_WEIGHT * eye_proba)
    elif fer_proba is not None:
        fused = fer_proba
    else:
        fused = eye_proba

    best_index = int(np.argmax(fused))
    return PredictionResponse(
        top_emotion=EMOTIONS[best_index],
        confidence=float(fused[best_index] * 100.0),
        emotions=[
            EmotionProb(label=EMOTIONS[i], value=float(fused[i] * 100.0))
            for i in range(len(EMOTIONS))
        ],
    )
