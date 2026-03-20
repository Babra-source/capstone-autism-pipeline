export type Emotion = {
  label: string;
  value: number;
};

export type EyeFeaturesData = {
  ear_left: number;
  ear_right: number;
  gaze: string;
  blink: boolean;
};

export type ResultData = {
  top_emotion: string;
  confidence: number;
  emotions: Emotion[];
  eye_features?: EyeFeaturesData; // optional for FER-only
};