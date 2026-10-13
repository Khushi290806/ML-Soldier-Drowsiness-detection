"""Real-time soldier drowsiness detection demo using a webcam and a trained Random Forest model."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.pkl"
WINDOW_NAME = "Soldier Drowsiness Detection"
FEATURES = [
    "left_ear",
    "right_ear",
    "avg_ear",
    "mouth_ratio",
    "head_pitch",
    "head_yaw",
    "eye_closure_frames",
    "blink_ratio",
]

CLOSED_EAR_THRESHOLD = 0.18
TEMPORAL_WARNING_FRAMES = 8
DROWSY_PROBABILITY_THRESHOLD = 0.55


def load_model():
    """Load the trained machine learning model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model was found. Run 'python train_model.py' before starting the webcam demo."
        )
    return joblib.load(MODEL_PATH)


def make_model_row(
    left_ear: float,
    right_ear: float,
    mouth_ratio: float,
    head_pitch: float,
    head_yaw: float,
    eye_closure_frames: int,
    blink_ratio: float,
) -> list[float]:
    """Build a feature vector in the exact shape expected by the trained model."""
    avg_ear = (left_ear + right_ear) / 2.0
    return [
        left_ear,
        right_ear,
        avg_ear,
        mouth_ratio,
        head_pitch,
        head_yaw,
        eye_closure_frames,
        blink_ratio,
    ]


def build_feature_frame(feature_row: list[float]) -> pd.DataFrame:
    """Create the DataFrame expected by the trained Random Forest classifier."""
    return pd.DataFrame([feature_row], columns=FEATURES)


def detect_face_and_features(frame: np.ndarray, face_cascade, eye_cascade):
    """Return a feature tuple if a face is detected; otherwise return None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None

    face = max(faces, key=lambda item: item[2] * item[3])
    x, y, w, h = [int(v) for v in face]
    face_roi = gray[y : y + h, x : x + w]

    eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    if len(eyes) < 2:
        return None

    eyes = sorted(eyes, key=lambda item: item[0])
    left_eye = eyes[0]
    right_eye = eyes[1]

    left_ear = compute_eye_ratio(left_eye)
    right_ear = compute_eye_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0

    mouth_width = max(1, int(w * 0.40))
    mouth_height = max(1, int(h * 0.18))
    mouth_ratio = mouth_height / max(mouth_width, 1)

    head_pitch = ((y + h / 2) - (frame.shape[0] / 2)) / (frame.shape[0] / 2)
    head_yaw = ((x + w / 2) - (frame.shape[1] / 2)) / (frame.shape[1] / 2)

    return left_ear, right_ear, avg_ear, mouth_ratio, head_pitch, head_yaw


def compute_eye_ratio(eye_box):
    """Approximate EAR using the eye ROI dimensions."""
    ex, ey, ew, eh = [int(v) for v in eye_box]
    if ew <= 0 or eh <= 0:
        return 0.12
    return max(0.04, min(0.42, eh / max(ew, 1)))


def draw_overlay(frame: np.ndarray, status: str, probability: float, ear_value: float, warning: bool) -> None:
    """Draw the information panel and warning text on the webcam frame."""
    height, width = frame.shape[:2]
    panel_x, panel_y, panel_w, panel_h = 20, 20, 360, 170
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (80, 220, 255), 2)

    cv2.putText(
        frame,
        "SOLDIER DROWSINESS DETECTION",
        (panel_x + 20, panel_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    cv2.putText(frame, f"STATUS: {status}", (panel_x + 20, panel_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"ML Prediction: {status}", (panel_x + 20, panel_y + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Drowsiness Probability: {probability:.1f}%", (panel_x + 20, panel_y + 124), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"EAR: {ear_value:.2f}", (panel_x + 20, panel_y + 152), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    if warning:
        warning_y = height - 70
        cv2.rectangle(frame, (10, warning_y - 25), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.putText(frame, "!!! DROWSINESS DETECTED !!!", (30, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


def play_warning_beep() -> None:
    """Play a short beep on Windows when drowsiness is sustained."""
    if os.name == "nt":
        import winsound

        winsound.Beep(700, 180)


def get_cascades():
    """Load and validate the OpenCV Haar cascade files used for detection."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    if face_cascade.empty() or eye_cascade.empty():
        raise RuntimeError("OpenCV Haar cascade files could not be loaded. Check your OpenCV installation.")

    return face_cascade, eye_cascade


def run_demo(camera_index: int, demo_mode: bool = False, demo_seconds: float = 2.0) -> None:
    """Run webcam detection or a short synthetic demo."""
    model = load_model()
    face_cascade, eye_cascade = get_cascades()

    if demo_mode:
        print("Running synthetic demo mode for verification.")
        frame_rate = 15
        total_frames = int(frame_rate * demo_seconds)
        for i in range(total_frames):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (150, 80), (480, 420), (70, 180, 255), 2)
            cv2.rectangle(frame, (220, 170), (290, 240), (255, 255, 255), 2)
            cv2.rectangle(frame, (340, 170), (410, 240), (255, 255, 255), 2)
            cv2.rectangle(frame, (260, 250), (380, 310), (255, 255, 255), 2)

            # Simulate a sleepy face during the second half of the demonstration.
            eye_open = 0.22 + 0.05 * np.sin(i / 10)
            if i > total_frames * 0.6:
                eye_open = 0.10

            left_ear = float(np.clip(eye_open, 0.05, 0.38))
            right_ear = float(np.clip(eye_open * 1.02, 0.05, 0.38))
            avg_ear = (left_ear + right_ear) / 2.0
            mouth_ratio = 0.18 if i < total_frames * 0.6 else 0.42
            head_pitch = 0.08
            head_yaw = -0.05
            eye_closure_frames = 0 if avg_ear > CLOSED_EAR_THRESHOLD else 12
            blink_ratio = max(0.0, min(1.0, (0.22 - avg_ear) / 0.22))

            feature_row = make_model_row(left_ear, right_ear, mouth_ratio, head_pitch, head_yaw, eye_closure_frames, blink_ratio)
            feature_frame = build_feature_frame(feature_row)
            prediction = int(model.predict(feature_frame)[0])
            probabilities = model.predict_proba(feature_frame)[0]
            probability = float(probabilities[prediction] * 100.0)
            status = "DROWSY" if prediction == 1 else "ALERT"
            draw_overlay(frame, status, probability, avg_ear, prediction == 1)
            cv2.putText(frame, f"Synthetic demo frame {i + 1}/{total_frames}", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Drowsiness Demo", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}. Try another camera index or connect a webcam.")

    consecutive_closed = 0
    warning_active = False
    last_beep_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            overlay = frame.copy()
            detected = detect_face_and_features(frame, face_cascade, eye_cascade)
            if detected is None:
                cv2.putText(overlay, "FACE NOT DETECTED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow(WINDOW_NAME, overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            left_ear, right_ear, avg_ear, mouth_ratio, head_pitch, head_yaw = detected
            if avg_ear < CLOSED_EAR_THRESHOLD:
                consecutive_closed += 1
            else:
                consecutive_closed = 0

            blink_ratio = max(0.0, min(1.0, (0.22 - avg_ear) / 0.22))
            feature_row = make_model_row(
                left_ear,
                right_ear,
                mouth_ratio,
                head_pitch,
                head_yaw,
                consecutive_closed,
                blink_ratio,
            )

            feature_frame = build_feature_frame(feature_row)
            prediction = int(model.predict(feature_frame)[0])
            probabilities = model.predict_proba(feature_frame)[0]
            probability = float(probabilities[prediction] * 100.0)

            status = "DROWSY" if prediction == 1 else "ALERT"
            warning = (
                prediction == 1 and consecutive_closed >= TEMPORAL_WARNING_FRAMES and probability >= DROWSY_PROBABILITY_THRESHOLD * 100.0
            )

            if warning:
                warning_active = True
                if time.monotonic() - last_beep_time > 2.5:
                    play_warning_beep()
                    last_beep_time = time.monotonic()
            else:
                warning_active = False

            draw_overlay(overlay, status, probability, avg_ear, warning_active)
            cv2.imshow(WINDOW_NAME, overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Real-time drowsiness detection demo.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index to open (default: 0).")
    parser.add_argument("--demo", action="store_true", help="Run a synthetic demo instead of using the webcam.")
    parser.add_argument("--demo-seconds", type=float, default=2.5, help="Demo duration in seconds for --demo mode.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args.camera_index, demo_mode=args.demo, demo_seconds=args.demo_seconds)
