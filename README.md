# Soldier Drowsiness Detection

A small computer-vision and machine-learning project that detects signs of drowsiness from a live webcam feed using OpenCV and a Random Forest classifier.

This repository is intended as a beginner-friendly educational demo and is not a medical or operational safety system.

## Overview

The project uses a simple, interpretable feature set built from face and eye measurements:

- left and right eye openness estimates
- average eye openness
- mouth ratio
- head pose offsets
- consecutive closed-eye frames
- blink ratio

A Random Forest model is trained on a synthetic dataset and then used in real time to classify the subject as either `ALERT` or `DROWSY`.

## Features

- OpenCV Haar cascades for face and eye detection
- Real-time webcam inference using a trained sklearn model
- Synthetic training data in `data/drowsiness_data.csv`
- Model evaluation with accuracy, precision, recall, F1-score, and confusion matrix
- Visualization of model performance and feature importance

## Repository structure

```text
soldier_drowsiness_ml/
├── LICENSE
├── README.md
├── .gitignore
├── requirements.txt
├── main.py
├── train_model.py
├── model.pkl
├── data/
│   └── drowsiness_data.csv
├── outputs/
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── .venv/
```

## Quick start

Clone the repository and create a virtual environment:

```bash
cd soldier_drowsiness_ml
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# or: .\.venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

This script:

- loads the dataset from `data/drowsiness_data.csv`
- trains a Random Forest classifier
- reports evaluation metrics
- saves the trained model to `model.pkl`
- saves plots to `outputs/`

## Run the webcam demo

```bash
python main.py
```

Optional demo mode without a webcam:

```bash
python main.py --demo --demo-seconds 2.5
```

Press `q` in the OpenCV window to quit the live demo.

## Dataset details

The dataset is intentionally small and synthetic for educational use. Labels are:

- `0` = `ALERT`
- `1` = `DROWSY`

The project is designed to demonstrate the full ML workflow, not to serve as a production safety system.

## Notes

- Model accuracy is shown on a held-out test set.
- Feature importance values are available in `outputs/feature_importance.png`.
- The confusion matrix is available in `outputs/confusion_matrix.png`.
- The repository includes `.gitignore` so local virtual environments are not accidentally committed.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

