# Face Emotion System

A local facial-expression demo that detects one face, estimates one of seven
visible expression categories with a FER-2013 CNN, records low-rate prediction
events, and displays the camera feed and recent events in a React dashboard.
It does not identify people or measure anyone's internal emotional state.

## Features

- Haar-cascade face detection with independent-cascade agreement to reduce
  background false positives.
- A 48 x 48 grayscale CNN trained for Angry, Disgust, Fear, Happy, Sad,
  Surprise, and Neutral.
- A configurable Neutral logit offset and a confidence threshold that reports
  low-scoring predictions as `Uncertain`.
- Optional webcam expression capture and guarded fine-tuning.
- FER-2013 evaluation with per-class precision, recall, F1, and confusion
  matrices.
- FastAPI health, events, and MJPEG video endpoints.
- React/Vite dashboard for the live feed, event timeline, distribution, and
  recent predictions.

## Project structure

```text
backend/                 FastAPI health, CSV events, and MJPEG video stream
config.yaml              Model/data paths, camera, confidence, and logit settings
data/                    Local-only FER CSV, webcam captures, and event logs
frontend/                React dashboard and Vite configuration
models/                  Git-LFS CNN checkpoint and Haar face cascade
notebooks/               FER exploration and CNN training notebooks
scripts/                 Training, evaluation, data download, capture, and demo
src/app/                 Camera handler and real-time expression inference
src/models/              CNN architecture and face detector
src/preprocessing/       FER and optional webcam dataset loaders
src/utils/               CSV event logger and drawing helpers
tests/                   Python unit tests
requirements.txt         Python dependencies
```

## Requirements

- Windows, macOS, or Linux; Python 3.10–3.13 with a TensorFlow wheel available
  for the selected Python version.
- Node.js and npm to run the dashboard.
- A webcam for live operation.
- Git LFS to fetch the included trained CNN checkpoint when cloning.
- The FER-2013 CSV locally for evaluation or training.

## Setup (Windows PowerShell)

Run from the cloned project directory:

```powershell
cd C:\path\to\face_emotion
git lfs install
git lfs pull
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The trained checkpoint `models/emotion_model.h5` is stored via Git LFS. The
dataset and personal webcam images are not stored in this repository.

## FER-2013 dataset

Place the permitted FER-2013 CSV at `data/fer2013/fer2013.csv`. It must contain
`emotion`, `pixels`, and `Usage` columns. Pixel values are 48 x 48 grayscale
integers; the standard `Training`, `PublicTest`, and `PrivateTest` splits map to
training, validation, and test.

Validate the local CSV and print split/class counts:

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
```

To download via the Kaggle CLI, first install/configure it, sign in to Kaggle,
and accept the FER-2013 competition terms:

```powershell
.\.venv\Scripts\python.exe -m pip install kaggle
.\.venv\Scripts\python.exe scripts\download_data.py --download
```

Dataset availability is subject to the Kaggle competition's terms. The app
does not download the dataset automatically. Dataset files, webcam captures,
logs, local identity data, virtual environments, and generated training
artifacts are ignored by Git; keep personal images local and obtain consent
before collecting them.

## Start the live demo

### Standalone OpenCV camera window

```powershell
.\.venv\Scripts\python.exe scripts\realtime_demo.py
```

Press **q** to quit. Only one program/browser client should use the camera at a
time. Camera index is `video.camera_index` in `config.yaml` (default `0`).

### Web dashboard

Start the API in one PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Start the dashboard in another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://localhost:5173`).
The backend health page is `http://127.0.0.1:8000/`; the most recent events
are at `http://127.0.0.1:8000/events?limit=100`. The live stream endpoint is
`http://127.0.0.1:8000/video_feed`.

## Evaluate the active checkpoint

With FER-2013 available locally:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_model.py
```

Reports and confusion matrices are written to the ignored local directory
`models/evaluation/`. The configured Neutral adjustment is included in the
evaluation so metrics reflect live decision behavior. The current measured
FER-2013 result is approximately 82.70% validation accuracy / 0.828 macro F1
and 83.45% PrivateTest accuracy / 0.830 macro F1. These are FER dataset
results, not a guarantee of webcam accuracy or performance on every
population.

## Train or fine-tune the CNN

The included checkpoint is ready for inference. To fine-tune it on FER-2013:

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py
```

Training writes candidates, history, and metrics under `models/training/`.
Candidates are selected by validation macro F1 and replace the active
checkpoint only if guarded accuracy, macro-F1, and per-class recall criteria
pass. The held-out PrivateTest set is evaluation-only. `--from-scratch` trains
from random initialization and is not the recommended default.

Optional webcam adaptation requires accurately labeled crops and distinct
train/validation recording sessions. For generalization to other people,
validation must include people not represented in training. Capture commands
and acceptance checks are described in [PROJECT_GUIDE.md](PROJECT_GUIDE.md).
Small single-person webcam collections are not evidence of cross-person
accuracy.

## Tests

Run the Python unit tests without needing pytest:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Build the React dashboard:

```powershell
cd frontend
npm ci
npm run build
```

## Prediction and privacy notes

The model predicts dataset-defined facial-expression categories. It cannot
reliably infer a person's internal emotional state, and FER-2013 is a
low-resolution dataset with known class ambiguity and imbalance. Inference
scores are adjusted decision scores, not calibrated probabilities. The
configured 0.70 threshold controls when the UI says `Uncertain`; it does not
make predictions more accurate. Lighting, face crop quality, pose, camera
differences, expression intensity, and demographic differences can all affect
results. Do not use this demo for consequential decisions.

The application does not perform person identification. The live event logger
records a generic `Face` source, timestamp, and expression in
`data/logs/events.csv`; the camera feed is processed locally by the backend.
Do not commit FER data, webcam images, credentials, or local event logs.

For a full architecture and end-to-end explanation, see
[PROJECT_GUIDE.md](PROJECT_GUIDE.md).
