# Face Emotion System: Complete Project Guide

This guide explains the project from its purpose and data sources through
training, live inference, the API, dashboard, evaluation, and known limits.
Use the shorter [README](README.md) for quick setup and commands.

## 1. What the project does

This is a local webcam application for classifying **visible facial-expression
categories**. Given a video frame, it looks for a face, resizes the detected
grayscale crop to 48 x 48 pixels, and asks a convolutional neural network (CNN)
to score seven labels:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral

It is not an identity-recognition system and does not determine a person's
internal emotional state. Its outputs are estimates learned from FER-2013
labels, which can be ambiguous, especially for small or subtle expressions.

## 2. End-to-end architecture

```text
Webcam
  |
  v
OpenCV frame capture
  |
  v
Grayscale frame -> Haar cascades -> require overlapping multi-cascade support
  |
  v
Crop selected face -> resize 48x48 -> scale pixels to [0, 1]
  |
  v
Trained FER CNN -> seven scores
  |
  v
Neutral logit adjustment -> 5-frame face-track smoothing
  |
  v
Top expression / Uncertain label + confidence threshold
  |                                   |
  v                                   v
Draw box and label                 CSV event log
  |                                   |
  +------ MJPEG API stream ----------+---- events API
                                         |
                                         v
                              React/Vite dashboard
```

There are two ways to display the live pipeline:

- `scripts/realtime_demo.py` runs the OpenCV camera window directly.
- `backend.main` owns the camera and exposes the MJPEG stream and CSV events;
  the React dashboard consumes those endpoints.

Run only one camera client at a time. Opening the standalone demo and dashboard
simultaneously can make either client fail to acquire the camera.

## 3. Repository layout and responsibility

### Root files

- `README.md` — installation, quickstart, common workflows, and limitations.
- `PROJECT_GUIDE.md` — this detailed system explanation.
- `config.yaml` — local dataset/model/log paths, camera index, confidence
  cutoff, and the Neutral decision offset.
- `requirements.txt` — Python runtime, training, evaluation, and notebook
  dependencies.
- `.gitignore` — excludes local datasets, environments, webcam images, logs,
  and generated model artifacts.
- `.gitattributes` — stores the active CNN checkpoint through Git LFS.

### `src/`

- `src/app/camera_handler.py` opens, reads, and releases a camera.
- `src/app/detector.py` performs real-time face and expression inference,
  temporal smoothing, low-confidence handling, drawing, and event logging.
- `src/models/face_detector.py` finds and filters a single face using multiple
  Haar cascades.
- `src/models/emotion_cnn.py` defines the CNN and the shared postprocessing
  helper for class-logit offsets.
- `src/preprocessing/data_loader.py` parses FER-2013 CSV rows into image
  tensors, labels, and train/validation/test partitions.
- `src/preprocessing/webcam_dataset.py` reads optional captured webcam crops
  and ensures recording sessions do not overlap between train and validation.
- `src/utils/logger.py` appends timestamped events to the configured CSV.
- `src/utils/visualizer.py` draws face rectangles and text labels.

### `scripts/`

- `train_model.py` fine-tunes or trains the FER CNN and guards candidate
  promotion.
- `evaluate_model.py` evaluates the configured active checkpoint on FER
  validation and test splits without training.
- `download_data.py` validates a local FER CSV or downloads the permitted
  competition archive with the user's configured Kaggle CLI.
- `capture_emotion_samples.py` records manually labeled face crops for optional
  webcam adaptation.
- `audit_webcam_dataset.py` reports crops where the face detector finds no face;
  it does not edit or relabel images.
- `realtime_demo.py` launches the standalone camera window.

### `backend/`

- `backend/main.py` defines the FastAPI health endpoint, recent-event endpoint,
  and MJPEG camera feed. It reads events from the configured CSV.

### `frontend/`

- `frontend/src/components/VideoFeed.jsx` displays the MJPEG stream.
- `useEventsData.jsx` polls the recent-events endpoint.
- `EmotionChart.jsx` and `EmotionDistribution.jsx` summarize expression events.
- `SummaryCards.jsx` shows event totals and prediction summaries.
- `RecentEventsTable.jsx` displays logged expression events, not attendance or
  person identity.
- `App.jsx`, `Dashboard.jsx`, `main.jsx`, and CSS files compose and style the
  user interface.
- `package.json` / `package-lock.json` define a reproducible npm dependency
  install.

### `models/`

- `emotion_model.h5` is the active, trained seven-class model. Git stores it
  using LFS because of its size.
- `haarcascade_frontalface_default.xml` is the configured OpenCV detector
  cascade.
- Candidate checkpoints, training reports, and evaluation output are generated
  locally and are not repository source.

### `notebooks/`

- `01_EDA_FER2013.ipynb` explores the FER dataset.
- `02_Emotion_Model_Training.ipynb` documents exploratory CNN training.

Notebook output cells are cleared in the repository to keep data-derived
results and generated images out of source control. The notebooks are
exploratory material; the supported reproducible training/evaluation entry
points are the scripts in `scripts/`.

### `tests/`

`unittest` tests cover data loading, webcam dataset constraints, CNN behavior,
face detection safeguards, live inference behavior, Kaggle archive handling,
and backend endpoints.

## 4. Data and emotion labels

### FER-2013

The training and evaluation scripts expect the FER-2013 CSV locally at
`data/fer2013/fer2013.csv` by default. It has `emotion`, `pixels`, and `Usage`
columns. Each pixel field contains 2,304 grayscale integers for a 48 x 48
image. FER integer classes are interpreted in this order:

| Class ID | Expression |
|---:|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

Standard `Training`, `PublicTest`, and `PrivateTest` rows become the training,
validation, and test partitions. Training fits on the `Training` partition;
validation selects checkpoints and can guide tuning; `PrivateTest` is held out
for reporting and is not the model-selection split.

FER-2013 is low resolution, imbalanced, and contains inherently ambiguous
expression labels. It is useful as a benchmark but does not represent every
camera, lighting condition, population, or expression style.

### Optional webcam data

Captured images are placed under a structure like:

```text
data/webcam_emotions/
  train/
    train-session-1/
      Angry/
      Disgust/
      Fear/
      Happy/
      Sad/
      Surprise/
      Neutral/
  validation/
    validation-session-1/
      Angry/
      Disgust/
      Fear/
      Happy/
      Sad/
      Surprise/
      Neutral/
```

Images use manually assigned labels. Separate session names prevent direct
session reuse across the train and validation partitions, but session
separation alone does not prove person independence. For a claim about unseen
people, the validation people must not appear in training. Ask for consent,
avoid names in folder paths, and keep these face images private.

The capture utility saves only when a key is pressed, which allows the user to
check that the displayed face box and selected label are appropriate. The
dataset audit only checks whether a face is detectable; it cannot verify that
the human-provided label is correct.

## 5. CNN and model training

`src/models/emotion_cnn.py` defines the project's 48 x 48 grayscale CNN:

1. Input: one grayscale face, shape `(48, 48, 1)`.
2. Mild training-only random horizontal flip, rotation, translation, and zoom.
3. Convolution blocks:
   - 64 filters, two 3 x 3 convolutions, batch normalization, pooling, dropout.
   - 128 filters, two 3 x 3 convolutions, batch normalization, pooling, dropout.
   - 256 filters, one 3 x 3 convolution, batch normalization, pooling, dropout.
4. Flatten, a 512-unit ReLU dense layer, batch normalization and dropout.
5. Seven-class softmax output.

Augmentation layers are inactive during inference. By default,
`train_model.py` starts from the compatible active checkpoint, uses a low
learning rate, and writes a separate candidate. A callback selects candidate
weights using validation macro F1, which gives each class equal weight.
Early stopping and learning-rate reduction limit unhelpful training. The
PrivateTest split does not choose the candidate.

The candidate becomes active only when the configured validation gates pass.
If gates fail, the current `models/emotion_model.h5` remains unchanged.
`--from-scratch` is available for deliberate experiments but is not the normal
path. Class weighting can trade precision or overall accuracy for recall, so
it is optional and should be evaluated rather than assumed beneficial.

Example default fine-tune:

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py
```

Example deliberately separate experiment:

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py --class-weighting --epochs 5 --learning-rate 0.00001 --artifacts-dir models\training\fer_class_balanced
```

The candidate, epoch history, confusion matrix, classification report, and
`training_results.json` are saved below the selected artifacts directory.
They are ignored by Git and do not replace the active model unless promotion
checks pass.

### Webcam fine-tuning (optional)

Webcam adaptation should only be tried with clean crops, correct labels,
balanced classes, and a validation protocol appropriate to the intended users.
The existing FER checkpoint stays in use if webcam validation fails its quality
gates or degrades the FER validation score beyond the allowed bound.

Capture one training session (repeat with different consenting volunteers to
include more people):

```powershell
.\.venv\Scripts\python.exe scripts\capture_emotion_samples.py --split train --session train-day1 --samples-per-class 100 --output-dir data\webcam_emotions --camera-index 0
```

Capture a separate validation session:

```powershell
.\.venv\Scripts\python.exe scripts\capture_emotion_samples.py --split validation --session validation-day1 --samples-per-class 50 --output-dir data\webcam_emotions --camera-index 0
```

Check the dataset and inspect questionable/mislabeled images manually:

```powershell
.\.venv\Scripts\python.exe scripts\audit_webcam_dataset.py --dataset-dir data\webcam_emotions
```

Fine-tune into a distinct artifacts folder:

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py --webcam-dir data\webcam_emotions --epochs 10 --artifacts-dir models\training\webcam
```

Read `models/training/webcam/webcam/training_results.json` to see whether the
candidate passed promotion checks. Do not infer cross-person performance from
a train/validation split containing only one person.

## 6. Inference, confidence, and live face detection

For each live frame, `RealTimeDetector`:

1. Converts the BGR camera frame to grayscale.
2. Uses the configured Haar cascade plus OpenCV's alternate frontal-face
   cascades. For single-subject operation, a candidate must overlap with
   detections from at least two cascades; the best supported/largest face is
   retained.
3. Crops the selected face and resizes it to 48 x 48 pixels.
4. Scales the image values by 255 to match training input range `[0, 1]`.
5. Calls the saved CNN with inference behavior.
6. Applies configured per-class logit offsets before choosing a class. Current
   config uses `Neutral: -0.75` to reduce Neutral overprediction measured on
   FER-2013.
7. Averages up to five adjacent-frame score vectors for the same tracked face
   to reduce flicker.
8. Labels the frame with the top class only when its adjusted top score meets
   `thresholds.emotion_min_confidence`; otherwise it reports `Uncertain` and
   includes the top estimate.
9. Draws the box/label and asks the CSV logger to record the event subject to
   a per-track, per-expression cooldown.

The `0.70` threshold is a reject/uncertainty display rule, not a target the
network can be made to satisfy. Scores after logit adjustment are normalized
decision scores but are not calibrated probabilities. A more frequent label
or a higher displayed number is not evidence of better correctness.

The multiple-cascade face check was introduced because the earlier live
screenshots showed background regions being boxed instead of the face. Correct
face localization is a prerequisite for meaningful expression testing; it
cannot by itself make ambiguous expressions easy to classify.

## 7. Evaluation and how to read results

Run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_model.py
```

The script evaluates the currently configured model on FER validation and
PrivateTest partitions using the same logit offsets as live inference. It
writes:

- `models/evaluation/evaluation_results.json`
- `models/evaluation/validation_confusion_matrix.csv`
- `models/evaluation/test_confusion_matrix.csv`

Important fields:

- **Precision:** of predictions made as a class, the proportion correct.
- **Recall:** of actual examples in a class, the proportion recognized.
- **F1:** the harmonic mean of precision and recall.
- **Macro F1:** mean class F1 with equal class weighting.
- **Confusion matrix:** row = true class, column = predicted class.

The Neutral offset was selected using validation and had one final check on
the untouched test split. Before the offset, test accuracy/macro F1 were
approximately 82.47%/0.8208; after it they were about 83.45%/0.8296. In that
test report, Angry recall was about 72%, Disgust 84%, Fear 68%, Sad 79%, Happy
97%, Surprise 93%, and Neutral 84%. These are FER benchmark measurements,
not guaranteed webcam performance. Precision-recall trade-offs are class
dependent.

When webcam predictions appear wrong:

1. Confirm the box tightly encloses the face, not a background region.
2. Make sure only one client owns the camera.
3. Keep the face forward-facing, large enough in frame, and adequately lit.
4. Read the top estimate and score; a label below threshold should appear as
   Uncertain.
5. Evaluate a set of correctly labeled examples from the actual target camera
   and intended population before training or changing thresholds.
6. Change one variable at a time and preserve a held-out validation set.

Do not choose a model by looking at training accuracy or a few live frames.
Do not tune repeatedly on PrivateTest; doing so leaks information from the
held-out test split.

## 8. API and dashboard

Start the backend from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /` — status and availability of the configured active checkpoint.
- `GET /events?limit=100` — most recent rows from the CSV event log.
- `GET /video_feed` — multipart MJPEG stream with face boxes and expression
  labels.

Events are written to `data/logs/events.csv` with timestamp, generic source
`Face`, and expression. The project does not match faces to names.

In another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

The frontend defaults to API root `http://127.0.0.1:8000`. Override with
`VITE_API_URL` when running the dashboard against another local API address.
The `useEventsData` hook polls the events endpoint every five seconds; the
video element consumes the MJPEG endpoint directly.

## 9. Configuration

`config.yaml`:

- `paths.fer_csv` — local FER dataset path.
- `paths.logs_csv` — local CSV event output path.
- `paths.emotion_model` — active CNN path.
- `paths.training_artifacts_dir` — ignored training output directory.
- `paths.haar_cascade` — primary face cascade path.
- `video.camera_index` — OpenCV camera index.
- `thresholds.emotion_min_confidence` — minimum displayed score before
  returning an expression instead of `Uncertain`.
- `thresholds.event_cooldown_seconds` — event logging throttle duration.
- `inference.emotion_logit_adjustments` — optional class-specific decision
  offsets; current Neutral offset is `-0.75`.

Use project-relative paths for portable local setup. Do not place credentials
or personal information in committed config.

## 10. Validation commands

Python tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Frontend production build:

```powershell
cd frontend
npm ci
npm run build
```

FER evaluation (requires local CSV and active checkpoint):

```powershell
cd ..
.\.venv\Scripts\python.exe scripts\evaluate_model.py
```

The test suite does not require camera hardware. A passing unit suite or
frontend build does not establish model accuracy on live footage.

## 11. Project history and cleanup boundaries

The runtime app is an expression-only demo. Earlier experimental code for
identity recognition/enrollment, face alignment, engagement scoring, and an
unmounted SQL database/WebSocket API was removed because it had no callers in
the active app and did not contribute to the supported expression workflow.
The dashboard's existing event table is not an identity/attendance feature.

Local data, trained experiments, and virtual environments are intentionally
not removed from the developer's machine. They are ignored by Git so source
control remains limited to the reproducible project, test suite, notebooks,
assets required for runtime, and active LFS checkpoint.

## 12. Responsible use

Facial-expression labels are not reliable ground truth for internal emotion.
The model can fail on occlusion, pose, motion blur, small faces, lighting,
different cameras, cultural variation in expression, or populations unlike
the training data. FER-2013 labels are noisy and class-imbalanced; webcams
usually differ significantly from its dataset images. Do not use this demo
for hiring, discipline, health, access control, grading, or any other
consequential evaluation of people. Obtain consent before collecting images,
store them securely, and delete them when no longer needed.
