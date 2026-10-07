# Face Emotion System

A local facial-expression demo with a FER-2013 CNN, webcam face detection, optional
known-person classification, a FastAPI event endpoint, and a React dashboard.
Expression labels describe visible facial-expression categories; they do not
establish a person's internal emotional state.

## Project layout

```text
backend/                 FastAPI health and recent-event endpoints
data/                    Local FER-2013 data, webcam captures, and event logs (not tracked)
frontend/                Vite + React analytics dashboard
models/                  Runtime emotion model (Git LFS) and face detector assets
notebooks/               Exploratory analysis and training notebooks
scripts/train_model.py   CNN fine-tuning and held-out test evaluation
scripts/capture_emotion_samples.py  Capture manually labeled webcam expressions
scripts/evaluate_model.py Evaluate the current model without training
scripts/train_lbph.py    Train the optional known-person KNN classifier
src/models/              CNN and face detector/recognizer implementations
```

## FER-2013 dataset

The project expects a local FER-2013 CSV at `data/fer2013/fer2013.csv`. The
dataset is not stored in this Git repository. It must have the columns
`emotion`, `pixels`, and `Usage`, with 48x48 grayscale pixels stored as
space-separated integers from 0 to 255. Standard FER-2013 usage values
(`Training`, `PublicTest`, `PrivateTest`) are mapped to training, validation,
and test respectively. The training script does not mix the held-out splits
into training.

Validate the local dataset and print its split/class counts:

```powershell
python scripts\download_data.py
```

If the CSV is not present, install and configure the Kaggle CLI, accept the
FER-2013 competition rules in your Kaggle account, then download and validate:

```powershell
pip install kaggle
python scripts\download_data.py --download
```

The download script uses your Kaggle CLI credentials; it never asks for or
stores an API token itself. Dataset files, webcam captures, logs, and generated
training artifacts are excluded from Git. Do not publish the FER-2013 dataset
or personal face images without confirming applicable license, consent, and
privacy requirements. To use a different location, update `paths.fer_csv` in
`config.yaml` or pass `--dataset` to the training script.

## Environment setup

Use Python 3.10-3.13 with a TensorFlow build that supports your interpreter.
On Windows, create and activate a virtual environment from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the venv Python directly, for example
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

The active checkpoint `models/emotion_model.h5` is stored with Git LFS.
Install Git LFS before cloning if you want Git to fetch that checkpoint.

## Train and evaluate the emotion CNN

From the repository root:

```powershell
python scripts\train_model.py
```

The default run fine-tunes the validated active checkpoint using a low learning
rate and mild training-only augmentation. It allows up to 20 additional epochs,
uses validation-based checkpointing, early stopping, and learning-rate
reduction. By default, it does not apply class weights, which can trade overall
accuracy for minority-class recall. Add `--class-weighting` if that trade-off
is desired.

Training saves a separate candidate checkpoint. FER training selects that
checkpoint by validation macro F1 (which weights the seven classes equally),
and only replaces the active model if macro F1 improves, validation accuracy
does not regress beyond the configured limit, and every class meets the
minimum recall. The independent `PrivateTest` split is reported for evaluation
and is never used to select the model. If training is interrupted, the active
model remains intact and the best candidate is resumable:

```powershell
python scripts\train_model.py --resume
```

Use `--from-scratch` only to deliberately ignore the pretrained checkpoint.
For a fresh scratch run, start without `--resume`; if a candidate file from a
previous run exists, move it aside before starting another run. The original
lower-performing interrupted scratch run is retained under
`models/training/interrupted_scratch_*` for reference.

The active CNN checkpoint, before inference-time class adjustment, scored
81.39% validation accuracy and 82.47% on the FER-2013 `PrivateTest` split
(3,589 examples). The current Neutral logit adjustment is also applied during
evaluation and scored 82.70% validation accuracy and 83.45% PrivateTest
accuracy. These remain dataset benchmark scores, not expected webcam accuracy.
Fine-tuning starts from the active checkpoint and cannot replace it unless
validation performance improves. The manually interrupted from-scratch run
reached only 61.24% validation accuracy and 62.44% PrivateTest accuracy, so it
is not used for inference.

To prioritize balanced FER-2013 performance across emotion classes, fine-tune
from the active checkpoint with class weighting. Keep the default promotion
gates; inspect macro F1 and each class's precision/recall in the resulting
report rather than trying to force every output score above a threshold:

```powershell
python scripts\train_model.py --class-weighting --epochs 5 --learning-rate 0.00001 --artifacts-dir models\training\fer_class_balanced
```

This is a deliberate experiment, not a guarantee of improved results. It
trains a separate candidate and leaves the active checkpoint unchanged unless
held-out validation metrics pass the promotion gates. Class weighting can
improve recall for weaker classes while reducing precision or overall accuracy;
the validation gate prevents an unacceptable trade-off from becoming active.

FER-2013 is the primary dataset and the default general-purpose model. Webcam
fine-tuning is optional and should not be used to claim performance on other
people unless its validation set contains people kept completely separate
from training. A high score on images of one person does not demonstrate
generalization to different people.

### Optional webcam expression fine-tuning

FER-2013 validation scores do not predict how well a model will work with a
particular webcam, camera position, or lighting. To adapt the existing CNN,
first capture deliberate, visible facial-expression examples. The label is a
human-selected description of the visible expression, not a measurement of
anyone's internal emotional state. Get the participants' consent; do not
include names or other identifying information in session names.

Capture training and validation data in separate recording sessions. For
stronger validation, use different volunteers in the validation split from
those in training, and vary pose, distance, and lighting in both splits. Do
not capture a single set of frames and divide it randomly; that produces an
overly optimistic validation result. Keep the seven labels balanced. The
capture tool uses the live app's Haar detector, selects the largest face, and
saves the same grayscale face-box crop used for live model input only when
SPACE is pressed. The single-subject live detector checks two Haar cascades on
locally contrast-enhanced and shadow-brightened frames, merges duplicate
detections, and emits only the largest candidate face per frame. This is
intended for the one-person webcam use case, not group face analysis. If it
does not detect a face, face the camera directly, improve front lighting, and
move closer. If the preview is blank or shows the wrong camera, run
`--list-cameras`, then select an available index with `--camera-index`. Click
the camera window before using its keyboard controls:

```powershell
python scripts\capture_emotion_samples.py --split train --session train-day1 --samples-per-class 50
python scripts\capture_emotion_samples.py --split validation --session validation-day1 --samples-per-class 20
```

In the camera window, press `1` through `7` to select Angry, Disgust, Fear,
Happy, Sad, Surprise, or Neutral; deliberately pose that visible expression,
then press SPACE to save a single crop. Press `q` to stop. The loader requires
at least 10 training crops and 5 validation crops for every class. Use
different session names for train and validation; it rejects identical
session IDs across the splits. A different session is not the same as a
different person: a model trained and validated only on one person may learn
that person's appearance, camera, or lighting and can perform poorly on other
people. For multi-person use, include multiple consenting people in training
and validate on people who were not in training. Review/remove mislabeled or poor-quality crops
before training. Run the read-only crop audit and inspect each listed crop:

```powershell
python scripts\audit_webcam_dataset.py
```

The audit reports crops where the face detector cannot find a face; it does
not delete or change any images. These images are stored locally in
`data/webcam_emotions/`; handle them as personal data.

After capturing both splits, fine-tune and evaluate without training from
scratch:

```powershell
python scripts\train_model.py --webcam-dir data\webcam_emotions --epochs 10
```

This writes a separate candidate and reports under
`models/training/webcam/`. The existing model is replaced only if webcam validation improves by at least
the configured minimum, reaches at least 60% accuracy, reaches at least 30%
recall in every class, and FER-2013 validation accuracy does not drop by more
than 2 percentage points. The accuracy and recall gates are conservative
minimums, not guarantees of production quality. FER-2013 PrivateTest is
reported for evaluation only; it does not select the candidate. A candidate
that fails any promotion check remains available for inspection but is not
made active. Resume an interrupted webcam run with the same `--webcam-dir` and
`--resume` arguments. Webcam fine-tuning is optional; the existing active model
is not changed by capturing data or running the FER-only evaluation command.

To evaluate the currently configured checkpoint without starting training:

```powershell
python scripts\evaluate_model.py
```

This writes validation and test classification reports plus confusion matrices
to `models/evaluation/`.

Useful options:

```powershell
python scripts\train_model.py --epochs 10 --batch-size 64 --seed 42
python scripts\train_model.py --from-scratch --epochs 60 --class-weighting
python scripts\train_model.py --dataset "D:\datasets\fer2013.csv" --output "models\emotion_model.h5"
```

Training writes the epoch log, per-class precision/recall/F1, confusion matrix,
loss/accuracy plot, and summary metrics under `models/training/`. FER-2013 is
imbalanced; compare per-class metrics rather than relying on accuracy alone. A
good held-out FER score does not guarantee performance on webcam images or
across different populations and lighting.

## Live expression-only mode

The live app detects face regions and classifies facial-expression categories.
It does not load, run, or display person identification. Older known-face
enrollment images, classifier files, and helper scripts remain in the workspace
but are not used by the live detector or dashboard.

Predictions are averaged across recent frames for the same face location to
reduce flicker. When the top model score is below
`thresholds.emotion_min_confidence` in `config.yaml` (default `0.70`), the
display and event log say `Uncertain` instead of presenting a weak guess as
certain. The configured `inference.emotion_logit_adjustments` applies a
validation-selected Neutral logit offset to reduce the model's tendency to
predict Neutral. With the supplied FER-2013 split, this changed validation
accuracy/macro F1 from 81.39%/0.8147 to 82.70%/0.8282; on the untouched
PrivateTest split, it changed accuracy/macro F1 from 82.47%/0.8208 to
83.45%/0.8296. This trades away some Neutral recall for fewer false Neutral
predictions. The scores are adjusted decision scores, not calibrated
probabilities. These are FER-2013 results, not guarantees for webcam footage.
Lighting, camera placement, face size, pose, occlusion, and differences from
FER-2013 can still make live predictions inaccurate.

## Run the application

The compatible emotion checkpoint is already present. Retrain it first if you
want a fresh model, then run the backend and dashboard from separate terminals.

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (usually `http://localhost:5173`).
The dashboard opens `/video_feed`, which captures and annotates frames using
the active trained model; detections are appended to `data/logs/events.csv`
and displayed as facial-expression samples only. The API health endpoint is
`http://127.0.0.1:8000/`; recent detections are available at
`http://127.0.0.1:8000/events?limit=100`.
The video overlay shows the top expression score and the next-most-likely class
so an uncertain prediction is not presented as an unquestionable result.

Keep only one webcam client open at a time. The browser stream and standalone
OpenCV demo both use the same local camera.

If you prefer the standalone OpenCV window instead of the browser stream, stop
the backend first (only one process can use the webcam at a time), then run:

```powershell
python scripts\realtime_demo.py
```

Press `q` to exit the standalone OpenCV window.

## Notes and limitations

- The emotion CNN expects a cropped, grayscale face resized to 48x48 and
  normalized to `[0, 1]`; the live detector follows this preprocessing.
- The seven output indices follow the FER-2013 order: Angry, Disgust, Fear,
  Happy, Sad, Surprise, Neutral.
- Haar-cascade face detection can miss faces and is sensitive to pose and
  lighting. Expression predictions can be wrong and should not be used for
  consequential decisions.
- The dashboard currently uses a CSV event log; `backend/database.py` is an
  optional SQLite foundation and is not the active event store.
