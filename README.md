# Hand Gesture Recognition using MediaPipe and Machine Learning

This project collects hand gesture landmark data using MediaPipe, trains a machine learning model, evaluates it, and performs live webcam prediction.

## Features

- Collect gesture samples from webcam
- Save landmark data as `.npy` files
- Train a gesture classification model
- Evaluate model performance
- Predict gestures live from webcam
- Export readable `.png` and `.csv` previews from saved data

## Project Files

- `ML_Gest.py` : main project file
- `hand_landmarker.task` : MediaPipe hand landmark model
- `dataset/` : collected gesture samples
- `readable_dataset/` : exported readable previews

## Install

```bash
pip install -r requirements.txt
```

## Run

Show dataset summary:

```bash
python ML_Gest.py --mode summary
```

Collect new samples:

```bash
python ML_Gest.py --mode collect --label call_help
```

Collect samples and also save original webcam frames:

```bash
python ML_Gest.py --mode collect --label water --save-frames
```

Train the model:

```bash
python ML_Gest.py --mode train
```

Evaluate the model:

```bash
python ML_Gest.py --mode evaluate
```

Run live prediction:

```bash
python ML_Gest.py --mode predict
```

Export readable previews:

```bash
python ML_Gest.py --mode export
```

## Dataset Format

Each sample is saved as a `.npy` file with 63 values:

- 21 hand landmarks
- each landmark has `x`, `y`, `z`
- total = `21 x 3 = 63`

## Notes

- Existing old `.npy` samples do not contain original webcam images.
- If you want original images for future samples, use `--save-frames` while collecting.
