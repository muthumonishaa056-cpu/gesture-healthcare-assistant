"""
Hand Gesture Dataset Collection and Classification using MediaPipe

This file does everything in one place:
1. Collect gesture samples from webcam
2. Train a machine learning model
3. Evaluate the trained model
4. Predict gestures live from webcam
5. Export readable preview files from saved .npy data

Dataset format:
- dataset/
    - call_help/
        - 0.npy
        - 1.npy
    - medicine/
    - emergency/
    - water/

Each .npy file stores 63 values:
21 hand landmarks x 3 coordinates (x, y, z)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------
# PATHS AND CONSTANTS
# -------------------------------------------------------------------

DATASET_DIR = Path("dataset")
MODEL_PATH = Path("gesture_model.joblib")
METADATA_PATH = Path("gesture_model_meta.json")
TASK_MODEL_PATH = Path("hand_landmarker.task")
EXPORT_DIR = Path("readable_dataset")

FEATURE_COUNT = 63
COLLECT_WINDOW = "Gesture Dataset Collector"
PREDICT_WINDOW = "Gesture Live Prediction"


# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def clean_label(label: str) -> str:
    """Convert label into a safe folder name."""
    return label.strip().lower().replace(" ", "_")


def get_hand_connections() -> list[tuple[int, int]]:
    """Return MediaPipe hand connections."""
    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    return [(item.start, item.end) for item in connections]


def open_camera(camera_index: int = 0) -> cv2.VideoCapture | None:
    """Open webcam using available OpenCV backends."""
    for backend in (cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF):
        camera = cv2.VideoCapture(camera_index, backend)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            camera.set(cv2.CAP_PROP_FPS, 30)
            return camera
        camera.release()
    return None


def count_samples(folder: Path) -> int:
    """Count .npy samples inside one gesture folder."""
    return len(list(folder.glob("*.npy")))


def draw_landmarks(frame: np.ndarray, result) -> None:
    """Draw detected hand landmarks on frame."""
    if result is None or not getattr(result, "hand_landmarks", None):
        return

    landmarks = result.hand_landmarks[0]
    for start_index, end_index in get_hand_connections():
        start = landmarks[start_index]
        end = landmarks[end_index]
        p1 = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
        p2 = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
        cv2.line(frame, p1, p2, (180, 180, 180), 2, cv2.LINE_AA)

    for landmark in landmarks:
        point = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
        cv2.circle(frame, point, 4, (0, 200, 255), -1, cv2.LINE_AA)


def create_model() -> Pipeline:
    """Create the ML pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    batch_size=16,
                    learning_rate_init=0.001,
                    max_iter=600,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=42,
                ),
            ),
        ]
    )


# -------------------------------------------------------------------
# MEDIAPIPE HAND EXTRACTOR
# -------------------------------------------------------------------

class HandLandmarkExtractor:
    """Extract 63 landmark values from webcam frames."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model file not found: {model_path.resolve()}"
            )

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve()))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def extract(self, frame: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray | None, object | None]:
        """Return flattened landmark data and raw MediaPipe result."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None, result

        values: list[float] = []
        for landmark in result.hand_landmarks[0]:
            values.extend([landmark.x, landmark.y, landmark.z])

        return np.asarray(values, dtype=np.float32), result

    def close(self) -> None:
        self.detector.close()


# -------------------------------------------------------------------
# DATA COLLECTION
# -------------------------------------------------------------------

def collect_data(label: str, camera_index: int = 0, save_frames: bool = False) -> None:
    """Collect gesture samples and save them as .npy files."""
    label = clean_label(label)
    if not label:
        raise ValueError("Gesture label cannot be empty.")

    class_folder = DATASET_DIR / label
    class_folder.mkdir(parents=True, exist_ok=True)
    frame_folder = class_folder / "frames"
    if save_frames:
        frame_folder.mkdir(parents=True, exist_ok=True)

    sample_number = count_samples(class_folder)
    extractor = HandLandmarkExtractor(TASK_MODEL_PATH)
    camera = open_camera(camera_index)
    if camera is None:
        extractor.close()
        raise RuntimeError("Unable to open webcam.")

    cv2.namedWindow(COLLECT_WINDOW, cv2.WINDOW_NORMAL)
    print(f"Collecting samples for: {label}")
    print("Press 's' to save sample")
    print("Press 'q' to quit")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            timestamp_ms = int(time.monotonic() * 1000)
            features, result = extractor.extract(frame, timestamp_ms)
            draw_landmarks(frame, result)

            cv2.putText(frame, f"Label: {label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Saved: {sample_number}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, "S = Save   Q = Quit", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            if features is None:
                message = "Show one hand clearly"
                color = (0, 120, 255)
            else:
                message = "Hand detected"
                color = (0, 255, 255)

            cv2.putText(frame, message, (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.imshow(COLLECT_WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("s"):
                if features is None:
                    print("No hand detected. Sample not saved.")
                    continue

                np.save(class_folder / f"{sample_number}.npy", features)
                if save_frames:
                    cv2.imwrite(str(frame_folder / f"{sample_number}.jpg"), frame)

                sample_number += 1
                print(f"Saved sample {sample_number}")
    finally:
        camera.release()
        extractor.close()
        cv2.destroyAllWindows()


# -------------------------------------------------------------------
# DATASET LOADING
# -------------------------------------------------------------------

def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all .npy samples from dataset folder."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR.resolve()}")

    x_data: list[np.ndarray] = []
    y_data: list[int] = []
    class_names: list[str] = []

    class_folders = sorted(folder for folder in DATASET_DIR.iterdir() if folder.is_dir())
    for folder in class_folders:
        files = sorted(folder.glob("*.npy"))
        if not files:
            continue

        class_index = len(class_names)
        class_names.append(folder.name)

        for file_path in files:
            sample = np.load(file_path).astype(np.float32).reshape(-1)
            if sample.size != FEATURE_COUNT:
                raise ValueError(
                    f"Invalid sample size in {file_path}. Expected {FEATURE_COUNT}, got {sample.size}."
                )
            x_data.append(sample)
            y_data.append(class_index)

    if not x_data:
        raise ValueError("No dataset samples found.")

    if len(class_names) < 2:
        raise ValueError("At least two gesture classes are required for training.")

    return (
        np.asarray(x_data, dtype=np.float32),
        np.asarray(y_data, dtype=np.int32),
        class_names,
    )


def split_dataset(
    x_data: np.ndarray, y_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train, validation, and test sets."""
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_data,
        y_data,
        test_size=0.30,
        random_state=42,
        stratify=y_data,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


# -------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------

def train_model() -> None:
    """Train model and save files."""
    x_data, y_data, class_names = load_dataset()
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(x_data, y_data)

    model = create_model()
    model.fit(x_train, y_train)

    val_predictions = model.predict(x_val)
    test_predictions = model.predict(x_test)

    val_accuracy = accuracy_score(y_val, val_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    report = classification_report(
        y_test,
        test_predictions,
        target_names=class_names,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, test_predictions)

    joblib.dump(model, MODEL_PATH)
    metadata = {
        "class_names": class_names,
        "feature_count": FEATURE_COUNT,
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "validation_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nTraining completed successfully")
    print(f"Training samples   : {len(x_train)}")
    print(f"Validation samples : {len(x_val)}")
    print(f"Testing samples    : {len(x_test)}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Test accuracy      : {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)
    print(f"\nModel saved at: {MODEL_PATH.resolve()}")
    print(f"Metadata saved at: {METADATA_PATH.resolve()}")


# -------------------------------------------------------------------
# EVALUATION
# -------------------------------------------------------------------

def evaluate_model() -> None:
    """Evaluate already trained model."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Model files not found. Run training first.")

    x_data, y_data, _ = load_dataset()
    _, _, x_test, _, _, y_test = split_dataset(x_data, y_data)

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    class_names = metadata["class_names"]

    predictions = model.predict(x_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=class_names, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# -------------------------------------------------------------------
# LIVE PREDICTION
# -------------------------------------------------------------------

def predict_live(camera_index: int = 0) -> None:
    """Use trained model to predict gestures live from webcam."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Model files not found. Run training first.")

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    class_names = metadata["class_names"]

    extractor = HandLandmarkExtractor(TASK_MODEL_PATH)
    camera = open_camera(camera_index)
    if camera is None:
        extractor.close()
        raise RuntimeError("Unable to open webcam.")

    cv2.namedWindow(PREDICT_WINDOW, cv2.WINDOW_NORMAL)
    print("Live prediction started. Press 'q' to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            timestamp_ms = int(time.monotonic() * 1000)
            features, result = extractor.extract(frame, timestamp_ms)
            draw_landmarks(frame, result)

            text = "No hand detected"
            if features is not None:
                probabilities = model.predict_proba(features.reshape(1, -1))[0]
                best_index = int(np.argmax(probabilities))
                confidence = probabilities[best_index] * 100
                text = f"{class_names[best_index]} ({confidence:.1f}%)"

            cv2.putText(frame, f"Prediction: {text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, "Press Q to quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.imshow(PREDICT_WINDOW, frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        camera.release()
        extractor.close()
        cv2.destroyAllWindows()


# -------------------------------------------------------------------
# EXPORT READABLE FILES
# -------------------------------------------------------------------

def create_landmark_preview(sample: np.ndarray, image_size: int = 512) -> np.ndarray:
    """Create a readable landmark image from saved sample values."""
    coords = sample.reshape(21, 3)
    canvas = np.full((image_size, image_size, 3), 250, dtype=np.uint8)

    margin = int(image_size * 0.10)
    usable_area = image_size - (2 * margin)

    points: list[tuple[int, int]] = []
    for x, y, _ in coords:
        px = int(margin + np.clip(x, 0.0, 1.0) * usable_area)
        py = int(margin + np.clip(y, 0.0, 1.0) * usable_area)
        points.append((px, py))

    for start_index, end_index in get_hand_connections():
        cv2.line(canvas, points[start_index], points[end_index], (70, 70, 70), 2, cv2.LINE_AA)

    for point_index, point in enumerate(points):
        color = (0, 180, 255) if point_index in {4, 8, 12, 16, 20} else (0, 130, 220)
        cv2.circle(canvas, point, 6, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, point, 8, (30, 30, 30), 1, cv2.LINE_AA)

    return canvas


def export_readable_dataset() -> None:
    """Convert saved .npy files into .png previews and .csv tables."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR.resolve()}")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    index_lines = ["label,sample_file,preview_image,csv_file"]
    exported_count = 0

    for folder in sorted(class_folder for class_folder in DATASET_DIR.iterdir() if class_folder.is_dir()):
        sample_files = sorted(folder.glob("*.npy"))
        if not sample_files:
            continue

        export_folder = EXPORT_DIR / folder.name
        export_folder.mkdir(parents=True, exist_ok=True)

        for sample_file in sample_files:
            sample = np.load(sample_file).astype(np.float32).reshape(-1)
            if sample.size != FEATURE_COUNT:
                print(f"Skipping invalid file: {sample_file}")
                continue

            file_name = sample_file.stem
            coords = sample.reshape(21, 3)
            preview_path = export_folder / f"{file_name}.png"
            csv_path = export_folder / f"{file_name}.csv"

            preview = create_landmark_preview(sample)
            cv2.imwrite(str(preview_path), preview)

            rows = np.column_stack((np.arange(21, dtype=np.int32), coords))
            np.savetxt(
                csv_path,
                rows,
                delimiter=",",
                header="landmark_id,x,y,z",
                comments="",
                fmt=["%d", "%.6f", "%.6f", "%.6f"],
            )

            index_lines.append(
                f"{folder.name},{sample_file.as_posix()},{preview_path.as_posix()},{csv_path.as_posix()}"
            )
            exported_count += 1

    index_file = EXPORT_DIR / "dataset_index.csv"
    index_file.write_text("\n".join(index_lines), encoding="utf-8")

    print(f"Exported {exported_count} files")
    print(f"Readable dataset folder: {EXPORT_DIR.resolve()}")
    print(f"Index file: {index_file.resolve()}")


# -------------------------------------------------------------------
# DATASET SUMMARY
# -------------------------------------------------------------------

def show_dataset_summary() -> None:
    """Print gesture-wise sample counts."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR.resolve()}")

    class_folders = sorted(folder for folder in DATASET_DIR.iterdir() if folder.is_dir())
    if not class_folders:
        print("No class folders found.")
        return

    total = 0
    print("Dataset Summary:")
    for folder in class_folders:
        count = count_samples(folder)
        total += count
        print(f"- {folder.name}: {count} samples")
    print(f"Total samples: {total}")


# -------------------------------------------------------------------
# MENU AND ARGUMENTS
# -------------------------------------------------------------------

def choose_mode_interactively() -> tuple[str, str]:
    """Show menu when mode is not given as argument."""
    print("\nGesture Dataset Collection and Classification")
    print("1. Collect gesture samples")
    print("2. Train model")
    print("3. Evaluate model")
    print("4. Live prediction")
    print("5. Export readable dataset")
    print("6. Show dataset summary")
    print("7. Exit")

    choice = input("Select an option (1-7): ").strip()
    mode_map = {
        "1": "collect",
        "2": "train",
        "3": "evaluate",
        "4": "predict",
        "5": "export",
        "6": "summary",
        "7": "exit",
    }
    mode = mode_map.get(choice, "")

    if mode == "collect":
        label = input("Enter gesture label: ").strip()
        return mode, label

    return mode, ""


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Gesture dataset collection, training, evaluation, and live prediction"
    )
    parser.add_argument(
        "--mode",
        choices=["collect", "train", "evaluate", "predict", "export", "summary"],
        help="Select what the file should do",
    )
    parser.add_argument(
        "--label",
        help="Gesture label for collect mode, for example: call_help",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="While collecting, also save original webcam images as jpg",
    )
    return parser.parse_args()


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main() -> int:
    args = parse_arguments()
    mode = args.mode
    label = args.label or ""

    if mode is None:
        mode, label = choose_mode_interactively()

    try:
        if mode == "collect":
            if not label:
                raise ValueError("Please provide a label for collect mode.")
            collect_data(label, args.camera_index, args.save_frames)
        elif mode == "train":
            train_model()
        elif mode == "evaluate":
            evaluate_model()
        elif mode == "predict":
            predict_live(args.camera_index)
        elif mode == "export":
            export_readable_dataset()
        elif mode == "summary":
            show_dataset_summary()
        elif mode == "exit":
            print("Exited.")
        else:
            raise ValueError("Invalid mode selected.")
        return 0

    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 1
    except Exception as error:
        print(f"\nError: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
