
"""
Dataset utilities for the LEAPGestRecog hand gesture recognition project.

Responsibilities:
- Detect the correct dataset root, including nested folder layouts.
- Discover images and assign gesture labels from folder names.
- Split the data by subject to reduce data leakage.
- Preprocess images for CNN training and prediction.
"""

from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps


# --------------------------------------------------
# Supported image formats
# --------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# --------------------------------------------------
# Gesture classes
# --------------------------------------------------

# Keep this order fixed. It defines the numeric label of each class.
GESTURES = {
    "01_palm": "Palm",
    "02_l": "L",
    "03_fist": "Fist",
    "04_fist_moved": "Moving Fist",
    "05_thumb": "Thumb",
    "06_index": "Index",
    "07_ok": "OK",
    "08_palm_moved": "Moving Palm",
    "09_c": "C",
    "10_down": "Down",
}

CLASS_FOLDERS = list(GESTURES.keys())
CLASS_NAMES = list(GESTURES.values())

CLASS_TO_INDEX = {
    folder: index
    for index, folder in enumerate(CLASS_FOLDERS)
}


# --------------------------------------------------
# Subject and folder helpers
# --------------------------------------------------

def _is_subject_folder(path):
    """Return True for numbered subject directories such as 00 or 09."""
    path = Path(path)
    return (
        path.is_dir()
        and re.fullmatch(r"\d{2}", path.name) is not None
    )


def _numbered_subject_folders(path):
    """Return numbered subject folders immediately inside a directory."""
    path = Path(path)

    if not path.is_dir():
        return []

    try:
        return sorted(
            child
            for child in path.iterdir()
            if _is_subject_folder(child)
        )
    except PermissionError:
        return []


def _count_gesture_images(subject_folders):
    """Count images only inside the recognized gesture folders."""
    total = 0

    for subject in subject_folders:
        for gesture_folder in CLASS_FOLDERS:
            gesture_path = subject / gesture_folder

            if not gesture_path.is_dir():
                continue

            try:
                for image_path in gesture_path.rglob("*"):
                    if (
                        image_path.is_file()
                        and image_path.suffix.lower()
                        in IMAGE_EXTENSIONS
                    ):
                        total += 1

            except PermissionError:
                continue

    return total


# --------------------------------------------------
# Dataset root detection
# --------------------------------------------------

def find_dataset_root(data_dir):
    """
    Find the directory whose immediate children are subject folders.

    Supports structures such as:

        data/
        └── leapGestRecog/
            ├── 00/
            ├── 01/
            └── ...

    and:

        data/
        └── leapGestRecog/
            └── leapGestRecog/
                ├── 00/
                ├── 01/
                └── ...

    If more than one valid root exists, the directory with the most
    recognized gesture images is preferred. Ties prefer the shallower
    directory to avoid accidentally selecting a nested duplicate.
    """
    data_dir = Path(data_dir).expanduser().resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Check the location of your extracted LEAPGestRecog dataset."
        )

    # Search the data directory and its nearby nested directories.
    # Limiting the search depth avoids walking through every image merely
    # to discover potential directory roots.
    candidates = [data_dir]
    current_level = [data_dir]
    max_depth = 4

    for _ in range(max_depth):
        next_level = []

        for parent in current_level:
            try:
                children = [
                    child
                    for child in parent.iterdir()
                    if child.is_dir()
                ]
            except PermissionError:
                continue

            candidates.extend(children)
            next_level.extend(children)

        current_level = next_level

        if not current_level:
            break

    # Remove duplicate paths while preserving discovery order.
    unique_candidates = []
    seen = set()

    for candidate in candidates:
        key = str(candidate.resolve())

        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    valid_roots = []

    for candidate in unique_candidates:
        subjects = _numbered_subject_folders(candidate)

        # A dataset needs at least three subjects for this project's
        # train/validation/test subject split.
        if len(subjects) < 3:
            continue

        recognized_gesture_folders = 0

        for subject in subjects:
            try:
                child_names = {
                    child.name
                    for child in subject.iterdir()
                    if child.is_dir()
                }
            except PermissionError:
                continue

            recognized_gesture_folders += len(
                child_names.intersection(CLASS_FOLDERS)
            )

        if recognized_gesture_folders == 0:
            continue

        image_count = _count_gesture_images(subjects)

        if image_count == 0:
            continue

        valid_roots.append(
            (
                image_count,
                -len(candidate.parts),
                candidate,
            )
        )

    if not valid_roots:
        raise FileNotFoundError(
            "Could not identify the LEAPGestRecog dataset root.\n\n"
            "Expected subject folders such as 00, 01, ..., 09, each "
            "containing gesture folders such as 01_palm and 10_down.\n"
            f"Search started from: {data_dir}"
        )

    # Highest image count first; for ties, prefer the shallower path.
    valid_roots.sort(
        key=lambda entry: (entry[0], entry[1]),
        reverse=True,
    )

    image_count, _, detected_root = valid_roots[0]

    print(f"Dataset root detected: {detected_root}")
    print(f"Gesture images found: {image_count:,}")

    return detected_root


# --------------------------------------------------
# Image and label discovery
# --------------------------------------------------

def collect_records(dataset_root):
    """
    Collect labeled image records from the dataset.

    Each record has the form:
        (image_path, class_id, subject_id)

    The label comes from the gesture folder name rather than the
    image filename. This matches the LEAPGestRecog directory structure.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}"
        )

    subjects = _numbered_subject_folders(dataset_root)

    if not subjects:
        raise ValueError(
            f"No numbered subject folders were found in {dataset_root}."
        )

    records = []
    class_counts = {name: 0 for name in CLASS_NAMES}

    for subject_folder in subjects:
        subject_id = subject_folder.name

        for gesture_folder in CLASS_FOLDERS:
            gesture_path = subject_folder / gesture_folder

            if not gesture_path.is_dir():
                continue

            class_id = CLASS_TO_INDEX[gesture_folder]
            class_name = GESTURES[gesture_folder]

            try:
                image_paths = sorted(
                    image_path
                    for image_path in gesture_path.rglob("*")
                    if (
                        image_path.is_file()
                        and image_path.suffix.lower()
                        in IMAGE_EXTENSIONS
                    )
                )
            except PermissionError:
                print(
                    f"Warning: Permission denied while reading "
                    f"{gesture_path}"
                )
                continue

            for image_path in image_paths:
                records.append(
                    (
                        str(image_path.resolve()),
                        class_id,
                        subject_id,
                    )
                )

                class_counts[class_name] += 1

    if not records:
        raise ValueError(
            f"No recognized gesture images were found under "
            f"{dataset_root}."
        )

    records.sort(
        key=lambda record: (
            record[2],
            record[1],
            record[0],
        )
    )

    print("\nDataset summary:")

    for class_name, count in class_counts.items():
        print(f"  {class_name:<15} {count:>7,}")

    print(f"  {'TOTAL':<15} {len(records):>7,}")

    return records


# --------------------------------------------------
# Subject-based splitting
# --------------------------------------------------

def split_by_subject(records):
    """
    Split records by subject rather than randomly by individual image.

    For ten subjects 00 through 09:
        Training:   00 through 07
        Validation: 08
        Testing:    09

    For a different number of subjects:
        All but the final two sorted subjects -> training
        Second-last subject                  -> validation
        Last subject                         -> test

    Returns:
        train_records,
        validation_records,
        test_records,
        split_information
    """
    if not records:
        raise ValueError(
            "Cannot split an empty dataset."
        )

    subjects = sorted(
        {record[2] for record in records}
    )

    if len(subjects) < 3:
        raise ValueError(
            "At least three distinct subject folders are required "
            "for a subject-based train/validation/test split."
        )

    training_subjects = set(subjects[:-2])
    validation_subject = subjects[-2]
    test_subject = subjects[-1]

    training_records = [
        record
        for record in records
        if record[2] in training_subjects
    ]

    validation_records = [
        record
        for record in records
        if record[2] == validation_subject
    ]

    test_records = [
        record
        for record in records
        if record[2] == test_subject
    ]

    if not training_records:
        raise ValueError("The training split is empty.")

    if not validation_records:
        raise ValueError("The validation split is empty.")

    if not test_records:
        raise ValueError("The test split is empty.")

    split_information = {
        "train_subjects": sorted(training_subjects),
        "validation_subject": validation_subject,
        "test_subject": test_subject,
    }

    return (
        training_records,
        validation_records,
        test_records,
        split_information,
    )


# --------------------------------------------------
# Image preprocessing for prediction
# --------------------------------------------------

def preprocess_image(image, size=(96, 96)):
    """
    Preprocess a PIL image using the same format as CNN training.

    Args:
        image: A PIL Image object.
        size: (height, width), normally (96, 96).

    Returns:
        NumPy array of shape (height, width, 1), float32,
        with pixel values normalized to [0, 1].
    """
    if len(size) != 2:
        raise ValueError(
            "size must be a (height, width) tuple."
        )

    height, width = int(size[0]), int(size[1])

    if height < 1 or width < 1:
        raise ValueError(
            "Image dimensions must be positive."
        )

    image = ImageOps.exif_transpose(image)
    image = image.convert("L")

    image = image.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    array = array / 255.0

    # Add the grayscale channel dimension.
    array = array[..., np.newaxis]

    return array