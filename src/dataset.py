"""
Dataset loading, preprocessing, and twin-safe partitioning utilities.
Author: Athanasios Nikolaou
"""

import os
from typing import List, Sequence, Tuple
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(
    base_dir: str,
    subfolders: Sequence[str] = ("clear", "clouds"),
    categories: Sequence[str] = ("Healthy_augmented", "Damaged_augmented"),
    img_height: int = 64,
    img_width: int = 64,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str, str]]]:
    """
    Load and preprocess drone canopy images across flight folders and categories.

    Args:
        base_dir: Root directory of the dataset.
        subfolders: Flight environment subdirectories (e.g. ['clear', 'clouds']).
        categories: Target classification folders (['Healthy_augmented', 'Damaged_augmented']).
        img_height: Target image height for resizing.
        img_width: Target image width for resizing.

    Returns:
        Tuple of:
            - data: NumPy array of normalized RGB images (N, H, W, 3) in range [0, 1].
            - labels: NumPy array of integer class labels (N,).
            - image_paths: List of tuples (subfolder, category, filename).

    Raises:
        FileNotFoundError: If base_dir does not exist or no images are found.
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(
            f"Dataset root directory '{base_dir}' does not exist. "
            "Please check your .env file or place the dataset in the configured path."
        )

    data = []
    labels = []
    image_paths = []

    for class_num, category in enumerate(categories):
        for subfolder in subfolders:
            folder_path = os.path.join(base_dir, subfolder, category)
            if not os.path.isdir(folder_path):
                continue
            images = sorted(os.listdir(folder_path))
            for img_name in images:
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(folder_path, img_name)
                    img = tf.keras.preprocessing.image.load_img(
                        img_path, target_size=(img_height, img_width)
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    data.append(img_array)
                    labels.append(class_num)
                    image_paths.append((subfolder, category, img_name))

    if not data:
        raise FileNotFoundError(
            f"No images matching supported formats (.png, .jpg, .jpeg) were found in '{base_dir}'. "
            f"Expected subdirectories: {list(subfolders)} with categories: {list(categories)}."
        )

    data_arr = np.array(data, dtype=np.float32) / 255.0
    labels_arr = np.array(labels, dtype=np.int32)
    return data_arr, labels_arr, image_paths


def split_data(
    image_paths: List[Tuple[str, str, str]],
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Perform a twin-safe train/validation/test split based on unique image root IDs
    to prevent data leakage between augmented pairs.

    Args:
        image_paths: List of (subfolder, category, filename) tuples.
        test_size: Proportion of unique image IDs assigned to the test set (default 0.20).
        val_size: Proportion of remaining IDs assigned to the validation set (default 0.25 -> 0.20 of total).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_ids, val_ids, test_ids).
    """
    unique_image_ids = list({img_name for _, _, img_name in image_paths})
    train_ids, test_ids = train_test_split(
        unique_image_ids, test_size=test_size, random_state=random_state
    )
    train_ids, val_ids = train_test_split(
        train_ids, test_size=val_size, random_state=random_state
    )
    return train_ids, val_ids, test_ids


def get_split_indices(
    image_paths: List[Tuple[str, str, str]],
    split_ids: List[str],
) -> List[int]:
    """
    Retrieve indices in the global dataset corresponding to the assigned split IDs.

    Args:
        image_paths: Global list of image metadata tuples.
        split_ids: List of unique image IDs in the target split.

    Returns:
        List of integer indices.
    """
    split_set = set(split_ids)
    return [i for i, (_, _, img_name) in enumerate(image_paths) if img_name in split_set]


def create_datagen(seed: int = 42) -> Tuple[ImageDataGenerator, int]:
    """
    Construct the standard data augmentation generator for aerial olive canopy imagery.

    Args:
        seed: Random seed for augmentation transforms.

    Returns:
        Tuple of (ImageDataGenerator instance, seed).
    """
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    return datagen, seed
