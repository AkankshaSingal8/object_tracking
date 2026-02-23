import pickle

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from tensorflow import keras

import os
import glob
from typing import Iterable, Dict
import tensorflow as tf

from tensorflow import keras
import numpy as np
from matplotlib.image import imread
from tqdm import tqdm
from PIL import Image
import pandas as pd


def get_output_normalization(root):
    training_output_mean_fn = os.path.join(root, 'stats', 'training_output_means.csv')
    if os.path.exists(training_output_mean_fn):
        print('Loading training data output means from: %s' % training_output_mean_fn)
        output_means = np.genfromtxt(training_output_mean_fn, delimiter=',')
    else:
        output_means = np.zeros(4)

    training_output_std_fn = os.path.join(root, 'stats', 'training_output_stds.csv')
    if os.path.exists(training_output_std_fn):
        print('Loading training data output std from: %s' % training_output_std_fn)
        output_stds = np.genfromtxt(training_output_std_fn, delimiter=',')
    else:
        output_stds = np.ones(4)

    return output_means, output_stds


def load_backgrounds(background_dir, image_shape_cv):
    """Load and resize all background images into a numpy array.

    Args:
        background_dir: Path to directory containing background .png/.jpg files
        image_shape_cv: (width, height) tuple for resizing

    Returns:
        np.ndarray of shape (N, H, W, 3) uint8, or None if no dir specified
    """
    if background_dir is None:
        return None

    bg_paths = sorted(
        glob.glob(os.path.join(background_dir, "*.png")) +
        glob.glob(os.path.join(background_dir, "*.jpg")) +
        glob.glob(os.path.join(background_dir, "*.jpeg"))
    )

    if len(bg_paths) == 0:
        raise FileNotFoundError(f"No image files found in {background_dir}")

    print(f"Loading {len(bg_paths)} background images from {background_dir}...")

    # image_shape_cv is (W, H), we need array of (H, W, 3)
    h, w = image_shape_cv[1], image_shape_cv[0]
    backgrounds = np.empty((len(bg_paths), h, w, 3), dtype=np.uint8)

    for i, p in enumerate(tqdm(bg_paths, desc="Loading backgrounds")):
        img = Image.open(p).convert('RGB').resize(image_shape_cv)
        backgrounds[i] = np.array(img)

    print(f"Loaded {len(backgrounds)} backgrounds, shape: {backgrounds.shape}")
    return backgrounds


@tf.function
def apply_background_sequence_tf(rgba_sequence, label_sequence, backgrounds):
    """Apply random background + per-frame color jitter to a windowed sequence.

    1. Pick ONE random background per sequence (consistent ground texture)
    2. Composite RGBA onto background using alpha blending
    3. Apply random brightness/contrast jitter PER FRAME to simulate
       lighting variation during a trajectory

    Args:
        rgba_sequence: (seq_len, H, W, 4) uint8
        label_sequence: (seq_len, 4) float32
        backgrounds: (N, H, W, 3) uint8

    Returns:
        rgb_sequence: (seq_len, H, W, 3) uint8
        label_sequence: passthrough
    """
    # Pick ONE random background per sequence
    n_bg = tf.shape(backgrounds)[0]
    idx = tf.random.uniform((), minval=0, maxval=n_bg, dtype=tf.int32)
    bg = tf.cast(backgrounds[idx], tf.float32)  # (H, W, 3)

    # Alpha compositing (vectorized over seq dim)
    rgba_float_rgb = tf.cast(rgba_sequence[:, :, :, :3], tf.float32)  # (seq, H, W, 3)
    alpha = tf.cast(rgba_sequence[:, :, :, 3:4], tf.float32) / 255.0  # (seq, H, W, 1)

    bg_expanded = tf.expand_dims(bg, 0)  # (1, H, W, 3)
    composite = rgba_float_rgb * alpha + bg_expanded * (1.0 - alpha)

    # ── Per-frame color jitter ──
    # Random brightness shift per frame: uniform in [-delta, +delta]
    seq_len = tf.shape(composite)[0]
    brightness_delta = 25.0  # max pixel shift
    brightness_offsets = tf.random.uniform(
        [seq_len, 1, 1, 1], minval=-brightness_delta, maxval=brightness_delta
    )
    composite = composite + brightness_offsets

    # Random contrast per frame: multiply by factor in [0.8, 1.2]
    contrast_factors = tf.random.uniform(
        [seq_len, 1, 1, 1], minval=0.8, maxval=1.2
    )
    mean_per_frame = tf.reduce_mean(composite, axis=[1, 2, 3], keepdims=True)
    composite = (composite - mean_per_frame) * contrast_factors + mean_per_frame

    composite = tf.clip_by_value(composite, 0, 255)
    rgb_sequence = tf.cast(composite, tf.uint8)

    return rgb_sequence, label_sequence


@tf.function
def apply_background_sequence_val_tf(rgba_sequence, label_sequence, backgrounds):
    """Apply a DETERMINISTIC background for validation (no jitter).

    Uses the sequence index modulo n_backgrounds to pick a consistent
    background. No color jitter applied.

    Args:
        rgba_sequence: (seq_len, H, W, 4) uint8
        label_sequence: (seq_len, 4) float32
        backgrounds: (N, H, W, 3) uint8

    Returns:
        rgb_sequence: (seq_len, H, W, 3) uint8
        label_sequence: passthrough
    """
    n_bg = tf.shape(backgrounds)[0]
    # Use a random but fixed-per-call index (deterministic within epoch if seeded)
    idx = tf.random.uniform((), minval=0, maxval=n_bg, dtype=tf.int32)
    bg = tf.cast(backgrounds[idx], tf.float32)

    rgba_float_rgb = tf.cast(rgba_sequence[:, :, :, :3], tf.float32)
    alpha = tf.cast(rgba_sequence[:, :, :, 3:4], tf.float32) / 255.0

    bg_expanded = tf.expand_dims(bg, 0)
    composite = rgba_float_rgb * alpha + bg_expanded * (1.0 - alpha)
    composite = tf.clip_by_value(composite, 0, 255)
    rgb_sequence = tf.cast(composite, tf.uint8)

    return rgb_sequence, label_sequence


@tf.function
def strip_alpha_sequence_tf(rgba_sequence, label_sequence):
    """Strip alpha channel from RGBA sequence, keeping RGB on black background."""
    return rgba_sequence[:, :, :, :3], label_sequence


def load_dataset_multi(root, image_size, seq_len, shift, stride, label_scale,
                       background_dir=None, backgrounds_np=None):
    """Load dataset as RGBA (if backgrounds are used) or RGB.

    NOTE: Background compositing is NOT applied here. It is applied
    selectively in get_dataset_multi() — only on training data.
    """
    file_ending = 'png'
    IMAGE_SHAPE = (144, 256, 3)
    IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

    use_backgrounds = (background_dir is not None) or (backgrounds_np is not None)
    n_channels = 4 if use_backgrounds else 3

    def sub_to_batch(sub_feature, sub_label):
        sfb = sub_feature.batch(seq_len, drop_remainder=True)
        slb = sub_label.batch(seq_len, drop_remainder=True)
        return tf.data.Dataset.zip((sfb, slb))

    datasets = []

    for i in range(len(os.listdir(root))):
        directory = i + 1
        csv_file_name = f"{root}/{str(directory)}/data_out.csv"
        labels = np.genfromtxt(csv_file_name, delimiter=',', skip_header=1, dtype=np.float32)
        print("labels", labels)

        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        n_images = len([fn for fn in os.listdir(f"./{root}/{str(directory)}") if file_ending in fn])
        print(n_images)
        print("no of imgs", n_images)

        load_shape = (IMAGE_SHAPE[0], IMAGE_SHAPE[1], n_channels)
        dataset_np = np.empty((n_images, *load_shape), dtype=np.uint8)

        for ix in range(n_images):
            img_file_name = root + "/" + str(directory) + '/Image' + str(ix + 1) + '.' + file_ending
            if use_backgrounds:
                img = Image.open(img_file_name).convert('RGBA').resize(IMAGE_SHAPE_CV)
            else:
                img = Image.open(img_file_name).convert('RGB').resize(IMAGE_SHAPE_CV)
            dataset_np[ix] = np.array(img)

        images_dataset = tf.data.Dataset.from_tensor_slices(dataset_np)
        dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))
        dataset = dataset.window(seq_len, shift=shift, stride=stride, drop_remainder=True).flat_map(sub_to_batch)

        datasets.append(dataset)

    return datasets


def get_dataset_multi(root, image_size, seq_len, shift, stride, validation_ratio, label_scale,
                      extra_data_root=None, background_dir=None, val_bg_mode="black"):
    """Load and split dataset into training and validation.

    Background compositing is applied to training data with color jitter.
    Validation data can be:
      - "black"     : strip alpha, black background (original behavior)
      - "background": composite with backgrounds but NO color jitter
                      (measures actual background robustness)
      - "both"      : returns black-bg val AND bg-val as separate datasets

    Args:
        root: Dataset root directory
        image_size: (H, W, C)
        seq_len: Sequence length
        shift, stride: Windowing params
        validation_ratio: Fraction for validation
        label_scale: Label scaling
        extra_data_root: Additional data (unused)
        background_dir: Path to background images
        val_bg_mode: "black" | "background" | "both"

    Returns:
        If val_bg_mode != "both": (training_dataset, validation_dataset)
        If val_bg_mode == "both": (training_dataset, val_black, val_with_bg)
    """
    IMAGE_SHAPE_CV = (256, 144)  # (W, H)
    use_backgrounds = background_dir is not None

    ds = load_dataset_multi(root, image_size, seq_len, shift, stride, label_scale,
                            background_dir=background_dir)
    print('n bags: %d' % len(ds))
    cnt = 0
    for d in ds:
        for (ix, _) in enumerate(d):
            pass
            cnt += ix
    print('n windows: %d' % cnt)

    # ── Split into train / val trajectories ──
    val_ix = int(len(ds) * validation_ratio)
    print('\nval_ix: %d\n' % val_ix)
    validation_datasets = ds[:val_ix]
    training_datasets = ds[val_ix:]

    training = tf.data.Dataset.from_tensor_slices(training_datasets).flat_map(lambda x: x)
    validation = tf.data.Dataset.from_tensor_slices(validation_datasets).flat_map(lambda x: x)

    # ── Apply backgrounds ──
    if use_backgrounds:
        backgrounds_np = load_backgrounds(background_dir, IMAGE_SHAPE_CV)
        backgrounds_tf = tf.constant(backgrounds_np, dtype=tf.uint8)
        print(f"Background tensor shape: {backgrounds_tf.shape}")

        # Training: RGBA + random background + color jitter → RGB
        training = training.map(
            lambda imgs, lbls: apply_background_sequence_tf(imgs, lbls, backgrounds_tf),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if val_bg_mode == "black":
            # Validation: RGBA → RGB (drop alpha, black background)
            validation = validation.map(
                strip_alpha_sequence_tf,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            print("Training: random bg + jitter | Validation: black bg")

        elif val_bg_mode == "background":
            # Validation: RGBA + random background, NO jitter
            validation = validation.map(
                lambda imgs, lbls: apply_background_sequence_val_tf(imgs, lbls, backgrounds_tf),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            print("Training: random bg + jitter | Validation: random bg (no jitter)")

        elif val_bg_mode == "both":
            # Return both validation sets
            val_black = validation.map(
                strip_alpha_sequence_tf,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            val_with_bg = validation.map(
                lambda imgs, lbls: apply_background_sequence_val_tf(imgs, lbls, backgrounds_tf),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            print("Training: random bg + jitter | Validation: black + bg")
            return training, val_black, val_with_bg

    return training, validation


def load_val_dataset_multi(root, image_size, seq_len, shift, stride, label_scale,
                           background_dir=None, backgrounds_np=None):
    """Load validation dataset. No background compositing — always black backgrounds."""
    file_ending = 'png'
    IMAGE_SHAPE = (144, 256, 3)
    IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

    def sub_to_batch(sub_feature, sub_label):
        sfb = sub_feature.batch(seq_len, drop_remainder=True)
        slb = sub_label.batch(seq_len, drop_remainder=True)
        return tf.data.Dataset.zip((sfb, slb))

    datasets = []

    for i in range(len(os.listdir(root))):
        directory = i + 1
        csv_file_name = f"{root}/{str(directory)}/data_out.csv"
        labels = np.genfromtxt(csv_file_name, delimiter=',', skip_header=1, dtype=np.float32)
        print("labels", labels)

        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        base_name = f"{root}/{directory}"
        n_images = len([fn for fn in os.listdir(base_name) if file_ending in fn])
        print(n_images)
        print("no of imgs", n_images)

        dataset_np = np.empty((n_images, *IMAGE_SHAPE), dtype=np.uint8)

        for ix in range(n_images):
            img_file_name = root + "/" + str(directory) + '/Image' + str(ix + 1) + '.' + file_ending
            img = Image.open(img_file_name).convert('RGB').resize(IMAGE_SHAPE_CV)
            dataset_np[ix] = np.array(img)

        images_dataset = tf.data.Dataset.from_tensor_slices(dataset_np)
        dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))
        dataset = dataset.window(seq_len, shift=shift, stride=stride, drop_remainder=True).flat_map(sub_to_batch)

        datasets.append(dataset)

    return datasets


def get_val_dataset_multi(root, image_size, seq_len, shift, stride, validation_ratio, label_scale,
                          extra_data_root=None, background_dir=None):
    """Load validation-only dataset. Always uses black backgrounds (no compositing)."""
    ds = load_val_dataset_multi(root, image_size, seq_len, shift, stride, label_scale)
    print('n bags: %d' % len(ds))
    cnt = 0
    for d in ds:
        for (ix, _) in enumerate(d):
            pass
            cnt += ix
    print('n windows: %d' % cnt)

    val_ix = 0
    training_datasets = ds[val_ix:]
    training = tf.data.Dataset.from_tensor_slices(training_datasets).flat_map(lambda x: x)
    return training