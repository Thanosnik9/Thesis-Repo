"""
Neural network architecture definitions for Verticillium wilt diagnosis.
Author: Athanasios Nikolaou
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    Add,
    SeparableConv2D,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50V2


def residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    use_batchnorm: bool = True,
) -> tf.Tensor:
    """
    Residual block utilizing Depthwise Separable Convolutions and identity mapping.

    Args:
        x: Input tensor.
        filters: Number of convolutional feature filters.
        kernel_size: Spatial kernel dimension (e.g. 3 or 5).
        use_batchnorm: Whether to apply BatchNormalization layers.

    Returns:
        Output tensor after residual addition and ReLU activation.
    """
    shortcut = x
    x = SeparableConv2D(filters, (kernel_size, kernel_size), padding="same")(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = SeparableConv2D(filters, (kernel_size, kernel_size), padding="same")(x)
    if use_batchnorm:
        x = BatchNormalization()(x)

    # 1x1 projection shortcut if channel dimensions differ
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same")(shortcut)
        if use_batchnorm:
            shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


def attention_block(x: tf.Tensor, filters: int, num_heads: int = 8) -> tf.Tensor:
    """
    Multi-Head Self-Attention block with residual skip connection.

    Args:
        x: Input feature map tensor.
        filters: Key dimension / projection size for attention heads.
        num_heads: Number of parallel attention heads.

    Returns:
        Attention-enhanced feature tensor.
    """
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=filters)(x, x)
    return Add()([x, attention])


def build_custom_model(
    img_height: int = 64,
    img_width: int = 64,
    num_residual_blocks: int = 4,
    filters: int = 121,
    kernel_size: int = 5,
    dense_units: int = 1387,
    dropout_rate: float = 0.408,
    learning_rate: float = 2.47e-4,
    num_classes: int = 2,
    use_batchnorm: bool = True,
    use_attention: bool = True,
) -> Model:
    """
    Construct the optimal Attention-Residual Deep CNN architecture.

    Default hyperparameters match the best trial found in the 300-run Optuna study.

    Args:
        img_height: Input image height.
        img_width: Input image width.
        num_residual_blocks: Number of sequential residual-attention stages.
        filters: Number of feature maps.
        kernel_size: Convolution kernel dimension.
        dense_units: Hidden units in the dense classification layer.
        dropout_rate: Dropout regularization rate.
        learning_rate: Initial Adam learning rate.
        num_classes: Number of diagnostic target classes (2: Healthy, Damaged).
        use_batchnorm: Toggle Batch Normalization in residual blocks.
        use_attention: Toggle MultiHeadAttention blocks.

    Returns:
        Compiled Keras Model instance.
    """
    inputs = Input(shape=(img_height, img_width, 3))

    x = Conv2D(filters, (3, 3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D((2, 2))(x)

    for _ in range(num_residual_blocks):
        x = residual_block(x, filters, kernel_size, use_batchnorm=use_batchnorm)
        if use_attention:
            x = attention_block(x, filters)
        if x.shape[1] >= 2 and x.shape[2] >= 2:
            x = MaxPooling2D((2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_units, activation="relu")(x)
    if dropout_rate > 0.0:
        x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="VerticilliumAttentionResNet")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transfer_learning_model(
    architecture_name: str,
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-4,
) -> Model:
    """
    Construct a transfer learning benchmark model with pre-trained ImageNet weights.

    Args:
        architecture_name: One of 'MobileNetV2', 'EfficientNetB0', or 'ResNet50V2'.
        input_shape: Input image shape (H, W, C).
        num_classes: Number of classification categories.
        dropout_rate: Classification head dropout rate.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras Model instance.
    """
    name_lower = architecture_name.lower()
    if "mobilenet" in name_lower:
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    elif "efficientnet" in name_lower:
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    elif "resnet" in name_lower:
        base_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported architecture '{architecture_name}'.")

    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name=f"{architecture_name}_Benchmark")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

