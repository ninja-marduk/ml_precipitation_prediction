"""
Lightweight helper to get a ConvGRU2D layer, trying native Keras first and
falling back to tensorflow-addons if needed.

Usage:
    from custom_layers.convgru2d import build_convgru2d_layer
    convgru = build_convgru2d_layer(filters=64, kernel_size=(3, 3))
    x = convgru(inputs)
"""

from typing import Tuple, Optional

try:
    # Preferred: native Keras (available in TF >= 2.11/2.12 depending on build)
    from tensorflow.keras.layers import ConvGRU2D  # type: ignore
except Exception:  # pragma: no cover - environment-dependent
    ConvGRU2D = None


def _get_from_tfa():
    try:
        import tensorflow_addons as tfa  # type: ignore

        return tfa.layers.ConvGRU2D
    except Exception:  # pragma: no cover - optional dependency
        return None


def build_convgru2d_layer(
    filters: int,
    kernel_size: Tuple[int, int] = (3, 3),
    return_sequences: bool = True,
    padding: str = "same",
    name: str = "conv_gru2d",
    **kwargs,
):
    """
    Returns a ConvGRU2D layer if available.

    Tries native tf.keras first; if missing, tries tensorflow-addons.
    Raises an ImportError with guidance when neither is present.
    """
    layer_cls = ConvGRU2D or _get_from_tfa()
    if layer_cls is None:
        raise ImportError(
            "ConvGRU2D is not available in this TensorFlow/Keras build. "
            "Install a TF version that includes ConvGRU2D (e.g., >=2.12) or "
            "add tensorflow-addons and retry."
        )
    return layer_cls(
        filters=filters,
        kernel_size=kernel_size,
        return_sequences=return_sequences,
        padding=padding,
        name=name,
        **kwargs,
    )
