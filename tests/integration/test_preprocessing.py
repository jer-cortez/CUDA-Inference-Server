"""Tests for the CPU preprocessing path.

These need no GPU and no exported model -- they pin down the input contract
(shape, layout, normalization) that OnnxExecutionEngine depends on, so a change
here that would silently mis-feed the model fails loudly instead.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from cuda_db.preprocessing.image_utils import (
    CHANNELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TARGET_SIZE,
    decode_image,
    preprocess,
    to_model_input,
)

pytest.importorskip("PIL", reason="Pillow is needed to decode test images")


def _png_bytes(color: tuple[int, int, int], size: int = TARGET_SIZE) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_produces_hwc_uint8():
    image = decode_image(_png_bytes((10, 20, 30)))

    assert image.shape == (TARGET_SIZE, TARGET_SIZE, CHANNELS)
    assert image.dtype == np.uint8


def test_decode_resizes_and_forces_rgb():
    """A grayscale or alpha-carrying image must still come out 3-channel at the
    target size, or the model gets the wrong input shape."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("L", (64, 100)).save(buffer, format="PNG")

    image = decode_image(buffer.getvalue())
    assert image.shape == (TARGET_SIZE, TARGET_SIZE, CHANNELS)


def test_decode_rejects_garbage():
    with pytest.raises(ValueError):
        decode_image(b"this is not an image")


def test_to_model_input_is_flat_chw_float32():
    image = decode_image(_png_bytes((0, 0, 0)))
    flat = to_model_input(image)

    assert flat.shape == (CHANNELS * TARGET_SIZE * TARGET_SIZE,)
    assert flat.dtype == np.float32
    # The pybind11 layer forcecasts non-contiguous arrays into a temporary,
    # which would cost a hidden copy on every request.
    assert flat.flags["C_CONTIGUOUS"]


def test_normalization_matches_imagenet_constants():
    """A solid mid-gray image has a known exact value per channel after
    normalization, so this catches a wrong constant, a missing /255, or a
    channel-order mix-up."""
    value = 128
    flat = to_model_input(decode_image(_png_bytes((value, value, value))))

    scaled = value / 255.0
    plane = TARGET_SIZE * TARGET_SIZE

    for channel in range(CHANNELS):
        expected = (scaled - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        channel_values = flat[channel * plane : (channel + 1) * plane]
        np.testing.assert_allclose(channel_values, expected, rtol=1e-5, atol=1e-5)


def test_channel_order_is_preserved():
    """Distinct per-channel values must land in distinct CHW planes -- this is
    what fails if HWC->CHW is transposed the wrong way."""
    flat = to_model_input(decode_image(_png_bytes((255, 0, 0))))
    plane = TARGET_SIZE * TARGET_SIZE

    red = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    green = (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]

    np.testing.assert_allclose(flat[0], red, rtol=1e-5)
    np.testing.assert_allclose(flat[plane], green, rtol=1e-5)


def test_to_model_input_rejects_wrong_shape():
    with pytest.raises(ValueError):
        to_model_input(np.zeros((10, 10, CHANNELS), dtype=np.uint8))


def test_preprocess_matches_the_two_step_path():
    data = _png_bytes((200, 100, 50))
    np.testing.assert_array_equal(preprocess(data), to_model_input(decode_image(data)))
