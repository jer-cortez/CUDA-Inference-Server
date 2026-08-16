"""Image decode and preprocessing for the ResNet-50 input contract.

WHERE NORMALIZATION HAPPENS -- read before adding preprocessing anywhere else.
ResNet-50 was trained on ImageNet mean/std-normalized tensors, so something must
apply that before inference. This module is that single place. The project also
has a CUDA normalize kernel (cpp/include/cuda_db/kernels/normalize.cuh), but it
deliberately does NOT run in the ONNX path -- applying both would normalize
twice and produce confident nonsense that still looks shape-correct. The kernel
takes over only when the device-resident IoBinding path lands, at which point
this module drops to decode/resize and the mean/std step moves to the GPU.

Kept on the CPU for now because JPEG/PNG decode has no CUDA kernel here anyway,
so the tensor is already on the host at this point.
"""

from __future__ import annotations

import io

import numpy as np

# Same constants torchvision's ImageNet transforms use, and the same ones
# baked into CudaExecutionEngine::Options. If you change one, change both.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGET_SIZE = 224
CHANNELS = 3


def decode_image(data: bytes) -> np.ndarray:
    """Decode encoded image bytes to an HWC uint8 array.

    Pillow is imported lazily so the module stays importable (and the stub-engine
    tests keep running) in an environment without Pillow installed.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Pillow is required to decode images; install it with 'pip install pillow'"
        ) from exc

    try:
        image = Image.open(io.BytesIO(data))
        # Forces RGB: PNGs may carry an alpha channel and some JPEGs are
        # grayscale, either of which would give the model the wrong channel
        # count.
        image = image.convert("RGB")
        image = image.resize((TARGET_SIZE, TARGET_SIZE))
    except Exception as exc:
        raise ValueError(f"could not decode image: {exc}") from exc

    return np.asarray(image, dtype=np.uint8)


def to_model_input(image_hwc: np.ndarray) -> np.ndarray:
    """Convert an HWC uint8 image to the flat NCHW float32 the engine expects.

    Scales to [0, 1], applies ImageNet mean/std per channel, transposes
    HWC -> CHW, and flattens. Returns a C-contiguous array of
    ``CHANNELS * TARGET_SIZE * TARGET_SIZE`` floats.
    """
    if image_hwc.ndim != 3 or image_hwc.shape[2] != CHANNELS:
        raise ValueError(
            f"expected an HWC image with {CHANNELS} channels, got shape {image_hwc.shape}"
        )
    if image_hwc.shape[0] != TARGET_SIZE or image_hwc.shape[1] != TARGET_SIZE:
        raise ValueError(
            f"expected a {TARGET_SIZE}x{TARGET_SIZE} image, got "
            f"{image_hwc.shape[0]}x{image_hwc.shape[1]}"
        )

    scaled = image_hwc.astype(np.float32) / 255.0
    normalized = (scaled - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> CHW, then a copy so the result is contiguous: transpose only
    # changes strides, and the binding forcecasts a non-contiguous array into
    # a temporary, silently costing a copy per request on the hot path.
    chw = np.transpose(normalized, (2, 0, 1))
    return np.ascontiguousarray(chw, dtype=np.float32).reshape(-1)


def preprocess(data: bytes) -> np.ndarray:
    """Decode encoded image bytes straight to flat NCHW model input."""
    return to_model_input(decode_image(data))
