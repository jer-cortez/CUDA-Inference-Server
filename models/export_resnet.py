"""Export torchvision ResNet-50 to ONNX with a dynamic batch dimension.

Run this on the GPU box (it needs torch/torchvision, which are not runtime
dependencies of the server itself):

    python models/export_resnet.py

Writes models/resnet50.onnx, models/imagenet_classes.txt, and prints a
reference prediction the C++ and integration tests can be checked against.

Why ResNet-50 rather than something lighter: the milestone-7 benchmark needs
per-request GPU work to dominate, otherwise dynamic batching's throughput gain
is buried under per-request overhead and the headline number understates the
result.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import torch
import torchvision

MODELS_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX_PATH = MODELS_DIR / "resnet50.onnx"
DEFAULT_LABELS_PATH = MODELS_DIR / "imagenet_classes.txt"

INPUT_NAME = "input"
OUTPUT_NAME = "output"
CHANNELS, HEIGHT, WIDTH = 3, 224, 224
NUM_CLASSES = 1000

# The opset needs to be recent enough that ORT binds the dynamic batch axis
# cleanly; 17 is widely supported by the 1.16+ GPU releases.
OPSET_VERSION = 17


def export(onnx_path: Path, labels_path: Path) -> None:
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    model.eval()

    # Traced with batch 2, not 1: a size-1 batch can hide bugs where a
    # dimension is squeezed away rather than kept symbolic.
    example = torch.randn(2, CHANNELS, HEIGHT, WIDTH)

    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        # The whole milestone depends on this. Without a symbolic batch axis
        # the session is frozen at the traced size, and every batch the
        # scheduler forms that isn't exactly that size fails to bind --
        # which would defeat dynamic batching entirely.
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
    )

    _assert_dynamic_batch(onnx_path)
    _write_labels(weights, labels_path)
    _print_reference(model, onnx_path)


def _assert_dynamic_batch(onnx_path: Path) -> None:
    """Verify the exported graph really has a symbolic batch dim.

    Checked rather than assumed: a silently-static export produces a model
    that works perfectly for one batch size and fails for every other, which
    surfaces later as a confusing runtime bind error rather than an export
    problem.
    """
    graph = onnx.load(str(onnx_path)).graph

    for tensor, label in ((graph.input[0], "input"), (graph.output[0], "output")):
        dim = tensor.type.tensor_type.shape.dim[0]
        if not dim.dim_param:
            raise SystemExit(
                f"export produced a static {label} batch dimension "
                f"({dim.dim_value}); dynamic_axes did not take effect"
            )

    print(f"verified dynamic batch axis on {onnx_path.name}")


def _write_labels(weights, labels_path: Path) -> None:
    categories = weights.meta["categories"]
    if len(categories) != NUM_CLASSES:
        raise SystemExit(f"expected {NUM_CLASSES} categories, got {len(categories)}")
    labels_path.write_text("\n".join(categories) + "\n")
    print(f"wrote {labels_path.name} ({len(categories)} classes)")


def _print_reference(model: torch.nn.Module, onnx_path: Path) -> None:
    """Print a deterministic reference prediction for the tests to assert on.

    Uses a fixed-seed synthetic input rather than a real image so the value is
    reproducible anywhere without shipping a JPEG into the repo.
    """
    generator = torch.Generator().manual_seed(0)
    reference_input = torch.randn(1, CHANNELS, HEIGHT, WIDTH, generator=generator)

    with torch.no_grad():
        logits = model(reference_input).numpy()

    top1 = int(np.argmax(logits[0]))
    digest = hashlib.sha256(reference_input.numpy().tobytes()).hexdigest()[:16]

    print()
    print("reference (seed=0 synthetic input):")
    print(f"  input sha256[:16] : {digest}")
    print(f"  logits shape      : {logits.shape}")
    print(f"  top-1 class index : {top1}")
    print(f"  top-1 logit       : {logits[0][top1]:.6f}")
    print()
    print(f"model written to {onnx_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export(args.output, args.labels)


if __name__ == "__main__":
    main()
