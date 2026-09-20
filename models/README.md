# Container artifacts

`export_resnet.py` also writes a SHA-256 manifest and a deterministic PyTorch
reference (`resnet50.manifest.json`, `resnet50.reference.npz`). Keep these with
the corresponding ONNX file. The GPU container verifies the manifest before
startup; see the [container runbook](../docker/README.md).
