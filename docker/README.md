# GPU container runbook

This packages the existing API and native scheduler for one Linux AMD64 NVIDIA
GPU. The local Compose service binds port 8000 to host loopback for development
validation. The separate [authenticated HTTPS runbook](SECURITY.md) describes
the standalone, TCP/443-only pilot deployment; it does not provision AWS
resources.

## Dependency contract

| Component | Selection |
| --- | --- |
| Architecture / OS | Linux AMD64 / Ubuntu 22.04 |
| CUDA | NVIDIA CUDA 12.8.1 development and runtime images, pinned by AMD64 manifest digest |
| cuDNN | 9.x, fixed by the selected image digest |
| ONNX Runtime GPU | 1.23.2 Linux x64 release, official release asset SHA-256 checked before extraction |
| Python | Ubuntu Python 3.10, patched by the distribution |
| Native GPU code | SM 75 (T4) and SM 89 (L4) |
| Python packages | Exact pins in `requirements-runtime.txt` and `requirements-build.txt` |
| pybind11 | Existing CMake dependency, v2.13.6 |

The [official ORT compatibility matrix](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
lists ORT 1.21–1.26 with CUDA 12.8 and cuDNN 9.x, and documents compatibility
across CUDA 12.x minor versions. The selected CUDA 12.8.1 libraries align with
the ORT build series; actual provider loading remains a GPU-host validation
gate. cuDNN major versions must match. The ORT checksum comes from the
[official 1.23.2 release asset metadata](https://api.github.com/repos/microsoft/onnxruntime/releases/tags/v1.23.2).
Base images and ORT artifacts are pinned; Ubuntu package repository updates are
intentionally retained, so repeated builds are not byte-identical. Record the
final image digest when promoting an image. Revalidate this stack when updating
pins; these versions are a packaging baseline, not a claim of current security
certification.

No host CUDA toolkit is needed to run the image. The host does need a compatible
NVIDIA driver, Docker Engine, Compose v2, and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Use a driver supported by CUDA 12.8 and the GPU/instance type. Prefer the current
provider-supported driver for the chosen GPU image, then verify it with the
container's real readiness/inference probe. G6f additionally
needs its supported GRID/vGPU driver; ordinary T4 validation does not validate
fractional L4 compatibility. Check the selected AWS image requirements before
benchmarking.

The model and a Linux GPU are not included in this repository. GPU execution
must be validated on that host; a Mac build or stub test cannot establish it.

## Export and identify the model

Run the existing export workflow in a separate environment with compatible
`torch`, `torchvision`, `onnx`, and NumPy installed (see
[model documentation](../models/README.md)). Export does not require a GPU.
The export and artifact validation were exercised on macOS ARM64 / Python 3.13
using `torch==2.14.0`, `torchvision==0.29.0`, `onnx==1.23.0`, and `numpy==2.5.3`.
These are exporter dependencies, separate from the Python 3.10 serving image.

```sh
python models/export_resnet.py --model-version resnet50-imagenet1k-v2
```

This writes `resnet50.onnx`, `resnet50.manifest.json`, `resnet50.reference.npz`,
and ImageNet labels. The manifest records the model SHA-256, dynamic input/output
contract, preprocessing, exporter versions, and reference tensor SHA-256.
The reference contains eight distinct fixed synthetic inputs and their PyTorch logits, allowing
GPU inference to be compared without shipping PyTorch in the serving image.
Treat the ONNX file and manifest as a versioned pair. Store trusted manifests
with release artifacts; a checksum detects mismatch, not malicious replacement
of both files. Keep the reference alongside them for verification.

The API accepts an already preprocessed tensor. It does not decode images or
apply the manifest's image transforms. The manifest records torchvision's
recommended IMAGENET1K_V2 resize-short-edge-to-232 then center-crop-to-224
transforms. The existing optional `image_utils.preprocess` helper instead
resizes directly to 224 × 224, which can produce different image predictions;
this milestone does not change that helper. Both apply ImageNet normalization.
The synthetic reference is already model input and should be sent directly.

## Build and run

From the repository root:

```sh
docker build --platform linux/amd64 -f docker/Dockerfile -t cuda-db:gpu-local .
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs --tail 100 inference
curl --fail http://127.0.0.1:8000/readyz
```

Compose mounts the repository `models` directory read-only. To use another
directory, export `CUDA_DB_MODELS_DIR` as its absolute path before running
Compose. The mounted artifacts must be readable by UID/GID 10001. Missing model
or manifest, unsupported manifest contract, and checksum mismatch fail before
server startup. A model probe and the existing GPU guard determine readiness;
the image refuses `CUDA_DB_REQUIRE_GPU=false`.

The final image contains the installed wheel and ORT shared libraries at
`/opt/onnxruntime/lib`, preserving the extension's RPATH. It contains no compiler,
source checkout, model exporter, or model. The image build imports the native
extension and checks `ldd` output for the extension, ORT core, and CUDA/shared
provider libraries. Every required library must resolve except `libcuda.so.1`,
which the host supplies at launch. This checks packaging without executing the
CUDA provider; actual provider loading is checked when the application creates
its real session.

Compose requests one GPU, runs non-root, drops capabilities, uses a read-only
root filesystem with a 256 MiB `/tmp`, and rotates logs at 10 MiB × 3 files.
The image defaults to one Uvicorn worker; each extra worker would create a
separate model and scheduler. Admission defaults remain 16 requests / 4 MiB body
/ 30-second deadline. Batch size, wait interval, and admission settings can be
overridden through the variables in Compose. Keep enough GPU memory for context,
workspace, and the engine's padded maximum batch, not only the model weights.

## Validate on a GPU host

Run the reference check from the host development environment with NumPy:

```sh
python docker/smoke_test.py --models models
```

It checks readiness, ONNX identity, finite 1000-class output, matching top-1,
maximum absolute logit error at most 0.01, and correct output mapping for eight
distinct concurrent inputs with unique response IDs.
It reports batching counters; actual coalescing is timing-dependent. The
existing model integration tests check distinct inputs and solo/batch agreement.

Run those four integration tests against the **installed GPU wheel** by using
a disposable container with test dependencies installed under `/tmp` using the
serving interpreter (a new venv would not inherit the installed serving wheel):

```sh
docker run --rm --gpus all --entrypoint sh \
  -v "$PWD/models:/models:ro" -v "$PWD/tests:/tests:ro" \
  cuda-db:gpu-local -c 'python -m pip install --no-cache-dir --constraint /opt/cuda-db/requirements-runtime.txt --target /tmp/test-deps pytest==8.3.5 pytest-asyncio==0.26.0 httpx==0.28.1 && PYTHONPATH=/tmp/test-deps python -m pytest --asyncio-mode=auto -p no:cacheprovider -q /tests/integration/test_onnx_model.py'
```

This verification-only command downloads test tools; normal startup downloads
nothing. Test dependencies are not part of the serving image. Confirm GPU
activity during repeated requests with `nvidia-smi` or an appropriate profiler.
Selecting the CUDA provider does not prove that every graph operator executes
on the GPU; some operators may be assigned to CPU by ORT.

Additional release checks on the GPU host:

1. Run the image with `--entrypoint sh` and inspect `ldd` on the installed
   native extension and `/opt/onnxruntime/lib/libonnxruntime_providers_cuda.so`.
   Check with GPU access enabled so injected driver libraries are present.
2. Start with a missing model, a mismatched manifest checksum, and without GPU
   access. Each must fail startup; none should serve the CPU stub.
3. Exercise overload and confirm 503 rejections stay bounded by admission.
4. Stop while requests are in flight, then restart and repeat readiness and
   smoke checks. Record signal/exit behavior and timings.
5. Repeat with maximum batch sizes 1, 4, and 8 to establish model compatibility.

## Stop, recovery, and limitations

```sh
docker compose -f docker/docker-compose.yml stop
docker compose -f docker/docker-compose.yml start
docker compose -f docker/docker-compose.yml restart inference
docker compose -f docker/docker-compose.yml down
```

The entrypoint uses `exec`, so Uvicorn receives SIGTERM directly. Uvicorn allows
45 seconds for request handling; Compose allows 60 seconds before killing the
container. Milestone 1 retains native work until completion, including work
whose HTTP deadline elapsed. Hung GPU calls cannot be cancelled safely; the
container stop deadline is the final bound, and a forced kill can lose accepted
requests. Increase the limits together if measured normal drain requires more.

Readiness has a 120-second startup allowance, then checks every 10 seconds.
Docker's unhealthy status does not automatically restart the container. This
milestone intentionally leaves operational recovery policy to the deployment
configuration; inspect logs and restart explicitly during local validation.

For missing libraries, inspect provider linkage and the pinned runtime image.
For `no CUDA device` or provider initialization errors, check host `nvidia-smi`,
driver compatibility and Container Toolkit configuration. For a checksum
failure, restore the matching model/manifest pair rather than regenerating a
checksum around an unknown artifact. For a read-only-filesystem error, identify
the actual needed cache path before granting further write access.

GPU correctness, overload/stop checks, and full image execution must be recorded
as pending until run on suitable hardware. AWS benchmarking, ECR publishing,
and Terraform remain outside this repository's deployment workflow.

## Local verification (2026-09-19)

- Built and loaded `cuda-db:gpu-local` for Linux AMD64 on Docker Desktop for
  Apple Silicon. Native CUDA/ONNX wheel compilation, import, `pip check`, and
  all required shared-library checks passed.
- Host suite: 48 passed, 4 GPU-model tests skipped, including a run with the
  container's pinned Python runtime packages in an isolated directory.
- Installed-image suite: 40 passed, 5 skipped, running as UID 10001 with a
  read-only root filesystem and temporary test dependencies constrained to
  the serving lockfile. The skips are four GPU-model tests and the optional
  Pillow preprocessing module (eight host tests); Pillow is not a serving
  dependency.
- Compose configuration, shell syntax, and whitespace checks passed.
- The image rejected a missing model and `CUDA_DB_REQUIRE_GPU=false`.
  With the exported model mounted read-only, checksum validation passed and
  startup exited with a CUDA driver error on this GPU-less host, rather than
  falling back to the stub.
- CPU export passed ONNX graph validation, dynamic-batch checks, model and
  reference checksums, and finite/distinct reference-tensor checks. Generated
  artifacts are available locally under `models/` and ignored by Git.

Successful GPU readiness, reference inference, GPU memory fit, and real GPU
stop/restart behavior remain unverified. Run the GPU-host checks above before
using this image for the pilot.
