"""Container-backed integration fixtures.

Two modes:
  * `container_base_url` / `http` — mock-mode container (VOICE_SVC_MOCK_ONLY=true).
    No GPU, no model download. Selected by `pytest -m integration`.
  * `gpu_container_base_url` / `gpu_http` — real-engine container with `--gpus all`
    and persistent model/HF caches. Selected by `pytest -m gpu`. Requires an
    NVIDIA host with the container toolkit.

Both modes need the image built locally. Default tag is
`voice-svc:integration`; override with VOICE_SVC_IMAGE.

    docker build -t voice-svc:integration .
    poetry run pytest -m integration       # mock mode
    poetry run pytest -m gpu               # real engines (XTTS-v2 + Piper)

Unlike services that load weights in a background thread, voice-svc loads its
engines synchronously in `__main__` *before* uvicorn binds the port — so once
`/health` answers 200 the engines are already registered (or recorded in
`engines_failed`). The GPU fixture therefore uses a long health timeout to
cover the one-time XTTS-v2 weight download (~2 GB) that happens before the
server comes up.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator

import httpx
import pytest

DEFAULT_IMAGE = "voice-svc:integration"
DEFAULT_GPU_MODELS_VOLUME = "voice-svc-integration-models"
DEFAULT_GPU_HF_VOLUME = "voice-svc-integration-hf"
HEALTH_TIMEOUT_S = 30.0
GPU_HEALTH_TIMEOUT_S = 900.0  # XTTS-v2 weight download + load before the port binds
HEALTH_POLL_INTERVAL_S = 1.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=5)
    except (subprocess.SubprocessError, OSError):
        return False
    return True


def _image_present(image: str) -> bool:
    try:
        subprocess.run(
            ["docker", "image", "inspect", image], check=True, capture_output=True, timeout=10
        )
    except subprocess.SubprocessError:
        return False
    return True


def _nvidia_host_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True, timeout=5)
    except (subprocess.SubprocessError, OSError):
        return False
    return True


def _nvidia_device_gid() -> str | None:
    """GID owning /dev/nvidia0. On hosts where the device nodes are 0660
    root:<group> (e.g. openSUSE root:video), the non-root container user must
    join this group to reach the GPU. Returns the GID as a string for
    `--group-add`, or None if the node is absent / world-accessible."""
    try:
        st = os.stat("/dev/nvidia0")
    except OSError:
        return None
    # Already world-accessible → no supplementary group needed.
    if st.st_mode & 0o006:
        return None
    return str(st.st_gid)


def _wait_for_health(base_url: str, container_id: str, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return
        except httpx.HTTPError as exc:
            last_err = exc
        time.sleep(HEALTH_POLL_INTERVAL_S)

    logs = subprocess.run(
        ["docker", "logs", container_id], capture_output=True, text=True, timeout=10
    )
    raise RuntimeError(
        f"container not healthy in {timeout_s}s (last error: {last_err!r})\n"
        f"--- container logs ---\n{logs.stdout}\n{logs.stderr}"
    )


@pytest.fixture(scope="session")
def container_base_url() -> Iterator[str]:
    if not _docker_available():
        pytest.skip("docker not available on PATH")
    image = os.environ.get("VOICE_SVC_IMAGE", DEFAULT_IMAGE)
    if not _image_present(image):
        pytest.skip(f"image {image!r} not present; build with `docker build -t {image} .`")

    port = _free_port()
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "-e",
            "VOICE_SVC_MOCK_ONLY=true",
            "-p",
            f"127.0.0.1:{port}:7000",
            image,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    container_id = run.stdout.strip()
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(base_url, container_id, timeout_s=HEALTH_TIMEOUT_S)
        yield base_url
    finally:
        subprocess.run(["docker", "stop", container_id], capture_output=True, timeout=15)


@pytest.fixture(scope="session")
def http(container_base_url: str) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=container_base_url, timeout=30.0) as client:
        yield client


@pytest.fixture(scope="session")
def gpu_container_base_url() -> Iterator[str]:
    if not _docker_available():
        pytest.skip("docker not available on PATH")
    if not _nvidia_host_available():
        pytest.skip("nvidia-smi not available; GPU integration requires an NVIDIA host")
    image = os.environ.get("VOICE_SVC_IMAGE", DEFAULT_IMAGE)
    if not _image_present(image):
        pytest.skip(f"image {image!r} not present; build with `docker build -t {image} .`")

    models_volume = os.environ.get("VOICE_SVC_INTEGRATION_MODELS_VOLUME", DEFAULT_GPU_MODELS_VOLUME)
    hf_volume = os.environ.get("VOICE_SVC_INTEGRATION_HF_VOLUME", DEFAULT_GPU_HF_VOLUME)
    port = _free_port()
    docker_args = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--gpus",
        "all",
        "-e",
        "COQUI_TOS_AGREED=1",
    ]
    # The image runs non-root; on hosts with 0660 root:video device nodes the
    # user must join that group's GID to initialize NVML / CUDA.
    if (gpu_gid := _nvidia_device_gid()) is not None:
        docker_args += ["--group-add", gpu_gid]
    docker_args += [
        "-v",
        f"{models_volume}:/home/app/.local/share/tts",
        "-v",
        f"{hf_volume}:/home/app/.cache/huggingface",
        "-p",
        f"127.0.0.1:{port}:7000",
    ]
    if hf_token := os.environ.get("HF_TOKEN"):
        docker_args.extend(["-e", f"HF_TOKEN={hf_token}"])
    docker_args.append(image)
    run = subprocess.run(docker_args, check=True, capture_output=True, text=True, timeout=15)
    container_id = run.stdout.strip()
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(base_url, container_id, timeout_s=GPU_HEALTH_TIMEOUT_S)
        yield base_url
    finally:
        subprocess.run(["docker", "stop", container_id], capture_output=True, timeout=15)


@pytest.fixture(scope="session")
def gpu_http(gpu_container_base_url: str) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=gpu_container_base_url, timeout=300.0) as client:
        yield client
