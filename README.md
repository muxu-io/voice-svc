# voice-svc

A small, **stateless** HTTP TTS server. Give it text plus a self-contained
**voice spec** and it streams back **PCM/WAV audio**. Two engines ship in the
image: **XTTS-v2** (Coqui TTS fork) for voice-cloning from a reference sample,
and **Piper** for fast preset voices.

The service has **no opinion** about who's calling it. It does **not** read
persona files, runtime state, or any shared filesystem — the caller resolves
which engine, which sample, and which prosody, and sends it on every request.
That is what lets this run as an independent service.

License: Apache-2.0 (service code). The XTTS-v2 / Piper model weights carry
their own licenses and are downloaded (XTTS) or baked into the image (Piper).

## Quickstart

```bash
docker compose -f compose.example.yml up
```

XTTS-v2 weights (~2 GB) download at first use into the `voice-svc-models`
volume. Piper voices are baked into the image under `/voices`. `/health` is
responsive as soon as the server is up and reports which engines loaded.

## API

| Method | Path                | Body / Result |
|--------|---------------------|---------------|
| GET    | `/health`           | `{ "status": "ok", "engines": [...], "engines_failed": [...], "mock_only": <bool> }` |
| GET    | `/voices`           | `{ "engines": [...], "piper_voices": [...] }` — loaded engines + the `.onnx` voice ids found in the piper model dir |
| POST   | `/synthesize`       | self-contained voice spec → streamed audio (see below) |
| POST   | `/v1/audio/speech`  | OpenAI-compatible passthrough → **piper only**; `voice` is a piper voice id, `speed` maps to baseline speed |

### `POST /synthesize`

```jsonc
{
  "text": "the line to speak",
  "engine": "xtts-v2",          // or "piper"
  "language": "en",
  "baseline_speed": 1.0,
  "fatigue_level": "rested",     // "rested" | "tired" | "exhausted" → prosody
  "piper_voice": null,           // required when engine == "piper"
  "sample_audio_b64": "UklGR...", // required when engine == "xtts-v2" (base64 WAV)
  "response_format": "pcm",      // "pcm" | "wav"
  "include_alignment": false
}
```

- **Engine-specific validation (422):** `xtts-v2` requires `sample_audio_b64`;
  `piper` requires `piper_voice`; any other `engine` is rejected.
- **PCM** responses stream `audio/L16; rate=<n>` chunks (the rate is in the
  Content-Type so the client can open its sink correctly). **WAV** buffers and
  returns a proper RIFF/WAVE file.
- **`include_alignment: true`** returns JSON:
  `{ "audio_base64", "audio_format", "sample_rate", "alignment": [...] }`.
- The XTTS reference sample is content-addressed to `<sha256>.wav` in the sample
  cache dir, so identical samples reuse the engine's conditioning latents across
  sentences and across requests — sending the sample on every request is cheap.

## Security & exposure

The service has **no authentication** — `/synthesize` is an open compute
endpoint. Run it **loopback-only**: the app binds `0.0.0.0` *inside the
container* (required for Docker's published port to reach it), and exposure is
gated at the publish layer — `compose.example.yml` binds `127.0.0.1:7000:7000`.
Do **not** publish the port to a non-loopback address without authentication in
front of it.

## GPU access (non-root)

The image runs as a non-root user (uid 10001). If your host exposes the GPU via
device nodes with restricted permissions — `crw-rw---- root:video` (mode 0660),
as openSUSE does — that user cannot initialize NVML and **XTTS-v2 fails to load**
with `Failed to initialize NVML: Insufficient Permissions` (Piper still works;
it doesn't use CUDA). Check with `ls -l /dev/nvidia0`.

The fix is to add the device-node's owning group to the container. Derive the
GID from the node — **do not hardcode it, it varies per host**:

```bash
VOICE_SVC_GPU_GID=$(stat -c '%g' /dev/nvidia0) docker compose -f compose.example.yml up
```

`compose.example.yml` wires that into `group_add`. (Hosts whose `/dev/nvidia*`
are already world-accessible, `0666`, don't need this.)

## Configuration (env)

| Variable | Default | Meaning |
|----------|---------|---------|
| `VOICE_SVC_PORT` | `7000` | Listen port |
| `VOICE_SVC_MOCK_ONLY` | unset | `true` → mock engine (silence shaped by text length), no GPU/weights (used in tests) |
| `VOICE_SVC_PIPER_BIN` | `piper` | Piper executable |
| `VOICE_SVC_PIPER_MODEL_DIR` | `/voices` | Directory of `.onnx` piper voices |
| `VOICE_SVC_SAMPLE_CACHE_DIR` | `/tmp/voice-samples` | Where base64 XTTS samples are materialized (`<sha256>.wav`) |

## Development

```bash
poetry install --with dev          # main + dev only (no torch / coqui-tts)
poetry run pytest                  # hermetic unit tests: mock engine + injected XTTS/Piper fakes, no GPU
poetry run ruff check src tests
poetry run black --check src tests
```

The engine stack (`coqui-tts`, `torch`, `torchaudio`, `torchcodec`,
`transformers`) is an optional, locked Poetry group installed in the image via
`poetry install --only main,engine`; unit tests never import it (they inject
fakes for the XTTS/Piper adapters).

> **Note:** `poetry.lock` is required for the Docker build (`COPY pyproject.toml
> poetry.lock`). Generate it once with `poetry lock` before the first image
> build / release.
