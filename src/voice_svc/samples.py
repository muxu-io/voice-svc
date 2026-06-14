"""Reference-sample materialization. The XTTS reference clip arrives as
base64 in the request; the engine needs a filesystem path to encode
conditioning latents. We write each distinct clip once to
`<cache_dir>/<sha256>.wav`. Because XTTSEngine caches latents keyed by that
path, identical samples re-use latents across sentences and across requests
for the service's lifetime."""

from __future__ import annotations

import base64
import binascii
import hashlib
from pathlib import Path


def materialize_sample(sample_audio_b64: str, cache_dir: Path) -> Path:
    try:
        data = base64.b64decode(sample_audio_b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"invalid base64 sample audio: {e}") from e

    digest = hashlib.sha256(data).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{digest}.wav"
    if not path.exists():
        tmp = path.with_suffix(".wav.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return path
