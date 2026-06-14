from __future__ import annotations

import base64
import hashlib

import pytest

from voice_svc.samples import materialize_sample


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_writes_content_addressed_file(tmp_path):
    data = b"RIFFfake-wav-bytes"
    path = materialize_sample(_b64(data), tmp_path)

    digest = hashlib.sha256(data).hexdigest()
    assert path == tmp_path / f"{digest}.wav"
    assert path.read_bytes() == data


def test_same_bytes_same_path(tmp_path):
    data = b"identical"
    p1 = materialize_sample(_b64(data), tmp_path)
    p2 = materialize_sample(_b64(data), tmp_path)

    assert p1 == p2


def test_different_bytes_different_path(tmp_path):
    p1 = materialize_sample(_b64(b"one"), tmp_path)
    p2 = materialize_sample(_b64(b"two"), tmp_path)

    assert p1 != p2


def test_creates_cache_dir(tmp_path):
    target_dir = tmp_path / "nested" / "samples"
    path = materialize_sample(_b64(b"x"), target_dir)

    assert path.parent == target_dir
    assert path.exists()


def test_no_tmp_left_behind(tmp_path):
    materialize_sample(_b64(b"x"), tmp_path)

    assert list(tmp_path.glob("*.tmp")) == []


def test_rejects_invalid_base64(tmp_path):
    with pytest.raises(ValueError):
        materialize_sample("not!!base64!!", tmp_path)
