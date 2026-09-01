"""Unit tests for file hashing — RED phase."""

import pytest

from reuse_gate.hashing import sha256_file


def test_sha256_known_value(tmp_path):
    """SHA256 must match the NIST test vector for 'abc'."""
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_file(path) == expected


def test_sha256_empty_file(tmp_path):
    """Empty file has known SHA256."""
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert sha256_file(path) == expected


def test_sha256_nonexistent_file(tmp_path):
    """Missing file must raise FileNotFoundError."""
    path = tmp_path / "does_not_exist.bin"
    with pytest.raises(FileNotFoundError):
        sha256_file(path)


def test_sha256_deterministic(tmp_path):
    """Same content must produce same hash every time."""
    path = tmp_path / "repeat.bin"
    path.write_bytes(b"repeatable content")
    h1 = sha256_file(path)
    h2 = sha256_file(path)
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)
