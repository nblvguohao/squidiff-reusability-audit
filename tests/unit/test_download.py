"""Unit tests for data verification — RED phase."""

import pytest

from reuse_gate.data.download import verify_existing_file


def test_wrong_checksum_is_rejected(tmp_path):
    """A file whose SHA256 doesn't match the expected must raise ValueError."""
    path = tmp_path / "x.bin"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_existing_file(path, "0" * 64)


def test_correct_checksum_is_accepted(tmp_path):
    """A file whose SHA256 matches must pass silently."""
    path = tmp_path / "correct.bin"
    path.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    # Should not raise
    verify_existing_file(path, expected)


def test_verify_existing_file_rejects_nonexistent(tmp_path):
    """Missing file must raise FileNotFoundError."""
    path = tmp_path / "missing.bin"
    with pytest.raises(FileNotFoundError):
        verify_existing_file(path, "0" * 64)
