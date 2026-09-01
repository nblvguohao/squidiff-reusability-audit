"""Verified downloads and checksum validation for data artifacts."""

from __future__ import annotations

from pathlib import Path

from reuse_gate.hashing import sha256_file


def verify_existing_file(path: str | Path, expected_sha256: str) -> None:
    """Verify that an existing file matches its expected SHA256.

    Args:
        path: Path to the file.
        expected_sha256: Expected lowercase hex SHA256 digest.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the SHA256 does not match.
    """
    path = Path(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"SHA256 mismatch for {path.name}: "
            f"expected {expected_sha256}, got {actual}"
        )
