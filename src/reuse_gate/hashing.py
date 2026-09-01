"""SHA256 file hashing for immutable artifact verification."""

from __future__ import annotations

import hashlib
from pathlib import Path

BUFFER_SIZE = 1 << 18  # 256 KiB chunks


def sha256_file(path: str | Path) -> str:
    """Compute the SHA256 hex digest of a file.

    Reads in 256 KiB chunks to handle large files without loading them into
    memory. Returns the lowercase hex digest string.

    Raises:
        FileNotFoundError: if the path does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    sha = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(BUFFER_SIZE)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()
