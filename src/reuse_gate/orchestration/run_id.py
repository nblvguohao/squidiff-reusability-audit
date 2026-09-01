"""Deterministic run ID generation.

Run IDs are derived from SHA256 of all inputs so that identical configuration
always produces the same ID, and any change is detectable.
"""

from __future__ import annotations

import hashlib


def generate_run_id(
    candidate: str,
    dataset_checksum: str,
    split_id: str,
    seed: int,
    git_commit: str,
    container_digest: str,
) -> str:
    """Generate a deterministic, human-readable run ID.

    Format: {candidate}-{seed}-{short_hash}
    """
    payload = "|".join(
        [
            candidate,
            dataset_checksum,
            split_id,
            str(seed),
            git_commit,
            container_digest,
        ]
    )
    full_hash = hashlib.sha256(payload.encode()).hexdigest()
    short_hash = full_hash[:12]
    return f"{candidate}-s{seed}-{short_hash}"
