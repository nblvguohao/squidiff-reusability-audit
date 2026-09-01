"""Unit tests for deterministic run ID generation — RED phase."""

from reuse_gate.orchestration.run_id import generate_run_id


def test_run_id_changes_with_commit():
    """Run ID must change when git commit changes."""
    id1 = generate_run_id(
        candidate="squidiff",
        dataset_checksum="abc123",
        split_id="split_01",
        seed=13,
        git_commit="commit_a",
        container_digest="sha256:xyz",
    )
    id2 = generate_run_id(
        candidate="squidiff",
        dataset_checksum="abc123",
        split_id="split_01",
        seed=13,
        git_commit="commit_b",
        container_digest="sha256:xyz",
    )
    assert id1 != id2


def test_run_id_changes_with_seed():
    """Run ID must change when seed changes."""
    id1 = generate_run_id(
        candidate="squidiff",
        dataset_checksum="abc123",
        split_id="split_01",
        seed=13,
        git_commit="abc",
        container_digest="sha256:xyz",
    )
    id2 = generate_run_id(
        candidate="squidiff",
        dataset_checksum="abc123",
        split_id="split_01",
        seed=37,
        git_commit="abc",
        container_digest="sha256:xyz",
    )
    assert id1 != id2


def test_run_id_is_deterministic():
    """Same inputs must produce same run ID."""
    kwargs = {
        "candidate": "squidiff",
        "dataset_checksum": "abc123",
        "split_id": "split_01",
        "seed": 13,
        "git_commit": "abc",
        "container_digest": "sha256:xyz",
    }
    assert generate_run_id(**kwargs) == generate_run_id(**kwargs)


def test_run_id_contains_candidate():
    """Run ID must include the candidate name for readability."""
    run_id = generate_run_id(
        candidate="squidiff",
        dataset_checksum="abc123",
        split_id="split_01",
        seed=13,
        git_commit="abc",
        container_digest="sha256:xyz",
    )
    assert "squidiff" in run_id
    assert len(run_id) >= 8
