"""Unit tests for external runner — RED phase."""


import pytest

from reuse_gate.models.external import ExternalProcessRunner
from reuse_gate.schemas import RunRequest


@pytest.fixture
def run_request(tmp_path):
    """Minimal valid RunRequest fixture."""
    return RunRequest(
        run_id="test-run-1",
        candidate="nichetrans",
        model_id="smoke",
        train_path=tmp_path / "train.h5ad",
        validation_path=tmp_path / "val.h5ad",
        test_path=tmp_path / "test.h5ad",
        config_path=tmp_path / "config.yaml",
        output_dir=tmp_path / "out",
        seed=13,
    )


def test_external_runner_captures_failure(tmp_path, run_request):
    """A failing command must produce a failed RunResult with preserved stderr."""
    # A command that will fail
    failing_cmd = ["python", "-c", "import sys; sys.stderr.write('BOOM\\n'); sys.exit(1)"]
    runner = ExternalProcessRunner(failing_cmd)
    result = runner.run(run_request)

    assert result.status == "failed"
    assert result.stderr_path.exists()
    assert result.stdout_path.exists()
    assert result.error_type is not None


def test_external_runner_captures_success(tmp_path, run_request):
    """A successful command must produce a success RunResult."""
    success_cmd = [
        "python", "-c",
        "import json, pathlib as p; "
        "p.Path('predictions.npy').write_text('dummy'); "
        "print(json.dumps({'metric': 0.9}))"
    ]
    runner = ExternalProcessRunner(success_cmd)
    result = runner.run(run_request)

    assert result.status == "success"
    assert result.error_type is None
    assert result.stdout_path.exists()


def test_external_runner_sets_timeout(tmp_path, run_request):
    """Long-running commands must be killed after timeout."""
    infinite_cmd = ["python", "-c", "import time; time.sleep(30)"]
    runner = ExternalProcessRunner(infinite_cmd, timeout_seconds=2)
    result = runner.run(run_request)

    assert result.status == "failed"
    assert "timeout" in (result.error_type or "").lower()


def test_external_runner_result_has_required_fields(tmp_path, run_request):
    """Every RunResult must have git_commit and container_digest."""
    cmd = ["python", "-c", ""]
    runner = ExternalProcessRunner(cmd)
    result = runner.run(run_request)

    assert result.run_id == run_request.run_id
    assert result.git_commit  # must not be empty
    assert result.container_digest  # must not be empty
