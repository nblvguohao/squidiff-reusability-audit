"""Subprocess-based execution for upstream model containers or venvs."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from reuse_gate.schemas import RunRequest, RunResult


def _get_git_commit() -> str:
    """Return the current git commit hash (short), or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


class ExternalProcessRunner:
    """Runs a candidate model via subprocess with timeout, capturing stdout/stderr."""

    def __init__(
        self,
        command: list[str],
        timeout_seconds: int = 300,
        container_digest: str = "containerless:uv-venv",
    ):
        self.command = command
        self.timeout_seconds = timeout_seconds
        self._container_digest = container_digest

    @property
    def model_id(self) -> str:
        return "external-process"

    def run(self, request: RunRequest) -> RunResult:
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = output_dir / "stdout.txt"
        stderr_path = output_dir / "stderr.txt"

        start_time = time.monotonic()

        try:
            with open(stdout_path, "w") as stdout_fh, open(stderr_path, "w") as stderr_fh:
                proc = subprocess.run(
                    self.command,
                    cwd=output_dir,
                    stdout=stdout_fh,
                    stderr=stderr_fh,
                    timeout=self.timeout_seconds,
                )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start_time
            stderr_path.write_text(f"Process timed out after {elapsed:.1f}s")
            return RunResult(
                run_id=request.run_id,
                status="failed",
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                error_type="timeout",
                error_message=f"Process exceeded {self.timeout_seconds}s timeout",
                git_commit=_get_git_commit(),
                container_digest=self._container_digest,
                train_seconds=elapsed,
            )

        elapsed = time.monotonic() - start_time
        success = proc.returncode == 0

        return RunResult(
            run_id=request.run_id,
            status="success" if success else "failed",
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            error_type=None if success else f"exit_code_{proc.returncode}",
            error_message=None if success else f"Process exited with code {proc.returncode}",
            git_commit=_get_git_commit(),
            container_digest=self._container_digest,
            train_seconds=elapsed,
        )
