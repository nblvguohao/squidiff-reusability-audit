"""Environment inventory collection for reproducibility.

Must work without assuming Docker, Apptainer, or GPU exist. External commands
time out after 10 seconds. stderr is preserved but does not cause failures.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

EVALUATION_DATE = "2026-07-22"
EXTERNAL_CMD_TIMEOUT = 10  # seconds


def _run_cmd(cmd: list[str]) -> tuple[str | None, str | None]:
    """Run a command with a 10s timeout. Returns (stdout, stderr) or (None, None)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=EXTERNAL_CMD_TIMEOUT,
        )
        return result.stdout.strip() or None, result.stderr.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        return None, None


def _check_tool_exists(name: str) -> bool:
    """Check whether a CLI tool is on PATH."""
    return shutil.which(name) is not None


def _check_gpu_available() -> bool:
    """Check for NVIDIA GPU via nvidia-smi."""
    stdout, _ = _run_cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return stdout is not None and len(stdout) > 0


def _get_disk_usage(path: Path) -> tuple[int, int]:
    """Return (free_bytes, total_bytes) for the filesystem containing path."""
    usage = shutil.disk_usage(path)
    return usage.free, usage.total


def _get_ram_total() -> int:
    """Get total system RAM in bytes. Returns 0 if unavailable."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        meminfo = MEMORYSTATUSEX()
        meminfo.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(meminfo))
        return int(meminfo.ullTotalPhys)
    except Exception:
        return 0


@dataclass
class EnvironmentInventory:
    """Snapshot of the execution environment for reproducibility."""

    evaluation_date: str = EVALUATION_DATE
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    python_implementation: str = field(default_factory=platform.python_implementation)
    platform: str = field(default_factory=platform.platform)
    git_version: str = ""
    disk_free_bytes: int = 0
    disk_total_bytes: int = 0
    docker_available: bool = False
    apptainer_available: bool = False
    gpu_available: bool = False
    gpu_name: str | None = None
    ram_total_bytes: int = 0
    working_directory: str = ""
    python_executable: str = ""


def collect_environment_inventory(workspace: Path | None = None) -> EnvironmentInventory:
    """Collect an environment inventory snapshot.

    All external commands time out after 10 seconds and failures are silently
    recorded as unavailable. Does not assume GPU, Docker, or Apptainer exist.
    """
    if workspace is None:
        workspace = Path.cwd()

    inventory = EnvironmentInventory()

    # Git version
    stdout, _ = _run_cmd(["git", "--version"])
    if stdout:
        inventory.git_version = stdout.replace("git version ", "").strip()

    # Disk
    free, total = _get_disk_usage(workspace)
    inventory.disk_free_bytes = free
    inventory.disk_total_bytes = total

    # Container runtimes
    inventory.docker_available = _check_tool_exists("docker")
    inventory.apptainer_available = _check_tool_exists("apptainer") or _check_tool_exists("singularity")

    # GPU
    inventory.gpu_available = _check_gpu_available()
    if inventory.gpu_available:
        gpu_name, _ = _run_cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        inventory.gpu_name = gpu_name

    # RAM
    inventory.ram_total_bytes = _get_ram_total()

    # Paths
    inventory.working_directory = str(workspace.resolve())
    inventory.python_executable = sys.executable

    return inventory


def write_inventory_markdown(inventory: EnvironmentInventory, output_path: Path) -> None:
    """Write an EnvironmentInventory as a Markdown report."""
    lines = [
        "# Environment Inventory",
        "",
        f"**Generated:** {inventory.evaluation_date}",
        f"**Python:** {inventory.python_version} ({inventory.python_implementation})",
        f"**Platform:** {inventory.platform}",
        "",
        "## Software",
        "",
        "| Tool | Available | Detail |",
        "|------|-----------|--------|",
        f"| Git | ✅ | {inventory.git_version} |",
        f"| Python | ✅ | {inventory.python_version} |",
        f"| Docker | {'✅' if inventory.docker_available else '❌'} | — |",
        f"| Apptainer/Singularity | {'✅' if inventory.apptainer_available else '❌'} | — |",
        "",
        "## Hardware",
        "",
        "| Component | Detail |",
        "|-----------|--------|",
        f"| GPU | {'✅ ' + (inventory.gpu_name or 'available') if inventory.gpu_available else '❌ Not available'} |",
        f"| RAM | {inventory.ram_total_bytes // (1024**3)} GB |",
        f"| Disk free | {inventory.disk_free_bytes // (1024**3)} GB |",
        f"| Disk total | {inventory.disk_total_bytes // (1024**3)} GB |",
        "",
        "## Paths",
        "",
        f"- Working directory: `{inventory.working_directory}`",
        f"- Python executable: `{inventory.python_executable}`",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
