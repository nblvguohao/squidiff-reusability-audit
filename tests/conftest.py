"""Shared pytest fixtures for the biomedical reusability gate test suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def tmp_h5ad(tmp_path):
    """Create a minimal temporary AnnData file path (no actual I/O)."""
    return tmp_path / "test.h5ad"


@pytest.fixture
def tmp_config(tmp_path):
    """Create a minimal temporary config YAML file path."""
    return tmp_path / "config.yaml"
