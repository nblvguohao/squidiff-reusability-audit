"""Unit tests for environment inventory collection — RED phase."""

from reuse_gate.provenance import collect_environment_inventory


def test_inventory_contains_required_fields():
    """EnvironmentInventory must capture all required reproducibility fields."""
    inventory = collect_environment_inventory()
    assert inventory.python_version
    assert inventory.platform
    assert inventory.git_version
    assert inventory.disk_free_bytes >= 0
    assert inventory.docker_available in {True, False}
    assert inventory.apptainer_available in {True, False}


def test_inventory_fields_have_expected_types():
    """All fields must have sensible types."""
    inventory = collect_environment_inventory()
    assert isinstance(inventory.python_version, str)
    assert isinstance(inventory.platform, str)
    assert isinstance(inventory.git_version, str)
    assert isinstance(inventory.disk_free_bytes, int)
    assert isinstance(inventory.disk_total_bytes, int)
    assert isinstance(inventory.docker_available, bool)
    assert isinstance(inventory.apptainer_available, bool)
    assert isinstance(inventory.gpu_available, bool)
    assert isinstance(inventory.ram_total_bytes, int)
    assert isinstance(inventory.evaluation_date, str)


def test_evaluation_date_matches_plan():
    """Evaluation date must be fixed at 2026-07-22 per the plan."""
    inventory = collect_environment_inventory()
    assert inventory.evaluation_date == "2026-07-22"
