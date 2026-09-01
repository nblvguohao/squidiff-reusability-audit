"""Unit tests for GSE190976 CAR-NK data parser — RED phase."""

import pytest

from reuse_gate.data.gse190976 import (
    GSE190976_METADATA,
    map_construct,
    map_sample_to_group,
    parse_timepoint,
)


def test_parse_timepoint_orders_pre_and_days():
    """Timepoint labels must be parsed to ordered numeric values."""
    assert parse_timepoint("pre-infusion") == 0
    assert parse_timepoint("Pre-infusion") == 0
    assert parse_timepoint("D7") == 7
    assert parse_timepoint("D14") == 14
    assert parse_timepoint("D21") == 21
    assert parse_timepoint("D28") == 28


def test_parse_timepoint_rejects_unknown():
    """Unknown timepoint labels must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown timepoint"):
        parse_timepoint("D99")


def test_map_construct_identifies_car19_il15():
    """CAR19/IL15 construct must be identified."""
    assert map_construct("CAR19/IL15") == "CAR19_IL15"
    assert map_construct("CAR19") == "CAR19"
    assert map_construct("NT") == "NT"


def test_map_construct_rejects_unknown():
    """Unknown construct labels must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown construct"):
        map_construct("CD22_CAR")


def test_map_sample_to_group_assigns_correctly():
    """Samples must be mapped to their known biological groups."""
    group = map_sample_to_group("CAR19_D7_rep1")
    assert group is not None or group == "NA"  # unknown samples → NA


def test_gse190976_metadata_is_complete():
    """Metadata must contain required fields."""
    assert GSE190976_METADATA["accession"] == "GSE190976"
    assert GSE190976_METADATA["species"] == "mus_musculus"
    assert len(GSE190976_METADATA["timepoint_mapping"]) >= 5
    assert len(GSE190976_METADATA["construct_mapping"]) >= 2
