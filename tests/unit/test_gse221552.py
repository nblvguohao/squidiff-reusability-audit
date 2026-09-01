"""Unit tests for GSE221552 CAR-NK CITE-seq parser — RED phase."""

from reuse_gate.data.gse221552 import (
    GSE221552_METADATA,
    parse_engineering_state,
)


def test_parse_engineering_state_identifies_knockout():
    """KLRC1 knockout must be identified."""
    assert "ko" in parse_engineering_state("CAR33-KLRC1ko").lower()
    assert "car33" in parse_engineering_state("CAR33-KLRC1ko").lower()


def test_parse_engineering_state_identifies_wildtype():
    """Wild-type CAR33 must be identified."""
    result = parse_engineering_state("CAR33-NK")
    assert "car33" in result.lower()


def test_gse221552_metadata_is_complete():
    """Metadata must contain required fields."""
    assert GSE221552_METADATA["accession"] == "GSE221552"
    assert GSE221552_METADATA["species"] == "homo_sapiens"
    assert len(GSE221552_METADATA["construct_mapping"]) >= 2
