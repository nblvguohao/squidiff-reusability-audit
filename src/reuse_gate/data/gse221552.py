"""Parser for GSE221552: CITE-seq of CAR-NK cells for AML immunotherapy.

Dataset: CITE-seq of CAR33-KLRC1ko-NK cells against acute myeloid leukemia.
Species: Homo sapiens
Samples: 7 samples (GSM6884089–GSM6884095)
Constructs: CAR33-NK, CAR33-KLRC1ko-NK
Technology: CITE-seq (single-cell multi-omics)
"""

from __future__ import annotations

# --- Explicit construct mapping ---
CONSTRUCT_MAPPING: dict[str, str] = {
    "car33-nk": "CAR33_NK",
    "car33_nk": "CAR33_NK",
    "car33": "CAR33_NK",
    "car33-klrc1ko": "CAR33_KLRC1ko",
    "car33_klrc1ko": "CAR33_KLRC1ko",
    "car33-klrc1ko-nk": "CAR33_KLRC1ko",
    "car33_klrc1ko_nk": "CAR33_KLRC1ko",
    "klrc1ko": "KLRC1ko",
}

ENGINEERING_STATE_MAPPING: dict[str, str] = {
    "car33-nk": "CAR33_wildtype",
    "car33_nk": "CAR33_wildtype",
    "car33-klrc1ko": "CAR33_KLRC1_knockout",
    "car33_klrc1ko": "CAR33_KLRC1_knockout",
}

# --- Full dataset metadata ---
GSE221552_METADATA: dict[str, object] = {
    "accession": "GSE221552",
    "title": "CITE-seq of CAR-NK cells for AML immunotherapy",
    "species": "homo_sapiens",
    "n_samples": 7,
    "technology": "CITE-seq (single-cell multi-omics with antibody-derived tags)",
    "construct_mapping": CONSTRUCT_MAPPING,
    "constructs_available": sorted(set(CONSTRUCT_MAPPING.values())),
    "engineering_states": sorted(set(ENGINEERING_STATE_MAPPING.values())),
    "sample_year": 2024,
    "notes": (
        "This dataset is used as external support for construct/stimulation analysis. "
        "Results must be reported separately from the GSE190976 longitudinal endpoint."
    ),
}


def parse_engineering_state(label: str) -> str:
    """Parse an engineering state label to a canonical form.

    Returns a string like 'CAR33_wildtype' or 'CAR33_KLRC1_knockout'.
    """
    key = label.strip().lower().replace(" ", "_").replace("-", "_")
    if key in ENGINEERING_STATE_MAPPING:
        return ENGINEERING_STATE_MAPPING[key]
    # Try construct mapping as fallback
    if key in CONSTRUCT_MAPPING:
        construct = CONSTRUCT_MAPPING[key]
        if "klrc1" in construct.lower():
            return "CAR33_KLRC1_knockout"
        return "CAR33_wildtype"
    return f"unknown:{key}"
