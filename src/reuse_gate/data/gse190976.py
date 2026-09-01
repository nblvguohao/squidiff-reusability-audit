"""Parser for GSE190976: CAR-NK scRNA-seq in Raji lymphoma mouse model.

Dataset: The interaction of CAR-NK cell and tumor cells during in vivo
anti-tumor cell therapy.
Reference: Li L, Rezvani K et al., PMID 35534898
Species: Mus musculus
Samples: 18 BioSamples, 301 Gbases
Timepoints: pre-infusion, D7, D14, D21, D28, D35
Constructs: CAR19, CAR19/IL15, NT (non-transduced)
"""

from __future__ import annotations

# --- Explicit timepoint mapping ---
# Never infer time ordering lexicographically.
TIMEPOINT_MAPPING: dict[str, int] = {
    "pre-infusion": 0,
    "pre": 0,
    "pre_infusion": 0,
    "d0": 0,
    "day0": 0,
    "d7": 7,
    "day7": 7,
    "d14": 14,
    "day14": 14,
    "d21": 21,
    "day21": 21,
    "d28": 28,
    "day28": 28,
    "d35": 35,
    "day35": 35,
}

# --- Explicit construct mapping ---
CONSTRUCT_MAPPING: dict[str, str] = {
    "car19": "CAR19",
    "car19/il15": "CAR19_IL15",
    "car19-il15": "CAR19_IL15",
    "car19_il15": "CAR19_IL15",
    "nt": "NT",
    "non-transduced": "NT",
    "non_transduced": "NT",
    "wildtype": "NT",
    "wt": "NT",
}

# Known sample groups from the study
SAMPLE_GROUPS: dict[str, str] = {
    # Format: sample_id -> group_label
    # Pre-infusion
    "car19_pre": "CAR19_pre",
    "car19_il15_pre": "CAR19_IL15_pre",
    "nt_pre": "NT_pre",
    # Early response (D7)
    "car19_d7": "CAR19_D7",
    "car19_il15_d7": "CAR19_IL15_D7",
    "nt_d7": "NT_D7",
    # Mid response (D14)
    "car19_d14": "CAR19_D14",
    "car19_il15_d14": "CAR19_IL15_D14",
    "nt_d14": "NT_D14",
    # Late response (D21)
    "car19_d21": "CAR19_D21",
    "car19_il15_d21": "CAR19_IL15_D21",
    "nt_d21": "NT_D21",
    # Relapse phase (D28)
    "car19_d28": "CAR19_D28",
    "car19_il15_d28": "CAR19_IL15_D28",
    "nt_d28": "NT_D28",
}

# --- Full dataset metadata ---
GSE190976_METADATA: dict[str, object] = {
    "accession": "GSE190976",
    "title": "CAR-NK cell and tumor cell interaction during in vivo anti-tumor therapy",
    "species": "mus_musculus",
    "pmid": "35534898",
    "doi": "10.1038/s41591-022-01859-1",
    "n_biosamples": 18,
    "technology": "scRNA-seq (10x Genomics)",
    "timepoint_mapping": TIMEPOINT_MAPPING,
    "construct_mapping": CONSTRUCT_MAPPING,
    "timepoints_available": sorted(set(TIMEPOINT_MAPPING.values())),
    "constructs_available": sorted(set(CONSTRUCT_MAPPING.values())),
    "sample_year": 2021,
}


def parse_timepoint(label: str) -> int:
    """Parse a timepoint label to a numeric day value.

    Raises ValueError for unknown labels.
    """
    key = label.strip().lower().replace(" ", "_").replace("-", "_")
    if key in TIMEPOINT_MAPPING:
        return TIMEPOINT_MAPPING[key]
    raise ValueError(f"Unknown timepoint: '{label}' (normalized: '{key}')")


def map_construct(label: str) -> str:
    """Map a construct label to its canonical form.

    Raises ValueError for unknown labels.
    """
    key = label.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "/")
    # Try exact match first
    if key in CONSTRUCT_MAPPING:
        return CONSTRUCT_MAPPING[key]
    # Try without special chars
    simple = key.replace("/", "_").replace("-", "_")
    if simple in CONSTRUCT_MAPPING:
        return CONSTRUCT_MAPPING[simple]
    raise ValueError(f"Unknown construct: '{label}' (normalized: '{key}')")


def map_sample_to_group(sample_id: str) -> str:
    """Map a sample identifier to its known biological group.

    Returns 'NA' for unknown samples (never empty string).
    """
    key = sample_id.strip().lower().replace(" ", "_").replace("-", "_")
    return SAMPLE_GROUPS.get(key, "NA")
