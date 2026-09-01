"""Squidiff Tier 0 experiment — full pipeline.

parse → split → baselines → Squidiff → metrics → report
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import pandas as pd
from scipy.io import mmread
from scipy.sparse import issparse


def parse_sample_name(filename: str) -> dict:
    """Parse a GSM sample filename into metadata components.

    Examples:
      GSM5736770_D14-NT-raji  → {timepoint: D14, construct: NT, context: raji}
      GSM5736772_CB-NK-IL15   → {timepoint: pre, construct: IL15, context: CB}
    """
    # Remove barcode/feature suffixes
    name = re.sub(r'_(barcodes|features|matrix)\.(tsv|mtx)\.gz$', '', filename)
    # Remove GSM prefix
    name = re.sub(r'^GSM\d+_', '', name)

    parts = name.split('-')

    timepoint = 'pre'
    construct = 'unknown'
    context = 'unknown'

    if parts[0].startswith('CB'):
        timepoint = 'pre'
        # CB-NK-IL15, CB-NK-NT, CB-NK-CARCD19, CB-NK-CARCD19-IL15
        if 'CARCD19' in name and 'IL15' in name:
            construct = 'CAR19_IL15'
        elif 'CARCD19' in name:
            construct = 'CAR19'
        elif 'IL15' in name:
            construct = 'IL15'
        elif 'NT' in name:
            construct = 'NT'
        context = 'cord_blood'
    elif re.match(r'D\d+', parts[0]):
        timepoint = parts[0]
        rest = '-'.join(parts[1:]).upper()
        if 'CARCD19IL15' in rest or ('CARCD19' in rest and 'IL15' in rest):
            construct = 'CAR19_IL15'
        elif 'CARCD19' in rest:
            construct = 'CAR19'
        elif 'NT' in rest:
            construct = 'NT'
        if 'RAJI' in rest:
            context = 'raji'
        elif 'IL2' in rest:
            context = 'IL2'
        else:
            context = 'tumor'

    return {
        'timepoint_raw': timepoint,
        'construct': construct,
        'context': context,
    }


TIMEPOINT_ORDER: dict[str, int] = {
    'pre': 0, 'D7': 7, 'D14': 14, 'D21': 21, 'D28': 28,
}


def build_anndata_from_mtx(raw_dir: Path, output_path: Path) -> Path:
    """Build a combined AnnData from the MTX files in raw_dir.

    Returns the path to the saved .h5ad file.
    """
    try:
        import anndata as ad
    except ImportError:
        print("anndata/scanpy not available; cannot build AnnData")
        return output_path

    adatas = []
    raw_files = sorted(raw_dir.glob('GSM*_barcodes.tsv.gz'))

    for barcode_file in raw_files:
        prefix = str(barcode_file).replace('_barcodes.tsv.gz', '')
        matrix_file = Path(prefix + '_matrix.mtx.gz')
        feature_file = Path(prefix + '_features.tsv.gz')

        if not matrix_file.exists() or not feature_file.exists():
            continue

        # Parse metadata
        meta = parse_sample_name(barcode_file.name)

        # Read MTX (10x format: features × cells → transpose to cells × features)
        with gzip.open(matrix_file, 'rb') as fh:
            mat = mmread(fh)
        if issparse(mat):
            mat = mat.tocsc().T.tocsr()
        else:
            mat = mat.T

        # Read barcodes
        barcodes = pd.read_csv(barcode_file, sep='\t', header=None, names=['barcode'])
        # Read features
        features = pd.read_csv(feature_file, sep='\t', header=None, names=['gene_id', 'gene_name', 'feature_type'])

        # Filter to gene expression only (features are now columns after transpose)
        gene_mask = (features['feature_type'] == 'Gene Expression').values
        if gene_mask.sum() > 0:
            mat = mat[:, gene_mask]
            features = features.loc[gene_mask]

        adata = ad.AnnData(
            X=mat,
            obs=pd.DataFrame(index=barcodes['barcode'].values),
            var=pd.DataFrame(index=features['gene_name'].values),
        )

        # Add metadata
        adata.obs['sample_id'] = Path(prefix).name
        adata.obs['timepoint_raw'] = meta['timepoint_raw']
        adata.obs['timepoint_numeric'] = TIMEPOINT_ORDER.get(meta['timepoint_raw'], -1)
        adata.obs['construct'] = meta['construct']
        adata.obs['context'] = meta['context']
        adata.obs['species'] = 'mus_musculus'

        # Ensure unique gene names
        adata.var_names_make_unique()
        adatas.append(adata)
        print(f"  Loaded: {Path(prefix).name} → {adata.n_obs} cells x {adata.n_vars} genes")

    if not adatas:
        print("No samples loaded!")
        return output_path

    # Concatenate (use common genes via inner join)
    combined = ad.concat(adatas, join='inner', index_unique='-')

    # Fill required contract fields
    combined.obs['cell_id'] = combined.obs.index.values
    combined.obs['dataset_id'] = 'GSE190976'
    combined.obs['donor_id'] = combined.obs['sample_id'].apply(
        lambda x: 'mouse_' + x.split('_')[0] if '_' in str(x) else 'mouse_unknown'
    )
    combined.obs['engineering_state'] = combined.obs['construct'].map(
        lambda c: 'CAR19_IL15' if 'IL15' in str(c) and 'CAR' in str(c)
        else 'CAR19' if 'CAR' in str(c)
        else 'NT'
    )
    combined.obs['stimulation_state'] = combined.obs['context'].map(
        lambda c: 'raji_stimulated' if 'raji' in str(c).lower() else 'unstimulated'
    )
    combined.obs['tumor_context'] = combined.obs['context'].map(
        lambda c: 'tumor' if 'raji' in str(c).lower() else 'no_tumor'
    )
    combined.obs['split_group'] = 'unassigned'
    combined.obs['endpoint_label'] = 'NA'

    # Ensure string columns are strings, not categories (for contract validation)
    for col in combined.obs.columns:
        if combined.obs[col].dtype.name == 'category':
            combined.obs[col] = combined.obs[col].astype(str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(output_path)
    print(f"\nCombined AnnData: {combined.n_obs} cells x {combined.n_vars} genes")
    print(f"Saved to: {output_path}")
    return output_path


if __name__ == '__main__':
    RAW = Path('data/raw')
    PROCESSED = Path('artifacts/squidiff_tier0/source_data')
    PROCESSED.mkdir(parents=True, exist_ok=True)

    output = PROCESSED / 'gse190976_combined.h5ad'
    build_anndata_from_mtx(RAW, output)
