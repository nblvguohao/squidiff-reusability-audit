# Squidiff compatibility patches

Upstream: <https://github.com/siyuh/Squidiff> pinned at `abdfc27d84947dcccd745d1067c0840a41d32eb8` (v1.0.8)
Paper: He et al., *Nature Methods* (2025), DOI [10.1038/s41592-025-02877-y](https://doi.org/10.1038/s41592-025-02877-y)

Three defects block end-to-end training on the upstream `use_encoder=True` +
`class_cond=True` code path. Each raises an unrecoverable `RuntimeError` on the
first optimizer step, so the path cannot have been exercised end-to-end upstream.
They surface in a fixed order — fixing one exposes the next — which is why they
are recorded as three separate patches rather than one.

No scientific behaviour is altered. Each patch changes dtype, device placement or
tensor rank only; none touches the diffusion process, the loss, the sampler, or
any hyperparameter.

| Patch | File | Defect | Symptom |
|-------|------|--------|---------|
| `0001-group-label-dtype.patch` | `Squidiff/scrna_datasets.py` | Conditioning labels leave the dataset as a raw NumPy array | `RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Float` |
| `0002-group-label-device.patch` | `Squidiff/train_util.py` | Conditioning labels are not moved to the compute device | `RuntimeError: Expected all tensors to be on the same device, but got mat1 is on cpu, different from other tensors on cuda:0` |
| `0003-label-embed-shape.patch` | `Squidiff/MLPModel.py` | `label_embed` expects rank-2 input but receives rank-1 | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x64 and 1x2048)` |

## 0001 — conditioning label dtype

`AnnDataDataset.__init__` assigns `adata.obs['Group'].copy().values` directly to
`self.encoded_obs_tensor`. For integer class labels pandas yields an `int64`
array, which the default DataLoader collate converts to a `torch.int64` tensor.
`EncoderMLPModel.label_embed` is an `nn.Linear`, so `F.linear` rejects the Long
input against Float weights.

The upstream file already contains the correct call, commented out on the line
immediately above (`#self.encoded_obs_tensor = torch.tensor(..., dtype=torch.float32)`),
in the `use_drug_structure=True` branch. The patch restores it in both branches.

The patch also drops trailing whitespace on one blank line inside the same hunk;
this is incidental to the diff and carries no behavioural meaning.

## 0002 — conditioning label device placement

`TrainLoop.forward_backward` moves `batch['feature']`, `batch['drug_dose']` and
`batch['control_feature']` to `dist_util.dev()`, but leaves `batch['group']` on
the host. The model has already been moved to the GPU, so the first
`label_embed` call mixes a CPU input with CUDA weights.

The defect is invisible on a CPU-only run, because host and compute device
coincide. It is therefore reachable only when the model actually trains on a GPU
— consistent with a code path that has not been run end-to-end.

## 0003 — label embedding rank

`EncoderMLPModel.label_embed` is `nn.Linear(1, hidden_sizes)`, so it requires
input of shape `(batch, 1)`. Labels arrive from the DataLoader with shape
`(batch,)`. The patch inserts a rank check and unsqueezes only when needed, so
callers that already pass rank-2 labels are unaffected.

## Applying

The patches are `git diff` output against the pinned commit and apply in order:

```bash
cd vendor/Squidiff
git checkout abdfc27d84947dcccd745d1067c0840a41d32eb8
cd ../..
git apply vendor/patches/squidiff/0001-group-label-dtype.patch
git apply vendor/patches/squidiff/0002-group-label-device.patch
git apply vendor/patches/squidiff/0003-label-embed-shape.patch
```

To confirm the working tree already carries them:

```bash
git apply --check --reverse vendor/patches/squidiff/*.patch
```

## Regression tests

`tests/regression/test_squidiff_patches.py` guards all three. The dtype and rank
patches are verified functionally on CPU. The device patch cannot fail on a
CPU-only host, so it is guarded by a source-level assertion plus a
`@pytest.mark.gpu` end-to-end training step that reproduces the original failure
when the patch is reverted.
