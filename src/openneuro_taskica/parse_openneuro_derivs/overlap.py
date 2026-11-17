"""Voxel-coverage overlap across stat maps."""

import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image

SUB_RE = re.compile(r"(sub-[^_/]+)")


def statmap_mask(f) -> Nifti1Image:
    """Binary mask (uint8) of finite, non-zero voxels in a stat map."""
    img = nib.load(f)
    data = np.asarray(img.dataobj)
    return nib.Nifti1Image((np.isfinite(data) & (data != 0)).astype(np.uint8),
                           img.affine)


def get_binary_masks(files, ref_img=None):
    """Load files and return (list of boolean arrays, reference image).

    Raises ValueError if any image's shape or affine differs from the reference.
    """
    imgs = [nib.load(f) for f in files]
    ref = ref_img if ref_img is not None else imgs[0]
    arrays = []

    for img in imgs:
        if img.shape != ref.shape:
            raise ValueError("Image dimensions do not match reference")

        if not np.allclose(img.affine, ref.affine, atol=1e-5):
            raise ValueError("Image affines do not match reference")

        data = np.asarray(img.dataobj)
        arrays.append(np.isfinite(data) & (data != 0))

    return arrays, ref


def compute_overlap(masks, ref_img):
    """Return (count image, percent image) across masks."""
    count = np.sum(masks, axis=0)
    percent = 100.0 * count / len(masks)
    return (
        nib.Nifti1Image(count.astype(np.int16), ref_img.affine),
        nib.Nifti1Image(percent.astype(np.float32), ref_img.affine),
    )


def overlap_summary(masks):
    """Summary stats: union, intersection, and intersection as % of union / mean mask."""
    stack = np.stack(masks)
    union = stack.any(axis=0).sum()
    inter = stack.all(axis=0).sum()
    mean_size = stack.sum(axis=(1, 2, 3)).mean()
    return {
        "n_masks": len(masks),
        "union_voxels": int(union),
        "intersection_voxels": int(inter),
        "pct_all_of_union": 100.0 * inter / union if union else float("nan"),
        "pct_all_of_mean_mask": 100.0 * inter / mean_size if mean_size else float("nan"),
    }


def run_overlap(groups, out_dir, ref_img=None, collapse_sessions=True):
    """
    groups: {(ds, task): [files]}
    collapse_sessions: if True, combine multiple files per subject (e.g. ses-1,
        ses-2) into one subject-level mask (voxel kept only if present in ALL
        of that subject's files), so overlap is computed across subjects.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for (ds, task), files in groups.items():
        if len(files) < 2:
            print(f"[warn] {ds}/{task}: {len(files)} file(s), skipping")
            continue

        masks, ref = get_binary_masks(files, ref_img=ref_img)

        if collapse_sessions:
            by_sub = defaultdict(list)
            for f, m in zip(files, masks):
                by_sub[SUB_RE.search(str(f)).group(1)].append(m)
            masks = [np.logical_and.reduce(ms) for ms in by_sub.values()]

        print(f"{ds}/{task}: {len(files)} files -> {len(masks)} masks")

        count_img, percent_img = compute_overlap(masks, ref)
        nib.save(percent_img, out_dir / f"{ds}_task-{task}_overlap-percent.nii.gz")
        nib.save(count_img,   out_dir / f"{ds}_task-{task}_overlap-count.nii.gz")

        rows.append({"study": ds, "task": task, **overlap_summary(masks)})

    return pd.DataFrame(rows)
