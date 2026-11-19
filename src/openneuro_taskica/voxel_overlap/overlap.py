import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

SUB_RE = re.compile(r"(sub-[^_/]+)")


def statmap_mask(f):
    """
    Create a binary mask of finite, non-zero voxels in a statistical map.
    """
    img = nib.load(f)
    data = np.asarray(img.dataobj)

    return nib.Nifti1Image(
        (np.isfinite(data) & (data != 0)).astype(np.uint8),
        img.affine,
        img.header,
    )

def get_data_level(filepath):
    path = str(filepath).lower()

    if "node-datalevel" in path:
        return "group"

    if SUB_RE.search(path):
        return "subject"

    raise ValueError(f"Could not identify data level: {filepath}")

    
def get_binary_masks(files, ref_img=None):
    """
    Load files and return a list of Boolean arrays and a reference image.

    Raises ValueError if any image shape or affine differs from reference.
    """
    if not files:
        raise ValueError("No input files provided.")

    imgs = [nib.load(f) for f in files]
    ref = ref_img if ref_img is not None else imgs[0]

    arrays = []
    for img in imgs:
        if img.shape != ref.shape:
            raise ValueError("Image dimensions do not match reference.")

        if not np.allclose(img.affine, ref.affine, atol=1e-5):
            raise ValueError("Image affines do not match reference.")

        data = np.asarray(img.dataobj)
        arrays.append(np.isfinite(data) & (data != 0))

    return arrays, ref


def compute_overlap(masks, ref_img):
    """
    Calculate voxel-wise overlap across subjects.

    Returns:
        count_img: Number of subjects containing each voxel.
        percent_img: Percentage of subjects containing each voxel.
    """
    count = np.sum(masks, axis=0)
    percent = 100.0 * count / len(masks)

    count_img = nib.Nifti1Image(
        count.astype(np.int16),
        ref_img.affine,
        ref_img.header,
    )
    percent_img = nib.Nifti1Image(
        percent.astype(np.float32),
        ref_img.affine,
        ref_img.header,
    )

    return count_img, percent_img


def overlap_summary(percent_img, thresholds=(60, 65, 75)):
    """
    Calculate the percentage of voxels within the union
    where at least the specified percentage of subjects
    have data.

    The union includes voxels present in at least one subject.
    """
    pct = percent_img.get_fdata()
    union = pct > 0
    n_union = union.sum()

    results = {}
    for thr in thresholds:
        n_covered = np.sum((pct >= thr) & union)
        results[f"pct_of_union_ge{thr}"] = (
            100.0 * n_covered / n_union if n_union else np.nan
        )

    return results


def coverage_from_percent(
    percent_img,
    thresholds=(70, 90, 100),
    brain_mask=None,
):
    """
    Calculate coverage relative to the whole-brain mask.

    If brain_mask is supplied:
        Uses the anatomical brain mask as denominator.

    If brain_mask is None:
        Uses the union of voxels present in at least one subject.

    Returns percentage of the denominator covered by
    at least the specified percentage of subjects.
    """
    pct = percent_img.get_fdata()

    if brain_mask is not None:
        if brain_mask.shape != percent_img.shape:
            raise ValueError("Brain mask dimensions do not match percentage map.")

        if not np.allclose(brain_mask.affine, percent_img.affine, atol=1e-5):
            raise ValueError("Brain mask affine does not match percentage map.")

        brain = brain_mask.get_fdata() > 0
    else:
        brain = pct > 0

    n_brain = brain.sum()

    results = {}
    for thr in thresholds:
        n_covered = np.sum((pct >= thr) & brain)
        results[f"pct_of_brain_ge{thr}"] = (
            100.0 * n_covered / n_brain if n_brain else np.nan
        )

    return results



def run_overlap(
    groups,
    out_dir,
    ref_img=None,
    collapse_sessions=True,
    brain_mask=None,
):
    """
    Calculate voxel-wise overlap across subjects or studies.

    Subject-level:
        Each subject contributes one observation.
        Multiple sessions can be collapsed within subjects.

    Group-level:
        Each study contributes one observation.
        Group-level maps are treated as independent observations.

    Both use the same overlap and coverage calculations.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    if brain_mask is not None and isinstance(brain_mask, (str, Path)):
        brain_mask = nib.load(brain_mask)

    for (ds, task), files in groups.items():

        print(f"\nProcessing {ds}/{task}")

        # Separate files by data level
        by_level = defaultdict(list)

        for f in files:
            level = get_data_level(f)
            by_level[level].append(f)

        # Process subject- and group-level data identically
        for level, level_files in by_level.items():

            print(f"Processing {level}: {len(level_files)} files")

            # Load binary masks
            masks, ref = get_binary_masks(
                level_files,
                ref_img=ref_img,
            )

            # -----------------------------------------
            # Determine independent observations
            # -----------------------------------------

            if level == "subject":

                if collapse_sessions:
                    by_observation = defaultdict(list)

                    for f, m in zip(level_files, masks):
                        match = SUB_RE.search(str(f))

                        if match is None:
                            raise ValueError(
                                f"Could not identify subject in {f}"
                            )

                        by_observation[match.group(1)].append(m)

                    # Intersection across sessions
                    masks = [
                        np.logical_and.reduce(ms)
                        for ms in by_observation.values()
                    ]

                n_observations = len(masks)
                observation_type = "subjects"

            elif level == "group":

                # Each group-level map is one observation.
                n_observations = len(masks)
                observation_type = "studies"

            else:
                raise ValueError(f"Unknown data level: {level}")

            if n_observations < 2:
                print(
                    f"[warn] {ds}/{task}/{level}: "
                    f"fewer than 2 {observation_type}, skipping"
                )
                continue

            print(
                f"{ds}/{task}/{level}: "
                f"{n_observations} {observation_type}"
            )

            # -----------------------------------------
            # Calculate voxel-wise overlap
            # -----------------------------------------

            count_img, percent_img = compute_overlap(masks, ref)

            # Save NIfTI maps
            nib.save(
                percent_img,
                out_dir / (
                    f"{ds}_task-{task}_{level}-overlap-percent.nii.gz"
                ),
            )

            nib.save(
                count_img,
                out_dir / (
                    f"{ds}_task-{task}_{level}-overlap-count.nii.gz"
                ),
            )

            # -----------------------------------------
            # Calculate summary statistics
            # -----------------------------------------

            summary = overlap_summary(
                percent_img,
                thresholds=(70, 90, 100),
            )

            coverage = coverage_from_percent(
                percent_img,
                thresholds=(70, 90, 100),
                brain_mask=brain_mask,
            )

            # -----------------------------------------
            # Store results
            # -----------------------------------------

            rows.append({
                "study": ds,
                "task": task,
                "data_level": level,
                "observation_type": observation_type,
                "n_observations": n_observations,
                **summary,
                **coverage,
            })

    return pd.DataFrame(rows)
