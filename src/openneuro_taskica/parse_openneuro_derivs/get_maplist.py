"""Locate FitLins outputs on disk: studies, tasks, contrasts, and stat maps."""

import re
from pathlib import Path
from typing import Dict, List, Mapping, Tuple, Union

CONTRAST_RE = re.compile(r"contrast-([^_/]+)")


def build_study_roots(fitlins_deriv_out):
    """Return {ds_name: path} for every ds* folder in fitlins_deriv_out."""
    fitlins_deriv_out = Path(fitlins_deriv_out)
    return {
        p.name: p
        for p in sorted(fitlins_deriv_out.glob("ds*"))
        if p.is_dir()
    }


def list_contrasts(task_dir):
    """Sorted unique contrast names found in *stat-effect_* maps under a task folder."""
    found = set()
    for f in Path(task_dir).rglob("*stat-effect_*"):
        m = CONTRAST_RE.search(f.name)
        if m:
            found.add(m.group(1))
    return sorted(found)


def build_contrasts(study_roots, n=1):
    """
    For each study and each task-* folder, pick the n-th contrast (1-indexed).
    Returns {ds_name: {task_name: contrast}}.
    If a task has fewer than n contrasts, it is skipped with a warning.
    """
    if n < 1:
        raise ValueError("n must be >= 1 (1 = first contrast)")

    contrasts = {}
    for ds_name, root in study_roots.items():
        contrasts[ds_name] = {}
        for task_dir in sorted(Path(root).glob("task-*")):
            if not task_dir.is_dir():
                continue
            task = task_dir.name.removeprefix("task-")
            available = list_contrasts(task_dir)
            if len(available) >= n:
                contrasts[ds_name][task] = available[n - 1]
            else:
                print(f"[warn] {ds_name}/{task}: only {len(available)} contrast(s) found, "
                      f"cannot pick #{n}")
    return contrasts


def find_contrast_files(
    study_roots: Mapping[str, Union[str, Path]],
    contrasts: Mapping[str, Mapping[str, str]],
    stat: str = "effect",
    node: str = "node-subjectLevel",
) -> Dict[Tuple[str, str], List[str]]:
    """Find stat maps for a contrast specified per study and per task.

    Searches only ``<study_root>/task-<task>/<node>``, which avoids walking
    the whole study tree.

    Args:
        study_roots: Mapping ``{study_name: FitLins output directory}``.
        contrasts: Nested mapping ``{study_name: {task_label: contrast_label}}``.
        stat: Statistic label (value after ``stat-``), e.g. ``effect`` or ``t``.
        node: Name of the FitLins node directory holding the maps.

    Returns:
        Mapping ``{(study, task): sorted file paths}``. Pairs whose directory
        is missing or has no matching files are omitted.
    """
    groups: Dict[Tuple[str, str], List[str]] = {}
    for study, task_contrasts in contrasts.items():
        for task, contrast in task_contrasts.items():
            folder = Path(study_roots[study]) / f"task-{task}" / node
            files = sorted(
                str(f)
                for f in folder.rglob(f"*contrast-{contrast}_*stat-{stat}_*")
                if f.name.endswith((".nii", ".nii.gz"))
            )
            if files:
                groups[(study, task)] = files
    return groups
