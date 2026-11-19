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
    node: Union[str, List[str]] = None,
) -> Dict[Tuple[str, str], List[str]]:
    """
    Find statistical maps for a selected contrast.

    Parameters
    ----------
    node : str or list of str, optional
        FitLins node level(s) to search.

        Options:
            "node-subjectLevel"
            "node-runLevel"
            "node-dataLevel"

        If None, searches subjectLevel and runLevel.
    """

    groups: Dict[Tuple[str, str], List[str]] = {}

    for study, task_contrasts in contrasts.items():
        for task, contrast in task_contrasts.items():

            task_dir = Path(study_roots[study]) / f"task-{task}"

            # Select node levels
            if node is None:
                nodes = [
                    "node-subjectLevel",
                    "node-runLevel",
                ]
            elif isinstance(node, str):
                nodes = [node]
            else:
                nodes = list(node)

            files = []

            for n in nodes:
                folder = task_dir / n

                if folder.is_dir():
                    files.extend(
                        str(f)
                        for f in folder.rglob(
                            f"*contrast-{contrast}_*stat-{stat}_*"
                        )
                        if f.name.endswith((".nii", ".nii.gz"))
                    )

            if files:
                groups[(study, task)] = sorted(files)

    return groups
