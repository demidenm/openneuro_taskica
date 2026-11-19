# openneuro_taskica

This project measures how much of the brain the FitLins subject-level maps cover in OpenNeuro studies. It finds the voxels that most tasks share.

## What it does

1. **Each task:** Make one binary mask for each subject. A voxel is present if it is finite and not zero. Collapse the runs and sessions of one subject by intersection. Count the masks voxel by voxel. The result is a % overlap map.
2. **All tasks:** Binarize each task map at 70% of subjects. Count the maps across tasks. Report the share of the MNI152NLin2009cAsym 2 mm brain mask that at least 70, 90 and 100% of tasks cover.

## Layout

- `src/openneuro_taskica/parse_openneuro_derivs/`: finds studies, contrasts and map files.
- `src/openneuro_taskica/voxel_overlap/overlap.py`: overlap and coverage functions.
- `notebooks/map_overlap_check.ipynb`: runs the full workflow. Each step has a note about its output.
- `overlap_output/`: results.
  - `overlap_summary.csv`: one row for each task (coverage at 70, 90 and 100%, number of subjects).
  - `subjlvl/`: overlap maps for each task (`*_subject-overlap-percent.nii.gz`, `*_subject-overlap-count.nii.gz`).
  - `groups/`: cross-task maps.
  - `plots/`: figures.

## Setup

```bash
uv sync
```

The notebook reads FitLins outputs from `/oak/stanford/...`. It downloads the MNI mask with `templateflow`. Run the notebook on the cluster.

## Assumptions

- Each task has one contrast. All contrasts of a task use the same subject mask.
- Group-level maps are not used for coverage. They keep voxels that some subjects do not have.
- All maps are in the same standard space.
