# Milestone 2: Dataset Spec

## Task
Eye-image clarity assessment (ordinal classification 0~4).

## Data Units
- Default ROI: eye_roi (224x224 RGB)
- File format: JPG
- Naming: <source>/<class>/<id>.jpg

## Manifest Schema
Columns:
- id
- relpath
- split (train/val/test)
- label (0..4)
- label_type (synthetic/human/none)
- session_id
- baseline_lapvar
- baseline_tenengrad