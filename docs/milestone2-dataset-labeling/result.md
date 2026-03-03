# Milestone 2 Results — Dataset + Labeling Pipeline

## Dataset
- Source: MPIIGaze (subset)
- Subset: p00–p04, total raw images: 10,000

## Labeling Strategy
- 5-level ordinal clarity labels: 0..4 (very_blurry → very_sharp)
- Inference score: label_bin/4 or softmax expected value later
- Synthetic blur augmentation (Gaussian blur sigma ranges per class)

## Outputs (Sanity)
- processed_items: 50,000
- unique_relpaths: 50,000 (no overwrite/collision)
- split by session_id (pXX/dayYY), sessions: 233
- split sizes:
  - train: 40,235
  - val: 5,270
  - test: 4,495

## Artifacts
- manifests: generated locally (not committed by default)
- calib_list.txt: relative paths for RKNN quantization
- splits.json: session lists to prevent leakage

## Reproduce
1. Place MPIIGaze subset under `host/data/raw/mpiigaze_subset/`
2. Run:
   - `python -u host/scripts/prepare_data.py --config host/src/configs/dataset.yaml`