# Changelog

## v0.2.0 - 2026-01-04
### Added
- Milestone 2 dataset & labeling pipeline (MPIIGaze subset).
- Synthetic blur-based ordinal labels (0..4) + normalized score (label_bin/4).
- Session-aware output paths (pXX/dayYY) to prevent processed image overwrite.
- Train/val/test split by session_id to avoid leakage.
- RKNN quantization calibration list (relative paths) for portability.

### Fixed
- Resolved relpath collision/overwrite bug in processed generation.
- Added sanity checks: unique relpaths == processed_items.

## v0.1.0
- Kickoff: Windows/Ubuntu/RV1126 connectivity established
- Verified RKNN runtime on RV1126 via `rknn_inference` demo
- Captured baseline latency (~28–31ms avg)

## Unreleased
- Dataset strategy and labeling pipeline
- First lightweight clarity model training + ONNX export
- ONNX → RKNN conversion and on-board validation
