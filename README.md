# capstone-eye-clarity

Embedded eye-image clarity assessment on **Rockchip RV1126** (NPU) using the **RKNN** runtime.  
This repository follows a strict engineering layout to keep **training → conversion → deployment → benchmarking** reproducible.

## Project Goal

Build an **eye image clarity classifier/scorer** that can run on RV1126 (edge device), and compare:
- **AI model** (RKNN / NPU inference)
- **Traditional focus/clarity metrics** (e.g., Laplacian / Tenengrad)

Optional (bonus): integrate results into a GUI layer (Qt) without coupling UI to the inference core.

---

## Current Status

-  Windows ↔ Ubuntu VM ↔ RV1126 connectivity established (SSH + file transfer)
-  RKNN runtime verified on RV1126 (Buildroot / ARMv7l)
-  On-board inference demo runs with a stable baseline (reported ~28–31 ms avg on vendor demo)

See: `docs/roadmap.md` and `docs/milestone1-kickoff/`

---

## Repository Layout

capstone-eye-clarity/
docs/ # roadmap, milestones, reports, figures, notes
host/ # training + onnx export + rknn conversion + offline eval (PC/Ubuntu)
target/ # RV1126 deployment + runtime inference + app integration
bench/ # traditional algorithms + unified evaluation scripts
models/ # model metadata + small demos (large artifacts via Release/LFS)
third_party/ # dependency notes/patches only
tools/ # optional tooling helpers

---

## Quick Start

### A) Host (training / conversion)

Recommended:
- Ubuntu 20.04 (VM is fine)
- Python 3.8+ (RKNN toolkit compatibility depends on your toolkit wheel)
- Keep datasets and heavy artifacts out of Git (`host/data/raw`, `host/outputs`, etc.)

Typical flow (to be implemented/expanded under `host/scripts/`):
1. Prepare data: `host/scripts/prepare_data.py`
2. Train model: `host/scripts/train.py`
3. Export ONNX: `host/scripts/export_onnx.py`
4. Convert to RKNN: `host/scripts/rknn_convert.py`
5. Evaluate: `host/scripts/eval.py`

> The goal is to make every step runnable from `host/scripts/` and configurable from `host/configs/`.

### B) Target (RV1126 on-board verification)

On RV1126 (vendor image) you can validate that NPU inference works via:
- `/rockchip_test/npu/rknn_inference`

Example (vendor demo):
```sh
/rockchip_test/npu/rknn_inference \
  /rockchip_test/npu/vgg_16_maxpool/vgg_16.rknn \
  /rockchip_test/npu/vgg_16_maxpool/goldfish_224x224.jpg \
  1
```
Deployment scripts belong in:

target/deploy/ (e.g., deploy.sh, run_on_board.sh)

Versioning

Git tags follow SemVer: vMAJOR.MINOR.PATCH

Milestones and acceptance criteria live in docs/roadmap.md

Releases should attach key logs and (optionally) model artifacts

Data & Artifacts Policy

Do not commit datasets: host/data/raw, host/data/processed

Do not commit large artifacts: .onnx, .rknn, .pt/.pth (use Release or Git LFS)

Keep only:

small samples (optional)

metadata (model_meta.json)

scripts/configs to reproduce results

## License

See LICENSE.

