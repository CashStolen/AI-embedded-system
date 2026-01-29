# Roadmap

This roadmap defines milestone deliverables and acceptance criteria.  
The goal is **repeatable engineering**: each milestone has verifiable outputs and scripts.

---

## Milestone 1 — Kickoff: Connectivity + RKNN Runtime Verified (v0.1.0)

### Goal
Establish a reliable development chain:
- Windows ↔ Ubuntu VM ↔ RV1126 (SSH / file transfer)
- Confirm RV1126 NPU runtime can execute RKNN inference

### Deliverables
- `docs/milestone1-kickoff/notes.md`
- `docs/milestone1-kickoff/logs/` (key terminal outputs)
- Optional screenshots under `docs/milestone1-kickoff/media/`

### Acceptance Criteria
- Can SSH into RV1126 from host machine
- On RV1126, `rknn_inference` demo runs successfully and produces output + latency report

### Suggested evidence to store
- OS fingerprint (`uname -a`, `/etc/os-release`, `uname -m`)
- RKNN runtime/driver line from demo output
- Baseline perf numbers (avg ms / fps)

---

## Milestone 2 — Dataset + Labeling Pipeline

### Goal
Build a dataset strategy for eye clarity:
- real captures (various conditions) + optional synthetic blur
- labeling scheme defined and reproducible

### Deliverables
- `host/scripts/prepare_data.py`
- `host/data/README.md` (data sources, structure, and how to place files locally)
- `host/configs/data.yaml` (or similar)
- `docs/notes/dataset_strategy.md`

### Acceptance Criteria
- Running `prepare_data.py` creates a deterministic index/manifest (CSV/JSON) and basic statistics:
  - label distribution
  - resolution distribution
  - train/val/test split

---

## Milestone 3 — First Trainable Baseline (ONNX Export)

### Goal
Train a lightweight model suitable for embedded deployment.

### Deliverables
- `host/scripts/train.py`
- `host/scripts/export_onnx.py`
- `host/src/models/` baseline architecture
- `host/configs/train.yaml` (or similar)
- Metrics report under `host/outputs/` (local, gitignored)
- `docs/reports/baseline_metrics.md` (summary + plots saved in `docs/figures/`)

### Acceptance Criteria
- Training is reproducible using config
- ONNX export runs and produces a valid ONNX model locally
- Baseline accuracy/metric is recorded with a fixed evaluation protocol

---

## Milestone 4 — ONNX → RKNN + On-board Validation (Your Model)

### Goal
Convert your ONNX model to RKNN and run it on RV1126.

### Deliverables
- `host/scripts/rknn_convert.py`
- `target/deploy/deploy.sh` + `target/deploy/run_on_board.sh`
- `models/clarity_vX/model_meta.json` (input size, preprocessing, labels, thresholds)
- `docs/notes/rknn_conversion.md` (toolchain version, quantization plan, pitfalls)

### Acceptance Criteria
- `.rknn` produced from your ONNX (artifact stored via Release/LFS if large)
- On RV1126, your model runs successfully and outputs expected format
- Preprocessing on host and target is consistent (documented in `model_meta.json`)

---

## Milestone 5 — Benchmark: AI vs Traditional Metrics + Final Report

### Goal
Compare AI method against traditional clarity metrics fairly, with clear reporting.

### Deliverables
- `bench/traditional/` (Laplacian/Tenengrad etc.)
- `bench/scripts/run_bench.py` and `bench/scripts/summarize.py`
- Results saved locally to `bench/results/` (gitignored) with summarized tables/plots committed to:
  - `docs/figures/`
  - `docs/reports/final_comparison.md`

### Acceptance Criteria
- One command can run the benchmark pipeline locally
- Produces:
  - accuracy/quality metrics
  - latency/throughput (where measurable)
  - clear tables/plots for the final report

---

## Optional Bonus — Qt Integration

### Goal
Add a UI without coupling it to model logic.

### Deliverables
- `target/app/` exposes stable API (e.g., stdout JSON / socket / file output)
- Qt UI consumes API and displays clarity score/class

### Acceptance Criteria
- UI works without modifying core inference logic
- Same `model_meta.json` contract is used end-to-end