# ZebrAI Crossing

A computer vision coursework prototype developed at Universitat Autònoma de Barcelona (UAB) by Albert Capdevila, Levon Kesoyan, and Luis Martínez. The project investigates crosswalk and pedestrian traffic-light detection for the OrionWay navigation project.

The implementation combines a custom YOLOv5 detector with classical image processing: filtering, thresholding, edge detection, line fitting, and crosswalk geometry estimation. It estimates crosswalk orientation/start position and recognizes red/green pedestrian signals.

## Repository contents

- [`src/ZebraAI.py`](src/ZebraAI.py): combined pipeline, parameter search, and evaluation.
- [`experiments/PTLdetection.py`](experiments/PTLdetection.py): traffic-light detection experiment.
- `experiments/*.ipynb`: data preparation, edge detection, and Hough transform exploration.
- `src/results/`: retained aggregate confusion matrices; historical per-sample CSVs are excluded.
- `latex/figs/`: aggregate plots. The submitted report PDF/source and photographic illustrations are preserved privately and excluded.
- `best.pt`: required local YOLO weights, not distributed because provenance/rights remain unresolved.
- `yolov5/`: Git submodule; its source is supplied by the submodule rather than copied into this repository.
- `data/`: local input data, excluded from version control.

## Setup

Use a Python environment compatible with the recorded YOLOv5 revision. The repository does not specify an exact Python version or locked dependency environment.

From the repository root, initialize the existing submodule:

```bash
git submodule update --init --recursive
python -m venv .venv
```

Activate in PowerShell with `.\.venv\Scripts\Activate.ps1`, or on macOS/Linux with `source .venv/bin/activate`.

`requirements.txt` contains a broad list, including optional model export backends with platform-specific requirements. It is preserved except for duplicate identical entries. Review compatibility before attempting a full installation:

```bash
python -m pip install -r requirements.txt
```

## Execution and recorded evidence

The main script resolves `../yolov5`, `../best.pt`, and `../data/dataset.csv` relative to the working directory. It expects local image paths in the dataset's `file` column and evaluation targets for the geometry/traffic-light workflow. See [DATA_ASSETS.md](DATA_ASSETS.md) and source contracts for local inputs and label format. Inference requires an authorized checkpoint and dataset not supplied here.

Run from `src/` on a platform supporting Unix signals:

```bash
cd src
python ZebraAI.py
```

This command runs parameter search and evaluation, and writes `src/results/test.csv`, `src/results/pred.csv`, and plots. Back up the existing recorded results before an intentional rerun. It was not executed during maintenance.

## Limitations and licensing

- `signal.SIGALRM` and `signal.alarm` in the main pipeline prevent the current timeout implementation from running unchanged on Windows.
- Local input images and `data/dataset.csv` are not supplied; full evaluation cannot be reproduced from this repository alone.
- The dependency list contains overlapping OpenCV packages and numerous optional export runtimes. No dependency modernization was performed.
- Original notebook code/narrative and implementation language remain. Restricted saved image representations are omitted; complete notebooks and submitted report are preserved privately. Aggregate plots remain historical evidence.
- The checkpoint is excluded, and `yolov5` is uninitialized here. Both must be supplied appropriately for historical local model loading.
- No source-code license is included. [DATA_ASSETS.md](DATA_ASSETS.md) records sources, unresolved terms and exclusions. Preserve all applicable upstream notices.

This repository documents an academic prototype; it does not establish a validated assistive-navigation system.
