# Data, model and academic-asset provenance

Publication review: 2026-10-03. Historical academic source and aggregate evidence are not a fully reproducible or validated navigation system. Repository-authored source code uses [MIT](LICENSE); third-party data, assets, libraries, submodules and models retain their own terms. Third-party licenses/notices remain independent and must be preserved.

## Upstream dataset families

- [ImVisible / Pedestrian Traffic Lights](https://github.com/samuelyu2002/ImVisible): identified in preparation/report evidence. Its repository has an [MIT notice](https://raw.githubusercontent.com/samuelyu2002/ImVisible/master/LICENSE); retain it for covered upstream material. Coverage of the separately downloaded image bundle was not independently established. No assertion that every historical photograph is MIT-covered.
- [NUS Global Streetscapes](https://github.com/ualsg/global-streetscapes): identified in historical evaluation preparation. The [official website](https://ual.sg/project/global-streetscapes/) states CC BY 4.0, whereas the [publisher dataset card](https://huggingface.co/datasets/NUS-UAL/global-streetscapes) states CC BY-SA 4.0. Exact historical material/version and image-source terms remain unresolved; neither license is assigned to local historical artifacts here.
- The historical bibliography also cites Roboflow collections `dkdkd/july_6`, `dkdkd/capstone-for-detection`, `dkdkd/capstone-for-detection1` and `esera/crosswalk-cz3sx`. These are cited candidate sources, **not a verified training manifest for best.pt**. Exact collection versions and terms were not established.

## Excluded assets and expected local paths

- `best.pt`: historical YOLOv5 checkpoint; creator, training provenance and redistribution rights unresolved. Scripts still expect a separately supplied authorized checkpoint at this location.
- `src/results/test.csv` and `src/results/pred.csv`: per-sample historical labels/predictions excluded for unresolved exact dataset/version/license alignment. Intentional execution writes new local files at these ignored paths; no historical evaluation was rerun.
- `experiments/cube.png`: illustration rights unknown; still the Hough notebook's expected local input.
- `latex/figs/neu1.jpg` and eleven processed photograph illustrations: exact source/version/reuse unresolved. Complete filenames and reasons are recorded in the cleanup report.
- Submitted academic PDF **and original LaTeX source**: both identify students; PDF embeds third-party/dataset imagery. Originals are preserved privately without editing. No public/redacted edition was generated. A separately reviewed public edition could be created later.
- Saved photograph/cube image derivatives were omitted from `experiments/edges.ipynb` (13 image representations) and `experiments/hough.ipynb` (6), without changing code, execution counts, cell metadata or retained numerical/text results. Complete originals are preserved privately. This is publication exclusion, not a new scientific evaluation.

## Retained aggregate evidence

The six confusion matrices (`latex/figs/{blocked,mode,zebra}.png` and `src/results/conf_matrix_{blocked,mode,zebra}.png`), `latex/figs/thresholds.png`, and the edge notebook's histogram/threshold plot remain project-generated aggregate figures. They do not reproduce source photographs. Historical values and methodological limitations remain; aggregate retention does not clear the raw dataset or checkpoint.

## Historical execution requirements

From `src/`, the main script expects `../yolov5`, `../best.pt` and `../data/dataset.csv`. Obtain authorized images, labels and checkpoint separately. The historical evaluation schema includes `file`, `zebra`, `mode`, `blocked`, `x`, `y`, `theta_rad` and `theta_deg`; check actual source contracts rather than reconstructing labels from figures. Notebook paths also refer to local `data/examples_cropped/` images. Full execution is unavailable from the public tree alone.

The pinned YOLOv5 submodule remains unchanged/uninitialized here. Check and preserve its exact revision's upstream license when supplying it; the project MIT license cannot override upstream terms. No third-party notice was removed.

Geometry semantics, metrics, RANSAC/fallback behavior, evaluation splits, Unix timeout limitations and conclusions remain unresolved. No model execution/report regeneration occurred. Prior Git copies remain; historical cleanup is a separate manual owner decision.
