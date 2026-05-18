# Object Detection Tools

## Overview

This repository is a work-in-progress collection of Python utilities for preparing, running, and evaluating object detection datasets that use Pascal VOC-style XML annotations. The scripts support common tasks around TensorFlow Object Detection workflows, including annotation conversion, TFRecord generation, label-map creation, image augmentation, dataset cleanup, prediction export, and evaluation reporting.

The project is best treated as a research/prototyping toolkit rather than a packaged command-line application. Several scripts contain experiment-specific paths and parameters that should be adjusted before running them on a new machine or dataset.

## Key Features

- Convert Pascal VOC XML annotations into Pandas DataFrames or CSV files.
- Generate TensorFlow Object Detection API TFRecord files from XML annotations and images.
- Create `.pbtxt` label maps from a saved class distribution.
- Validate bounding boxes against image dimensions.
- Clean datasets by removing missing, invalid, or low-frequency classes.
- Apply image augmentations such as scaling, contrast, saturation, hue, illumination, and Gaussian blur.
- Run TensorFlow SavedModel object detection inference over `.jpg` images.
- Export predictions to CSV and JSON.
- Draw annotation or prediction boxes onto images.
- Build confusion-matrix and precision/recall/F1 reports from predictions and ground-truth XML annotations.

## Technical Approach

The repository is organized as standalone Python scripts. Most utilities operate on folders containing paired `.jpg` and `.xml` files, where each XML file follows the Pascal VOC annotation structure with `object`, `name`, and `bndbox` fields.

The main workflow supported by the code is:

1. Inspect and clean XML annotations.
2. Optionally rebalance or augment image data.
3. Generate class distributions and label maps.
4. Convert annotations into TFRecord files for TensorFlow Object Detection training.
5. Load a trained TensorFlow SavedModel for inference.
6. Save prediction summaries and evaluate detections against ground truth.

Some scripts expose reusable functions, while others are configured through constants inside the file. Review and edit local dataset/model paths before executing those scripts.

## Tech Stack

- Python 3
- TensorFlow / TensorFlow Object Detection API
- OpenCV
- NumPy
- Pandas
- Pillow
- XML parsing with Python's standard `xml.etree.ElementTree`
- Multiprocessing with Python's standard `multiprocessing`
- Excel input/output through Pandas-compatible engines

## Techniques Used

- Computer vision dataset preprocessing
- Pascal VOC XML annotation parsing
- TFRecord generation for object detection training
- TensorFlow SavedModel inference
- Bounding-box IoU matching
- Confusion-matrix based evaluation
- Image augmentation
- Dataset class balancing and filtering
- Parallel processing for dataset cleanup and copying

## Repository Structure

```text
.
├── README.md
├── XML2PD.py                    # Converts Pascal VOC XML annotations to Pandas/CSV
├── checkSizeBB.py               # Validates bounding boxes against image dimensions
├── clasesUnicas.py              # Compares annotation classes against catalog spreadsheets
├── copiarDatasetparallel.py     # Cleans/copies datasets and can generate distributions/label maps
├── eliminarClasesImg.py         # Removes selected classes and supports class balancing
├── evaluation.py                # Produces confusion matrix and metric reports
├── generate_label_map.py        # Generates TensorFlow label_map.pbtxt files
├── generate_tfrecord.py         # Converts XML annotations and images to TFRecord
├── plotBoxes.py                 # Draws ground-truth boxes on images
├── prediction.py                # Runs model inference and exports predictions
├── splitDataset.py              # Splits/copies paired image/XML datasets
├── transformacionesImagenes.py  # Image augmentation helpers
├── catalogo.xlsx                # Catalog data used by cleaning/class mapping scripts
└── catalogo2.xlsx               # Additional catalog data
```

## Getting Started

No `requirements.txt`, `pyproject.toml`, or environment file is included, so dependency installation must be inferred from imports.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the likely Python dependencies:

```bash
pip install tensorflow opencv-python numpy pandas pillow openpyxl
```

The scripts that import `object_detection.utils` also require the TensorFlow Object Detection API to be installed and available on `PYTHONPATH`. Follow the TensorFlow Object Detection API installation instructions that match your TensorFlow version, then verify the import:

```bash
python -c "from object_detection.utils import label_map_util; print('ok')"
```

## Configuration

There are no environment variables or `.env` files in this repository.

Several scripts use constants for dataset, model, output, catalog, and label-map paths. Before running those scripts, replace the placeholder-like local values in the relevant file with paths for your own workspace:

- `IMAGE_PATHS`
- `PATH_TO_MODEL_DIR`
- `PATH_TO_LABELS`
- `PATH_PREDICTIONS`
- `PATH_IMAGE_PREDICTIONS`
- `pathDataset`
- `paths`
- `pathsSave`
- `rutaCatalogo`
- `path_distribution`
- `pathLabelMap`

Do not commit private dataset paths, credentials, or API keys.

## Usage

### Convert XML annotations to TFRecord

`generate_tfrecord.py` is the most command-line friendly script in the repository:

```bash
python generate_tfrecord.py \
  --xml_dir path/to/annotations \
  --image_dir path/to/images \
  --labels_path path/to/label_map.pbtxt \
  --output_path path/to/output.record \
  --csv_path path/to/annotations.csv
```

If `--image_dir` is omitted, the script uses the XML directory as the image directory.

### Convert XML annotations to a DataFrame or CSV

Use `XML2PD.py` from Python:

```python
from XML2PD import xml2pd

df = xml2pd("path/to/images_and_annotations", csv=True)
```

When `csv=True`, the script writes `dataset.csv`.

### Generate a label map

`generate_label_map.py` expects a `distribution.npy` file where the first column contains class names:

```bash
python generate_label_map.py
```

By default, it writes `label_map.pbtxt` in the current directory.

### Run predictions

`prediction.py` provides callable functions for loading a TensorFlow SavedModel and running inference:

```python
from prediction import predict

predictions_df, prediction_json = predict(
    PATH_TO_MODEL_DIR="path/to/model_dir",
    IMAGE_PATHS="path/to/images",
    PATH_TO_LABELS="path/to/label_map.pbtxt",
    PATH_PREDICTIONS="path/to/output",
    PATH_IMAGE_PREDICTIONS="path/to/annotated_images",
    Tscore=0.5,
)
```

Expected outputs include:

- `predictions.csv`
- `jsonindicadores.json`
- Optional annotated `.jpg` images when `PATH_IMAGE_PREDICTIONS` is provided

### Evaluate predictions

`evaluation.py` compares predictions against XML annotations using IoU matching and writes:

- `confusion_matrix.xlsx`
- `report.xlsx`

Before running it, update its dataset, model, label-map, and output path constants.

## Results / Outputs

Depending on the script and configuration, this repository can produce:

- TFRecord files for TensorFlow training.
- CSV annotation exports.
- TensorFlow label-map files.
- Cleaned or augmented image/XML datasets.
- Prediction CSV files with class labels, scores, and bounding boxes.
- JSON prediction summaries.
- Annotated images with visualized boxes.
- Excel reports for confusion matrix and precision/recall/F1 metrics.

## Limitations

- The repository does not include a dependency manifest or automated tests.
- Several scripts contain hard-coded local paths and must be edited before use.
- There is no single CLI entry point for the full pipeline.
- Some scripts are experiment-specific and mix reusable functions with executable configuration.
- The TensorFlow Object Detection API setup is external to this repository.
- The included catalog spreadsheets are referenced by cleaning/class-mapping utilities, but their schema is only implicitly documented by the code.

## Future Improvements

- Add a `requirements.txt` or `pyproject.toml` with tested dependency versions.
- Move script constants into command-line arguments or a documented config file.
- Add small sample data for smoke testing XML parsing, TFRecord generation, and evaluation.
- Add unit tests for IoU calculation, XML parsing, bounding-box validation, and label-map generation.
- Separate reusable library functions from experiment-specific scripts.
- Document the expected catalog spreadsheet columns and dataset folder conventions.
