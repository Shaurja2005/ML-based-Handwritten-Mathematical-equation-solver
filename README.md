# Handwritten Mathematical Equation Solver

A full-stack application that recognizes handwritten mathematical symbols from online strokes, builds equations in real time, and solves them.

This project uses **traditional machine learning only** (scikit-learn), with no deep learning models.

---

## 1) Project Summary

The system converts raw canvas strokes into recognized symbols and then into solvable math expressions.

High-level flow:

1. User draws on browser canvas.
2. Frontend sends stroke JSON to backend (`/api/predict`) after 800 ms idle.
3. Backend preprocesses strokes, extracts features, runs SVM prediction.
4. Backend returns symbol + confidence + superscript flag.
5. Frontend builds equation string and calls `/api/solve` when requested.
6. Backend solves with SymPy and returns result/error.

---

## 2) Core Capabilities

- Real-time symbol recognition from online stroke data
- Confidence-based rejection (`Not Recognized` below threshold)
- Superscript detection for exponents using spatial heuristics
- Equation parsing + solving with SymPy (supports `^`, implicit multiplication)
- Expression evaluation mode (`2+3` -> `= 5`)
- Equation solving mode (`2*x+4=10` -> `x = 3`)
- Browser UI with dual canvas layers (ink + digital rendering)

---

## 3) Tech Stack

### Frontend

- HTML, CSS, JavaScript (vanilla)
- HTML5 Canvas for drawing and digital glyph rendering

### Backend

- Flask + Flask-CORS
- SymPy for parsing/equation solving

### ML/Training

- scikit-learn
  - `SVC` (primary production model)
  - `RandomForestClassifier` (benchmark/comparison)
  - `GradientBoostingClassifier` (benchmark/comparison)
- `StandardScaler`
- `joblib`

---

## 4) Data Format

Input samples are JSON files with stroke sequences and labels.

Typical structure:

```json
{
  "label": "3",
  "strokes": [
    [
      { "x": 0.12, "y": 0.18 },
      { "x": 0.15, "y": 0.2 }
    ],
    [
      { "x": 0.3, "y": 0.45 },
      { "x": 0.33, "y": 0.47 }
    ]
  ]
}
```

Recognized symbol set includes digits, operators, parentheses, and variable `x` (including `\times` and `\div` label variants).

---

## 5) Preprocessing Strategy

Used both in training and live inference:

1. **Stroke resampling**
   - Each stroke is resampled to a fixed 30 points.
   - Reduces variation caused by drawing speed and sampling rate.

2. **Geometric normalization**
   - All points are normalized into a **1x1 unit box** using max-dimension scaling.
   - Removes absolute position and scale differences.

> Note: The model does **not** scale to a 3x3 box. Grid sizes (3x3 / 5x5) are used only for density features after 1x1 normalization.

---

## 6) Feature Extraction

### Original Pipeline (`pipeline.py`)

- 15 features total
- 6 base geometric features + 9 grid-density features (3x3)

### Enhanced Pipeline (`pipeline_enhanced.py`)

- 48 features total
- 6 base features
- 8 direction histogram bins
- 4 curvature statistics
- 4 endpoint features (first-stroke start/end)
- 1 direction-change count
- 25 grid-density features (5x5)

The enhanced feature set is designed to reduce confusion among visually similar symbols (e.g., `(` vs `)`, `x` vs `\times`).

---

## 7) Model Training Pipeline

Primary training script: `src/model_script/train_model_enhanced.py`

It performs:

1. Dataset load from `data/dataset_enhanced.csv`
2. Stratified train/test split (80/20)
3. Hyperparameter search:
   - SVM: GridSearchCV over `C`, `gamma` (RBF)
   - RF: GridSearchCV over tree count/depth/split
   - GB: GridSearchCV over estimators/depth/learning rate
4. Evaluation with classification report + confusion-pair summary
5. Model artifact saving

Saved artifacts:

- `models/svm_model_enhanced.pkl`: production SVM classifier
- `models/scaler_enhanced.pkl`: scaler paired with production SVM
- `models/best_model.pkl`: best among SVM/RF/GB for experiment tracking
- `models/best_scaler.pkl`: scaler for best model if required

---

## 8) Backend Architecture

Main entry: `backend/src/app.py`

### Startup behavior

- Tries to load enhanced artifacts first:
  - `svm_model_enhanced.pkl`
  - `scaler_enhanced.pkl`
- Falls back to original artifacts if enhanced not present:
  - `svm_model.pkl`
  - `scaler.pkl`
- Chooses matching pipeline automatically:
  - enhanced -> `pipeline_enhanced`
  - original -> `pipeline`

### Inference behavior

- `/api/predict`:
  - validates stroke payload
  - preprocesses + extracts features with active pipeline
  - predicts class using SVM
  - computes confidence using sigmoid of top1-top2 decision margin
  - applies confidence threshold (`0.60`)
  - maps model labels to display/math-safe symbols
  - runs superscript detection (`spatial.py`)

- `/api/solve`:
  - validates equation/expression string
  - normalizes unicode operators (`×`, `÷`)
  - solves equation mode or expression mode via SymPy (`solver.py`)

---

## 9) Frontend Architecture

Core file: `frontend/src/main.js`

- Two canvas layers:
  - `drawing-canvas` for user ink
  - `display-canvas` for recognized digital symbols
- Idle recognition trigger: 800 ms after stroke stop
- Character-level equation construction
- Superscript-aware rendering and math string generation
- Buttons: Solve, Undo, Clear
- Status + confidence feedback in bottom bar

---

## 10) API Reference

### `POST /api/predict`

Request body (example):

```json
{
  "strokes": [
    [
      { "x": 120, "y": 80 },
      { "x": 124, "y": 92 }
    ]
  ],
  "previousCharacter": {
    "bbox": { "minX": 10, "minY": 10, "maxX": 40, "maxY": 60 }
  }
}
```

Success response (recognized):

```json
{
  "recognized": true,
  "label": "1",
  "display": "1",
  "mathChar": "1",
  "confidence": 0.733,
  "bbox": { "minX": 100, "minY": 20, "maxX": 100, "maxY": 140 },
  "isSuperscript": false
}
```

Low-confidence response:

```json
{
  "recognized": false,
  "confidence": 0.41,
  "message": "Not Recognized"
}
```

### `POST /api/solve`

Request:

```json
{ "equation": "2*x+4=10" }
```

Success:

```json
{ "success": true, "result": "x = 3", "type": "equation" }
```

Error:

```json
{ "success": false, "error": "Invalid characters in equation" }
```

---

## 11) Setup and Run

## Prerequisites

- Python 3.x
- Windows PowerShell or Command Prompt
- (Optional) Conda

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run options

### Option A: Batch launcher (Windows)

```bash
start.bat
```

### Option B: Conda session

```bash
conda activate base
python backend/src/app.py
```

### Option C: venv session

```bash
venv\Scripts\activate
python backend/src/app.py
```

Then open:

```text
http://localhost:5000
```

---

## 12) Training / Retraining Workflow

From repository root:

1. Preprocess raw JSON dataset:

```bash
python src/data_scripts/data_preprocessing_script.py
```

2. Build enhanced feature CSV:

```bash
python src/features/feature_extraction_enhanced.py
```

3. Train/tune models and save artifacts:

```bash
python src/model_script/train_model_enhanced.py
```

4. Start backend and verify inference:

```bash
python backend/src/app.py
```

---

## 13) Project Structure

```text
Handwritten-Mathematical-equation-solver/
├── backend/
│   └── src/
│       ├── app.py
│       ├── pipeline.py
│       ├── pipeline_enhanced.py
│       ├── spatial.py
│       └── solver.py
├── frontend/
│   └── src/
│       ├── index.html
│       ├── main.js
│       └── styles.css
├── src/
│   ├── data_scripts/
│   │   └── data_preprocessing_script.py
│   ├── features/
│   │   └── feature_extraction_enhanced.py
│   └── model_script/
│       ├── train_model.py
│       └── train_model_enhanced.py
├── data/
│   ├── cleaned-dataset/
│   ├── dataset_preprocessed/
│   └── dataset_enhanced.csv
├── models/
│   ├── best_model.pkl
│   ├── best_scaler.pkl
│   ├── scaler_enhanced.pkl
│   └── svm_model_enhanced.pkl
├── Reports/
│   ├── iters.txt
│   ├── model_backend_functioning.txt
│   ├── preprocessing_and_features.txt
│   └── test_set_evualation.txt
├── requirements.txt
├── start.bat
└── README.md
```

---

## 14) Current Performance Snapshot

From `Reports/test_set_evualation.txt`:

- Enhanced model comparison:
  - SVM: **98.38%**
  - RF: 98.05%
  - GB: 98.30%
- Current production choice: **Enhanced SVM + Enhanced Scaler**

---

## 15) Notes

- `best_model.pkl` is for experiment tracking; backend inference currently uses SVM artifacts (`svm_model_enhanced.pkl` + `scaler_enhanced.pkl`) or original fallback files.
- Label mapping includes compatibility handling for symbol variants (`\div`, `\\div`, `\times`, minus variants).
- This project is optimized for online strokes (vector points), not offline image OCR.

---

## Author

Sauryadipta Bhattacharya
