"""
Flask backend for Handwritten Mathematical Equation Solver.

Endpoints:
  GET  /              -> Serves the frontend
  POST /api/predict   -> Predict a character from handwritten strokes
  POST /api/solve     -> Parse and solve an equation string

The SVM model and StandardScaler are loaded once at startup.
Supports both original (15-feature) and enhanced (48-feature) pipelines.
"""

import os
import sys
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import both pipelines - we'll select based on which model is loaded
import pipeline as pipeline_original
import pipeline_enhanced
from spatial import detect_superscript
from solver import solve_equation

# Active pipeline module (set during load_model)
active_pipeline = None

# ─── Paths ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend', 'src')

# ─── App Setup ──────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

CONFIDENCE_THRESHOLD = 0.60

# ─── Label Mappings ─────────────────────────────────────────────
# Model labels -> Unicode display characters
DISPLAY_MAP = {
    '\\\\div': '\u00f7',   # model label is \\div (double backslash)
    '\\times': '\u00d7',    # model label is \times (single backslash)
}

# Model labels -> math-safe characters (for solver parsing)
MATH_MAP = {
    '\\\\div': '/',
    '\\times': '*',
}

# ─── Load Model & Scaler ────────────────────────────────────────
model = None
scaler = None


def load_model():
    global model, scaler, active_pipeline
    
    # Check for enhanced model first (48 features), then fall back to original (15 features)
    enhanced_model_path = os.path.join(MODELS_DIR, 'svm_model_enhanced.pkl')
    enhanced_scaler_path = os.path.join(MODELS_DIR, 'scaler_enhanced.pkl')
    original_model_path = os.path.join(MODELS_DIR, 'svm_model.pkl')
    original_scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    
    if os.path.exists(enhanced_model_path) and os.path.exists(enhanced_scaler_path):
        model = joblib.load(enhanced_model_path)
        scaler = joblib.load(enhanced_scaler_path)
        active_pipeline = pipeline_enhanced
        print(f"Enhanced model loaded (48 features).")
    elif os.path.exists(original_model_path) and os.path.exists(original_scaler_path):
        model = joblib.load(original_model_path)
        scaler = joblib.load(original_scaler_path)
        active_pipeline = pipeline_original
        print(f"Original model loaded (15 features).")
    else:
        print(f"ERROR: No model files found in {MODELS_DIR}")
        print("  Expected: svm_model_enhanced.pkl + scaler_enhanced.pkl")
        print("       or: svm_model.pkl + scaler.pkl")
        sys.exit(1)

    print(f"  Classes ({len(model.classes_)}): {list(model.classes_)}")
    print(f"  Features expected: {scaler.n_features_in_}")


def predict_with_confidence(features):
    """
    Run SVM prediction and compute confidence via sigmoid of the margin
    between the top-1 and top-2 OvR decision function scores.
    """
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]

    # OvR decision function: shape (1, n_classes)
    scores = model.decision_function(X_scaled)[0]
    sorted_scores = np.sort(scores)[::-1]
    margin = sorted_scores[0] - sorted_scores[1]

    # Sigmoid of margin: maps margin -> (0, 1) confidence
    confidence = float(1.0 / (1.0 + np.exp(-margin)))

    return str(prediction), confidence


# ─── Frontend Serving ────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


# ─── API: Predict ────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    strokes = data.get('strokes')
    if not strokes or not isinstance(strokes, list):
        return jsonify({'error': 'Missing or invalid strokes'}), 400

    # Validate stroke structure
    for stroke in strokes:
        if not isinstance(stroke, list):
            return jsonify({'error': 'Each stroke must be an array of points'}), 400
        for pt in stroke:
            if not isinstance(pt, dict) or 'x' not in pt or 'y' not in pt:
                return jsonify({'error': 'Each point must have x and y'}), 400

    # 1. Compute raw bounding box (canvas pixel coordinates)
    bbox = active_pipeline.compute_bounding_box(strokes)
    if not bbox:
        return jsonify({'error': 'Could not compute bounding box'}), 400

    # 2. Preprocess: resample + normalize to 1x1 box
    normalized = active_pipeline.preprocess_strokes(strokes)
    if normalized is None:
        return jsonify({'error': 'Preprocessing failed'}), 400

    # 3. Extract features (15 for original, 48 for enhanced)
    features = active_pipeline.extract_features(normalized)
    if features is None:
        return jsonify({'error': 'Feature extraction failed'}), 400

    # 4. Predict with SVM
    label, confidence = predict_with_confidence(features)

    # 5. Confidence thresholding
    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({
            'recognized': False,
            'confidence': round(confidence, 3),
            'message': 'Not Recognized',
        })

    # 6. Map label to display + math characters
    display_char = DISPLAY_MAP.get(label, label)
    math_char = MATH_MAP.get(label, label)

    # 7. Superscript detection (compare against previous character)
    prev = data.get('previousCharacter')
    prev_bbox = prev.get('bbox') if prev else None
    is_superscript = detect_superscript(bbox, prev_bbox)

    return jsonify({
        'recognized': True,
        'label': label,
        'display': display_char,
        'mathChar': math_char,
        'confidence': round(confidence, 3),
        'bbox': bbox,
        'isSuperscript': is_superscript,
    })


# ─── API: Solve ──────────────────────────────────────────────────
@app.route('/api/solve', methods=['POST'])
def api_solve():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    equation = data.get('equation', '').strip()
    if not equation:
        return jsonify({'success': False, 'error': 'Empty equation'}), 400

    result = solve_equation(equation)
    return jsonify(result)


# ─── Main ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    print(f"\nServing frontend from: {FRONTEND_DIR}")
    print(f"Open http://localhost:5000 in your browser\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
