"""
Enhanced Feature Extraction for Handwritten Math Symbols.

Improvements over basic 3x3 grid features:
1. Direction histogram (8 bins) - captures stroke direction, critical for ( vs )
2. Curvature statistics - helps distinguish curved vs angular symbols
3. Start/end point positions - stroke order information
4. Direction change count - cusps and corners
5. 5x5 grid zones - finer spatial resolution (25 zones instead of 9)
6. Stroke-level aggregations - mean/std of stroke lengths

Total features: 6 base + 8 direction + 4 curvature + 4 endpoints + 1 dir_changes + 25 grid = 48
"""

import os
import json
import csv
import math
import numpy as np


def compute_direction_histogram(strokes, num_bins=8):
    """
    Compute histogram of stroke directions (angles) across all point pairs.
    This is KEY for distinguishing ( vs ) - they curve in opposite directions.
    Returns normalized histogram with `num_bins` values.
    """
    angles = []
    
    for stroke in strokes:
        for i in range(1, len(stroke)):
            dx = stroke[i]['x'] - stroke[i-1]['x']
            dy = stroke[i]['y'] - stroke[i-1]['y']
            
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                angle = math.atan2(dy, dx)  # Range: [-pi, pi]
                angle_normalized = (angle + math.pi) / (2 * math.pi)  # Range: [0, 1]
                angles.append(angle_normalized)
    
    if not angles:
        return [0.0] * num_bins
    
    # Create histogram
    hist = [0] * num_bins
    for a in angles:
        bin_idx = min(int(a * num_bins), num_bins - 1)
        hist[bin_idx] += 1
    
    # Normalize
    total = sum(hist)
    if total > 0:
        hist = [h / total for h in hist]
    
    return hist


def compute_curvature_features(strokes):
    """
    Compute curvature statistics across all strokes.
    Curvature = rate of change of direction.
    Returns: [mean_curvature, std_curvature, max_curvature, total_curvature]
    """
    curvatures = []
    
    for stroke in strokes:
        if len(stroke) < 3:
            continue
            
        for i in range(1, len(stroke) - 1):
            # Vectors: prev->curr and curr->next
            dx1 = stroke[i]['x'] - stroke[i-1]['x']
            dy1 = stroke[i]['y'] - stroke[i-1]['y']
            dx2 = stroke[i+1]['x'] - stroke[i]['x']
            dy2 = stroke[i+1]['y'] - stroke[i]['y']
            
            len1 = math.sqrt(dx1**2 + dy1**2)
            len2 = math.sqrt(dx2**2 + dy2**2)
            
            if len1 > 1e-6 and len2 > 1e-6:
                # Cross product gives signed curvature
                cross = dx1 * dy2 - dy1 * dx2
                curvature = cross / (len1 * len2 + 1e-6)
                curvatures.append(curvature)
    
    if not curvatures:
        return [0.0, 0.0, 0.0, 0.0]
    
    arr = np.array(curvatures)
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.max(np.abs(arr))),
        float(np.sum(np.abs(arr)))
    ]


def compute_endpoint_features(strokes):
    """
    Extract start and end point positions of the first stroke.
    Normalized to [0,1] range.
    Returns: [start_x, start_y, end_x, end_y]
    """
    if not strokes or not strokes[0]:
        return [0.0, 0.0, 0.0, 0.0]
    
    first_stroke = strokes[0]
    start_x = first_stroke[0]['x']
    start_y = first_stroke[0]['y']
    end_x = first_stroke[-1]['x']
    end_y = first_stroke[-1]['y']
    
    return [start_x, start_y, end_x, end_y]


def count_direction_changes(strokes, angle_threshold=math.pi / 4):
    """
    Count significant direction changes (cusps/corners).
    A direction change > threshold counts as a corner.
    """
    changes = 0
    
    for stroke in strokes:
        if len(stroke) < 3:
            continue
            
        prev_angle = None
        for i in range(1, len(stroke)):
            dx = stroke[i]['x'] - stroke[i-1]['x']
            dy = stroke[i]['y'] - stroke[i-1]['y']
            
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                angle = math.atan2(dy, dx)
                
                if prev_angle is not None:
                    diff = abs(angle - prev_angle)
                    if diff > math.pi:
                        diff = 2 * math.pi - diff
                    if diff > angle_threshold:
                        changes += 1
                
                prev_angle = angle
    
    return changes


def compute_grid_density(strokes, grid_size=5):
    """
    Compute point density in a grid_size x grid_size grid.
    5x5 = 25 zones for finer spatial resolution.
    """
    grid_counts = [0] * (grid_size * grid_size)
    total_points = 0
    
    for stroke in strokes:
        for pt in stroke:
            x, y = pt['x'], pt['y']
            col = min(int(x * grid_size), grid_size - 1)
            row = min(int(y * grid_size), grid_size - 1)
            grid_index = row * grid_size + col
            grid_counts[grid_index] += 1
            total_points += 1
    
    if total_points == 0:
        return [0.0] * (grid_size * grid_size)
    
    return [count / total_points for count in grid_counts]


def calculate_enhanced_features(filepath):
    """
    Extract all enhanced features from a JSON file.
    Returns dict with label + 48 features.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    strokes = data.get("strokes", [])
    label = data.get("label", "unknown")
    
    if not strokes:
        return None
    
    # === Base features (same as before) ===
    num_strokes = len(strokes)
    all_x, all_y = [], []
    total_length = 0.0
    
    for stroke in strokes:
        for i, pt in enumerate(stroke):
            all_x.append(pt['x'])
            all_y.append(pt['y'])
            if i > 0:
                dx = pt['x'] - stroke[i-1]['x']
                dy = pt['y'] - stroke[i-1]['y']
                total_length += math.sqrt(dx**2 + dy**2)
    
    if not all_x:
        return None
    
    width = max(all_x) - min(all_x)
    height = max(all_y) - min(all_y)
    aspect_ratio = width / (height + 1e-6)
    area = width * height
    
    # === New features ===
    direction_hist = compute_direction_histogram(strokes, num_bins=8)
    curvature_feats = compute_curvature_features(strokes)
    endpoint_feats = compute_endpoint_features(strokes)
    dir_changes = count_direction_changes(strokes)
    grid_5x5 = compute_grid_density(strokes, grid_size=5)
    
    # === Build feature dict ===
    features = {
        "label": label,
        # Base (6)
        "num_strokes": num_strokes,
        "width": round(width, 6),
        "height": round(height, 6),
        "aspect_ratio": round(aspect_ratio, 6),
        "total_length": round(total_length, 6),
        "area": round(area, 6),
    }
    
    # Direction histogram (8)
    for i, val in enumerate(direction_hist):
        features[f"dir_hist_{i}"] = round(val, 6)
    
    # Curvature (4)
    curv_names = ["curv_mean", "curv_std", "curv_max", "curv_total"]
    for name, val in zip(curv_names, curvature_feats):
        features[name] = round(val, 6)
    
    # Endpoints (4)
    ep_names = ["start_x", "start_y", "end_x", "end_y"]
    for name, val in zip(ep_names, endpoint_feats):
        features[name] = round(val, 6)
    
    # Direction changes (1)
    features["dir_changes"] = dir_changes
    
    # 5x5 Grid (25)
    for i, val in enumerate(grid_5x5):
        features[f"grid_{i}"] = round(val, 6)
    
    return features


def get_feature_headers():
    """Return ordered list of feature column names."""
    headers = [
        "label",
        "num_strokes", "width", "height", "aspect_ratio", "total_length", "area",
    ]
    headers += [f"dir_hist_{i}" for i in range(8)]
    headers += ["curv_mean", "curv_std", "curv_max", "curv_total"]
    headers += ["start_x", "start_y", "end_x", "end_y"]
    headers += ["dir_changes"]
    headers += [f"grid_{i}" for i in range(25)]
    return headers


def build_enhanced_csv(input_folder, output_csv):
    """
    Process all JSON files and create enhanced feature CSV.
    """
    headers = get_feature_headers()
    processed = 0
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for filename in os.listdir(input_folder):
            if filename.endswith(".json"):
                filepath = os.path.join(input_folder, filename)
                features = calculate_enhanced_features(filepath)
                
                if features:
                    writer.writerow(features)
                    processed += 1
    
    print("=" * 50)
    print(f"Enhanced Feature Extraction Complete!")
    print(f"Total features: {len(headers) - 1} (excluding label)")
    print(f"Processed {processed} samples -> '{output_csv}'")
    print("=" * 50)


if __name__ == "__main__":
    INPUT_FOLDER = r"data\dataset_preprocessed"
    OUTPUT_CSV = r"data\dataset_enhanced.csv"
    build_enhanced_csv(INPUT_FOLDER, OUTPUT_CSV)
