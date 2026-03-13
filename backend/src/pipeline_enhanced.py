"""
Enhanced preprocessing and feature extraction pipeline for live inference.
Matches the enhanced training feature extraction (48 features).

Pipeline: raw canvas strokes -> resample to 30pts -> normalize to 1x1 box -> 48 features

Features:
  - 6 base: num_strokes, width, height, aspect_ratio, total_length, area
  - 8 direction histogram bins
  - 4 curvature: mean, std, max, total
  - 4 endpoints: start_x, start_y, end_x, end_y
  - 1 direction_changes
  - 25 grid zones (5x5)
"""

import numpy as np
import math


def resample_stroke(stroke, num_points=30):
    """Resample a single stroke to a fixed number of evenly-spaced points."""
    x = np.array([p['x'] for p in stroke])
    y = np.array([p['y'] for p in stroke])

    if len(x) == 1:
        return np.column_stack((np.repeat(x, num_points), np.repeat(y, num_points)))

    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative_dist[-1]

    if total_length == 0:
        return np.column_stack((np.repeat(x[0], num_points), np.repeat(y[0], num_points)))

    target_dist = np.linspace(0, total_length, num_points)
    new_x = np.interp(target_dist, cumulative_dist, x)
    new_y = np.interp(target_dist, cumulative_dist, y)

    return np.column_stack((new_x, new_y))


def preprocess_strokes(strokes, target_points_per_stroke=30):
    """
    Preprocess raw strokes: resample + normalize to unit bounding box.
    Returns list of numpy arrays, each shape (num_points, 2).
    """
    if not strokes:
        return None

    resampled = []
    for stroke in strokes:
        if len(stroke) > 0:
            resampled.append(resample_stroke(stroke, target_points_per_stroke))

    if not resampled:
        return None

    all_points = np.vstack(resampled)
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)

    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height)

    if max_dim == 0:
        max_dim = 1

    normalized = []
    for stroke in resampled:
        norm_x = (stroke[:, 0] - min_x) / max_dim
        norm_y = (stroke[:, 1] - min_y) / max_dim
        normalized.append(np.column_stack((norm_x, norm_y)))

    return normalized


def _compute_direction_histogram(normalized_strokes, num_bins=8):
    """Compute histogram of stroke directions (angles)."""
    angles = []
    
    for stroke in normalized_strokes:
        for i in range(1, len(stroke)):
            dx = stroke[i, 0] - stroke[i-1, 0]
            dy = stroke[i, 1] - stroke[i-1, 1]
            
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                angle = math.atan2(dy, dx)
                angle_norm = (angle + math.pi) / (2 * math.pi)  # [0, 1]
                angles.append(angle_norm)
    
    if not angles:
        return [0.0] * num_bins
    
    hist = [0] * num_bins
    for a in angles:
        bin_idx = min(int(a * num_bins), num_bins - 1)
        hist[bin_idx] += 1
    
    total = sum(hist)
    if total > 0:
        hist = [h / total for h in hist]
    
    return hist


def _compute_curvature_features(normalized_strokes):
    """Compute curvature statistics."""
    curvatures = []
    
    for stroke in normalized_strokes:
        if len(stroke) < 3:
            continue
        
        for i in range(1, len(stroke) - 1):
            dx1 = stroke[i, 0] - stroke[i-1, 0]
            dy1 = stroke[i, 1] - stroke[i-1, 1]
            dx2 = stroke[i+1, 0] - stroke[i, 0]
            dy2 = stroke[i+1, 1] - stroke[i, 1]
            
            len1 = math.sqrt(dx1**2 + dy1**2)
            len2 = math.sqrt(dx2**2 + dy2**2)
            
            if len1 > 1e-6 and len2 > 1e-6:
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


def _compute_endpoint_features(normalized_strokes):
    """Extract start and end point of first stroke."""
    if not normalized_strokes or len(normalized_strokes[0]) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    
    first = normalized_strokes[0]
    return [
        float(first[0, 0]),   # start_x
        float(first[0, 1]),   # start_y
        float(first[-1, 0]),  # end_x
        float(first[-1, 1])   # end_y
    ]


def _count_direction_changes(normalized_strokes, threshold=math.pi / 4):
    """Count significant direction changes (corners)."""
    changes = 0
    
    for stroke in normalized_strokes:
        if len(stroke) < 3:
            continue
        
        prev_angle = None
        for i in range(1, len(stroke)):
            dx = stroke[i, 0] - stroke[i-1, 0]
            dy = stroke[i, 1] - stroke[i-1, 1]
            
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                angle = math.atan2(dy, dx)
                
                if prev_angle is not None:
                    diff = abs(angle - prev_angle)
                    if diff > math.pi:
                        diff = 2 * math.pi - diff
                    if diff > threshold:
                        changes += 1
                
                prev_angle = angle
    
    return changes


def _compute_grid_density(normalized_strokes, grid_size=5):
    """Compute point density in a grid_size x grid_size grid."""
    grid_counts = [0] * (grid_size * grid_size)
    total_points = 0
    
    for stroke in normalized_strokes:
        for pt in stroke:
            x, y = pt[0], pt[1]
            col = min(int(x * grid_size), grid_size - 1)
            row = min(int(y * grid_size), grid_size - 1)
            grid_counts[row * grid_size + col] += 1
            total_points += 1
    
    if total_points == 0:
        return [0.0] * (grid_size * grid_size)
    
    return [c / total_points for c in grid_counts]


def extract_features(normalized_strokes):
    """
    Extract 48 enhanced features matching the training pipeline.
    
    Feature order:
        num_strokes, width, height, aspect_ratio, total_length, area,
        dir_hist_0..7, curv_mean, curv_std, curv_max, curv_total,
        start_x, start_y, end_x, end_y, dir_changes,
        grid_0..24
    """
    if not normalized_strokes:
        return None

    # Base features
    num_strokes = len(normalized_strokes)
    all_x, all_y = [], []
    total_length = 0.0

    for stroke in normalized_strokes:
        for i in range(len(stroke)):
            all_x.append(float(stroke[i, 0]))
            all_y.append(float(stroke[i, 1]))
            if i > 0:
                dx = stroke[i, 0] - stroke[i-1, 0]
                dy = stroke[i, 1] - stroke[i-1, 1]
                total_length += math.sqrt(dx**2 + dy**2)

    if not all_x:
        return None

    width = max(all_x) - min(all_x)
    height = max(all_y) - min(all_y)
    aspect_ratio = width / (height + 1e-6)
    area = width * height

    # Enhanced features
    dir_hist = _compute_direction_histogram(normalized_strokes)
    curvature = _compute_curvature_features(normalized_strokes)
    endpoints = _compute_endpoint_features(normalized_strokes)
    dir_changes = _count_direction_changes(normalized_strokes)
    grid = _compute_grid_density(normalized_strokes, grid_size=5)

    # Assemble in exact training CSV column order
    features = [
        num_strokes,
        round(width, 6),
        round(height, 6),
        round(aspect_ratio, 6),
        round(total_length, 6),
        round(area, 6),
    ]
    features += [round(v, 6) for v in dir_hist]      # 8
    features += [round(v, 6) for v in curvature]     # 4
    features += [round(v, 6) for v in endpoints]     # 4
    features += [dir_changes]                         # 1
    features += [round(v, 6) for v in grid]          # 25

    return features  # Total: 48


def compute_bounding_box(strokes):
    """Compute bounding box from raw strokes (canvas pixel coords)."""
    all_x, all_y = [], []
    for stroke in strokes:
        for point in stroke:
            all_x.append(point['x'])
            all_y.append(point['y'])

    if not all_x:
        return None

    return {
        'minX': min(all_x),
        'minY': min(all_y),
        'maxX': max(all_x),
        'maxY': max(all_y),
    }
