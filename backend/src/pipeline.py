"""
Preprocessing and feature extraction pipeline for live inference.
Replicates the exact processing used during model training (3x3 grid features).

Pipeline: raw canvas strokes -> resample to 30pts -> normalize to 1x1 box -> 16 features
"""

import numpy as np
import math


def resample_stroke(stroke, num_points=30):
    """Resample a single stroke to a fixed number of evenly-spaced points
    using linear interpolation along cumulative arc length."""
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
    Preprocess raw strokes: resample to fixed points and normalize
    to a unit bounding box (preserving aspect ratio via max-dimension scaling).
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


def extract_features(normalized_strokes):
    """
    Extract the 15 features (6 base + 9 grid zone densities) matching
    the 3x3 training pipeline exactly.

    Feature order (must match training CSV column order):
        num_strokes, width, height, aspect_ratio, total_length,
        area, grid_zone_1 .. grid_zone_9
    """
    if not normalized_strokes:
        return None

    num_strokes = len(normalized_strokes)
    all_x, all_y = [], []
    total_length = 0.0
    grid_counts = [0] * 9
    total_points = 0

    for stroke in normalized_strokes:
        for i in range(len(stroke)):
            x_val = float(stroke[i][0])
            y_val = float(stroke[i][1])
            all_x.append(x_val)
            all_y.append(y_val)
            total_points += 1

            if i > 0:
                dx = x_val - float(stroke[i - 1][0])
                dy = y_val - float(stroke[i - 1][1])
                total_length += math.sqrt(dx**2 + dy**2)

            # 3x3 spatial grid density (same formula as training script)
            col = min(int(x_val * 3), 2)
            row = min(int(y_val * 3), 2)
            grid_counts[row * 3 + col] += 1

    if not all_x or total_points == 0:
        return None

    width = max(all_x) - min(all_x)
    height = max(all_y) - min(all_y)
    aspect_ratio = width / (height + 0.0001)
    area = width * height
    grid_density = [c / total_points for c in grid_counts]

    features = [
        num_strokes,
        round(width, 4),
        round(height, 4),
        round(aspect_ratio, 4),
        round(total_length, 4),
        round(area, 4),
    ] + [round(d, 4) for d in grid_density]

    return features


def compute_bounding_box(strokes):
    """Compute axis-aligned bounding box from raw strokes (canvas pixel coords)."""
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
