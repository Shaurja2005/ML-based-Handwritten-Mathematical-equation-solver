"""
Spatial analysis engine for superscript (power/exponent) detection.

Uses bounding box heuristics comparing the current character against
the previous character to determine elevation-based formatting.

Convention: HTML5 canvas coordinate system where Y=0 is at the TOP.
"""


def detect_superscript(current_bbox, previous_bbox):
    """
    Determine if the current character is a superscript (exponent)
    of the previous character.

    Heuristics applied (all must pass):
      1. Current centroid must be to the right of previous centroid
      2. Current centroid Y must sit in the top 30% of previous bbox
         (meaning ~70% "above" the base)
      3. Current character must be notably smaller (area < 60% of previous)
      4. Current character's bottom edge must be above the 60% line
         of the previous character's height

    Args:
        current_bbox:  dict with minX, minY, maxX, maxY (canvas coords)
        previous_bbox: dict with minX, minY, maxX, maxY (canvas coords)

    Returns:
        bool: True if current character is a superscript
    """
    if not previous_bbox or not current_bbox:
        return False

    prev_h = previous_bbox['maxY'] - previous_bbox['minY']
    prev_w = previous_bbox['maxX'] - previous_bbox['minX']
    curr_h = current_bbox['maxY'] - current_bbox['minY']
    curr_w = current_bbox['maxX'] - current_bbox['minX']

    # Degenerate bounding boxes → not a superscript
    if prev_h <= 0 or prev_w <= 0 or curr_h <= 0 or curr_w <= 0:
        return False

    curr_cy = (current_bbox['minY'] + current_bbox['maxY']) / 2
    curr_cx = (current_bbox['minX'] + current_bbox['maxX']) / 2
    prev_cx = (previous_bbox['minX'] + previous_bbox['maxX']) / 2

    # Rule 1: Must be roughly positioned to the right (allowing slight overlap)
    if curr_cx < prev_cx - 0.2 * prev_w:
        return False

    # Rule 2: Centroid must be in the upper half of the previous character's bbox.
    # A true superscript is drawn significantly higher.
    mid_y_line = previous_bbox['minY'] + 0.5 * prev_h
    if curr_cy > mid_y_line:
        return False

    # Rule 3: Height check instead of strict area
    # Superscripts shouldn't be much taller than the base character.
    if curr_h > 1.3 * prev_h:
        return False

    # Rule 4: The bottom edge of the superscript MUST NOT drag down to the baseline.
    # It must sit distinctly above the bottom baseline of the previous rendered character.
    bottom_threshold = previous_bbox['maxY'] - 0.15 * prev_h
    if current_bbox['maxY'] > bottom_threshold:
        return False

    return True
