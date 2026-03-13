# =============================================================
# utils/overlay_utils.py
#
# Reusable helper functions used by multiple modules.
# Kept here so no module has to rewrite the same utility code.
# =============================================================

import cv2
import numpy as np


def alpha_blend(background, overlay, alpha):
    """
    Blends an overlay image onto a background image.

    Parameters:
        background : np.ndarray  — the base frame (e.g. camera feed)
        overlay    : np.ndarray  — the layer to blend on top (same size)
        alpha      : float       — 0.0 = invisible overlay, 1.0 = full overlay

    Returns:
        Blended frame as np.ndarray.

    Why cv2.addWeighted instead of simple array math?
    It's faster (uses SIMD hardware acceleration) and handles
    uint8 overflow (values > 255) automatically via saturation.
    """
    # Clamp alpha to valid range just in case
    alpha = max(0.0, min(1.0, alpha))
    return cv2.addWeighted(overlay, alpha, background, 1.0 - alpha, 0)


def draw_text_with_bg(frame, text, pos, font_scale=0.6, text_color=(255, 255, 255),
                       bg_color=(0, 0, 0), thickness=2, padding=6):
    """
    Draws text with a filled background rectangle behind it.
    Plain cv2.putText is unreadable on bright/busy video backgrounds.

    Parameters:
        frame       : np.ndarray — frame to draw on (modified in-place)
        text        : str        — text to display
        pos         : (x, y)     — top-left position of the text box
        font_scale  : float      — text size
        text_color  : (B, G, R)
        bg_color    : (B, G, R)
        thickness   : int        — text stroke thickness
        padding     : int        — pixels of padding around text inside box
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Measure how big the text will be so we can draw the box first
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = pos
    # Box coords
    box_x1 = x - padding
    box_y1 = y - text_h - padding
    box_x2 = x + text_w + padding
    box_y2 = y + baseline + padding

    # Draw filled background
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)

    # Draw text on top
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_landmark_point(frame, x, y, color=(0, 255, 255), radius=5):
    """
    Draws a single landmark dot on the frame.
    Used by hand_tracker.py to visualize finger positions.
    """
    cv2.circle(frame, (int(x), int(y)), radius, color, -1)
    # Thin white ring around the dot for visibility on dark & light backgrounds
    cv2.circle(frame, (int(x), int(y)), radius, (255, 255, 255), 1)


def draw_connection_line(frame, p1, p2, color=(100, 100, 100), thickness=1):
    """
    Draws a line between two (x, y) landmark points.
    Used to render the hand skeleton.
    """
    cv2.line(frame, (int(p1[0]), int(p1[1])),
             (int(p2[0]), int(p2[1])), color, thickness, cv2.LINE_AA)


def resize_with_aspect(image, target_w=None, target_h=None):
    """
    Resizes an image while preserving aspect ratio.
    Provide either target_w OR target_h (not both).

    Returns resized image.
    """
    h, w = image.shape[:2]

    if target_w is not None:
        scale = target_w / w
    elif target_h is not None:
        scale = target_h / h
    else:
        return image  # Nothing to do

    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)