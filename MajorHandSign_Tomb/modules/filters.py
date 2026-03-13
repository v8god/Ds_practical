# =============================================================
# modules/filters.py
#
# All real-time video filters.
# Each filter is a pure function: takes a BGR frame, returns
# a BGR frame. They don't store state — just transform pixels.
#
# The FilterEngine class manages which filter is active and
# exposes a single .apply(frame) method to main.py.
# =============================================================
# =============================================================
# modules/filters.py
#
# All real-time video filters.
# Each filter is a pure function: takes a BGR frame, returns
# a BGR frame. They don't store state — just transform pixels.
#
# The FilterEngine class manages which filter is active and
# exposes a single .apply(frame) method to main.py.
# =============================================================

import cv2
import numpy as np
import config


# -------------------------------------------------------------
# Individual filter functions
# All take: frame (np.ndarray, BGR, uint8)
# All return: filtered frame (np.ndarray, BGR, uint8)
# -------------------------------------------------------------

def filter_none(frame):
    """No filter — return frame as-is."""
    return frame


def filter_bw(frame):
    """
    Black and White.
    Convert to grayscale, then back to BGR so the rest of the
    pipeline (which expects 3 channels) doesn't break.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def filter_sepia(frame):
    """
    Warm sepia tone.
    Sepia is achieved by applying a color matrix transform.
    The matrix values are the standard sepia coefficients —
    they redistribute BGR channels to give a warm brownish look.
    """
    # Convert to float so matrix math doesn't overflow uint8
    frame_float = np.array(frame, dtype=np.float64)

    # Sepia matrix (applied per pixel: new_channel = sum of old channels * weights)
    # Row order: output B, output G, output R
    sepia_matrix = np.array([
        [0.131, 0.534, 0.272],   # Output Blue
        [0.168, 0.686, 0.349],   # Output Green
        [0.189, 0.769, 0.393],   # Output Red
    ])

    sepia_frame = cv2.transform(frame_float, sepia_matrix)

    # Clip values to 0-255 and convert back to uint8
    sepia_frame = np.clip(sepia_frame, 0, 255).astype(np.uint8)
    return sepia_frame


def filter_paris_blue(frame):
    """
    Paris Blue — cool blue/teal tone.
    Achieved by boosting the blue channel and slightly reducing red.
    Inspired by the washed-out cool aesthetic in street photography.
    """
    frame = frame.astype(np.float32)

    # Split into B, G, R channels
    b, g, r = cv2.split(frame)

    # Boost blue, slightly tint green, reduce red
    b = np.clip(b * 1.4, 0, 255)   # Strong blue boost
    g = np.clip(g * 1.1, 0, 255)   # Slight green/teal lift
    r = np.clip(r * 0.75, 0, 255)  # Reduce red (makes it cooler)

    result = cv2.merge([b, g, r]).astype(np.uint8)

    # Add a subtle contrast boost
    result = cv2.convertScaleAbs(result, alpha=1.1, beta=-10)
    return result


def filter_sketch(frame):
    """
    Pencil sketch effect.
    Algorithm:
    1. Convert to grayscale
    2. Invert the grayscale
    3. Apply heavy Gaussian blur to the inverted image
    4. Divide the grayscale by the inverted-blurred image
       This amplifies edges (where gray / near-white ≈ gray,
       but gray / near-black ≈ very bright) producing sketch lines
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0)

    # Dodge blend: divide gray by blurred (scaled to avoid div-by-zero)
    # np.where avoids dividing by 0 — returns 255 wherever blurred == 255
    sketch = np.where(
        blurred == 255,
        255,
        np.minimum(255, gray.astype(np.int32) * 255 // (255 - blurred.astype(np.int32)))
    ).astype(np.uint8)

    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def filter_edge(frame):
    """
    Edge detection using Canny algorithm.
    Shows only the outlines/edges of objects in white on black.
    Good for a "x-ray" or "blueprint" look.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # GaussianBlur first: reduces noise so Canny doesn't detect noise as edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny thresholds: lower=50, upper=150
    # Edges with gradient > 150 are definitely edges
    # Edges between 50-150 are edges only if connected to a strong edge
    # Edges < 50 are discarded
    edges = cv2.Canny(blurred, 50, 150)

    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def filter_cartoon(frame):
    """
    Cartoon / cel-shaded effect.
    Algorithm:
    1. Bilateral filter: smooths colors but preserves edges
       (unlike Gaussian blur which blurs everything including edges)
    2. Detect edges with adaptive threshold on grayscale
    3. Combine: smooth color + black edges = cartoon look
    """
    # Step 1: Smooth colors while keeping edges sharp
    # bilateralFilter params: d=9 (neighborhood), sigmaColor=75, sigmaSpace=75
    # Higher sigmaColor = more colors mixed (more smoothed)
    smooth = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 2: Get edges from grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)   # Median blur removes salt-and-pepper noise

    # adaptiveThreshold finds edges based on local pixel neighborhoods
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,    # Neighborhood size (must be odd)
        C=2             # Constant subtracted from mean — tune this for more/less edge detail
    )
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Step 3: Multiply smooth colors by edge mask
    # Where edges=0 (black line), result=0. Where edges=255, result=smooth pixel.
    cartoon = cv2.bitwise_and(smooth, edges)
    return cartoon


# -------------------------------------------------------------
# Map filter name strings to functions
# This lets us look up the function by name from config.FILTER_NAMES
# -------------------------------------------------------------
FILTER_MAP = {
    "none":       filter_none,
    "bw":         filter_bw,
    "sepia":      filter_sepia,
    "paris_blue": filter_paris_blue,
    "sketch":     filter_sketch,
    "edge":       filter_edge,
    "cartoon":    filter_cartoon,
}


# -------------------------------------------------------------
# FilterEngine class — used by main.py
# -------------------------------------------------------------
class FilterEngine:
    def __init__(self):
        self.active_index = config.DEFAULT_FILTER
        self.filter_names = config.FILTER_NAMES

    def apply(self, frame):
        """
        Apply the currently active filter to the frame.
        Returns the filtered frame.
        """
        name = self.filter_names[self.active_index]
        fn = FILTER_MAP.get(name, filter_none)
        return fn(frame)

    def cycle(self):
        """
        Move to the next filter in the list.
        Wraps around to 0 after the last one.
        Called when the user presses 'f' or does the swipe gesture.
        """
        self.active_index = (self.active_index + 1) % len(self.filter_names)
        print(f"[Filter] Active: {self.current_name}")

    def prev(self):
        """Move to the previous filter. Wraps around."""
        self.active_index = (self.active_index - 1) % len(self.filter_names)
        print(f"[Filter] Active: {self.current_name}")

    def set_filter(self, name):
        """
        Set filter by name directly (used by gesture engine in Phase 2).
        Silently ignores unknown names.
        """
        if name in self.filter_names:
            self.active_index = self.filter_names.index(name)

    @property
    def current_name(self):
        return self.filter_names[self.active_index]

    def draw_hud(self, frame):
        """
        Draws a small filter name label in the top-right of the frame.
        So the user always knows which filter is active.
        """
        from utils.overlay_utils import draw_text_with_bg

        label = f"Filter: {self.current_name.upper()}"
        # Position: top-right, with a little margin
        x = frame.shape[1] - 220
        y = 28
        draw_text_with_bg(frame, label, (x, y),
                          font_scale=0.6,
                          text_color=(0, 255, 255),
                          bg_color=(0, 0, 0))
        return frame


# -------------------------------------------------------------
# STANDALONE TEST — run directly:
#   python modules/filters.py
# Press 'f' to cycle filters, 'q' to quit.
# -------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    engine = FilterEngine()

    print("Filter test. Press 'f' to cycle, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        filtered = engine.apply(frame)
        engine.draw_hud(filtered)

        cv2.imshow("Filter Test", filtered)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('f'):
            engine.cycle()

    cap.release()
    cv2.destroyAllWindows()