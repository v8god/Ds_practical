# =============================================================
# modules/drawing_canvas.py
#
# Virtual drawing canvas — draws in the air using finger movement.
#
# WHAT THIS MODULE DOES:
#   - Maintains a transparent overlay the same size as the camera frame
#   - When gesture = "draw": tracks index fingertip and draws strokes
#   - When gesture = "erase": erases drawn content near fingertip
#   - Blends the canvas onto the video frame each tick
#   - Saves drawing to file on request
#
# HOW DRAWING WORKS:
#   - Each "draw" gesture frame: record current index fingertip (x, y)
#   - Connect consecutive points with cv2.line → smooth stroke
#   - Points are stored in self.strokes for Phase 4 (3D generation)
#
# HOW ERASING WORKS:
#   - When gesture = "erase": draw a filled black circle on the canvas
#     at the fingertip position with radius = config.ERASER_RADIUS
#   - Black on a black canvas = transparent (since we use black as "empty")
#   - Show a visual eraser circle indicator on the video frame
#
# STROKE STORAGE FORMAT:
#   self.strokes = [
#       [(x1,y1), (x2,y2), ...],   ← stroke 1 (one continuous drag)
#       [(x1,y1), (x2,y2), ...],   ← stroke 2
#       ...
#   ]
#   A new stroke starts when draw gesture begins after a non-draw frame.
#   This is how Phase 4 knows which points are connected.
# =============================================================
# =============================================================
# modules/drawing_canvas.py
#
# Virtual drawing canvas — draws in the air using finger movement.
#
# WHAT THIS MODULE DOES:
#   - Maintains a transparent overlay the same size as the camera frame
#   - When gesture = "draw": tracks index fingertip and draws strokes
#   - When gesture = "erase": erases drawn content near fingertip
#   - Blends the canvas onto the video frame each tick
#   - Saves drawing to file on request
#
# HOW DRAWING WORKS:
#   - Each "draw" gesture frame: record current index fingertip (x, y)
#   - Connect consecutive points with cv2.line → smooth stroke
#   - Points are stored in self.strokes for Phase 4 (3D generation)
#
# HOW ERASING WORKS:
#   - When gesture = "erase": draw a filled black circle on the canvas
#     at the fingertip position with radius = config.ERASER_RADIUS
#   - Black on a black canvas = transparent (since we use black as "empty")
#   - Show a visual eraser circle indicator on the video frame
#
# STROKE STORAGE FORMAT:
#   self.strokes = [
#       [(x1,y1), (x2,y2), ...],   ← stroke 1 (one continuous drag)
#       [(x1,y1), (x2,y2), ...],   ← stroke 2
#       ...
#   ]
#   A new stroke starts when draw gesture begins after a non-draw frame.
#   This is how Phase 4 knows which points are connected.
# =============================================================

import cv2
import numpy as np
import os
import datetime
import config
from utils.overlay_utils import alpha_blend


class DrawingCanvas:
    def __init__(self, width, height):
        """
        width, height : dimensions of the camera frame.
        Must match exactly so the overlay aligns with the video.
        """
        self.width  = width
        self.height = height

        # The canvas: a black BGR image, same size as camera frame.
        # Black = transparent when we blend it (black adds nothing to video).
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Stroke storage for Phase 4
        self.strokes        = []        # List of strokes (each stroke = list of points)
        self._current_stroke = []       # Points in the stroke being drawn right now

        # Track whether we were drawing last frame
        # so we know when a new stroke begins
        self._was_drawing   = False

        # Drawing settings (from config)
        self.draw_color     = config.DRAW_COLOR
        self.draw_thickness = config.DRAW_THICKNESS
        self.eraser_radius  = config.ERASER_RADIUS
        self.canvas_alpha   = config.CANVAS_ALPHA

        # Last fingertip position (needed to draw LINE from prev to current)
        self._prev_point    = None

        print(f"[DrawingCanvas] Initialized {width}x{height}")

    # ----------------------------------------------------------
    # MAIN UPDATE — call once per frame
    # ----------------------------------------------------------
    def update(self, frame, landmarks, gesture):
        """
        Update canvas state based on current gesture and landmarks.
        Then blend canvas onto frame and return the result.

        Parameters:
            frame     : np.ndarray (BGR) — current camera frame (after filter)
            landmarks : list of 21 (x,y) tuples or None
            gesture   : str from GestureEngine

        Returns:
            frame with canvas blended on top
        """
        if landmarks is None:
            # Hand left — end any active stroke cleanly
            self._end_stroke()
            self._prev_point = None
            # Still blend whatever is on canvas
            return self._blend(frame)

        # Index fingertip = landmark 8
        finger_x = int(landmarks[8][0])
        finger_y = int(landmarks[8][1])

        if gesture == "draw":
            self._handle_draw(finger_x, finger_y)

        elif gesture == "erase":
            self._handle_erase(finger_x, finger_y)
            self._end_stroke()
            self._prev_point = None

        else:
            # Any other gesture ends the current stroke
            self._end_stroke()
            self._prev_point = None

        self._was_drawing = (gesture == "draw")

        # Blend canvas onto frame
        result = self._blend(frame)

        # Draw HUD indicators on top of blend
        if gesture == "erase":
            self._draw_eraser_indicator(result, finger_x, finger_y)

        if gesture == "draw":
            self._draw_pen_indicator(result, finger_x, finger_y)

        return result

    # ----------------------------------------------------------
    # DRAW HANDLER
    # ----------------------------------------------------------
    def _handle_draw(self, x, y):
        """
        Called every frame while gesture == "draw".
        Connects current point to previous point with a line.
        """
        current_point = (x, y)

        if not self._was_drawing or self._prev_point is None:
            # Start of a new stroke — just record the first point
            # Don't draw yet (no prev point to connect to)
            self._current_stroke = [current_point]
        else:
            # Continue stroke — draw line from previous point to current
            cv2.line(
                self._canvas,
                self._prev_point,
                current_point,
                self.draw_color,
                self.draw_thickness,
                cv2.LINE_AA    # Anti-aliased for smooth lines
            )
            # Also draw a filled circle at each point to avoid gaps
            # when moving slowly (line between two same points = nothing)
            cv2.circle(
                self._canvas,
                current_point,
                self.draw_thickness // 2,
                self.draw_color,
                -1
            )
            self._current_stroke.append(current_point)

        self._prev_point = current_point

    # ----------------------------------------------------------
    # ERASE HANDLER
    # ----------------------------------------------------------
    def _handle_erase(self, x, y):
        """
        Called every frame while gesture == "erase".
        Paints a black circle on the canvas (removes drawn content).
        """
        cv2.circle(
            self._canvas,
            (x, y),
            self.eraser_radius,
            (0, 0, 0),   # Black = erase
            -1            # Filled circle
        )

        # Also remove points from self.strokes that fall within erase radius
        # so Phase 4 doesn't try to render erased content
        self._erase_stroke_points(x, y)

    def _erase_stroke_points(self, ex, ey):
        """
        Remove points from self.strokes that are within eraser_radius
        of the erase center (ex, ey).
        Splits strokes if a point in the middle is erased.
        """
        import math
        r = self.eraser_radius
        new_strokes = []

        for stroke in self.strokes:
            # Split stroke at erased points
            current_segment = []
            for (px, py) in stroke:
                dist = math.hypot(px - ex, py - ey)
                if dist <= r:
                    # This point is erased
                    if len(current_segment) > 1:
                        new_strokes.append(current_segment)
                    current_segment = []
                else:
                    current_segment.append((px, py))
            if len(current_segment) > 1:
                new_strokes.append(current_segment)

        self.strokes = new_strokes

    # ----------------------------------------------------------
    # STROKE MANAGEMENT
    # ----------------------------------------------------------
    def _end_stroke(self):
        """Finalize the current stroke and add it to history."""
        if len(self._current_stroke) > 1:
            self.strokes.append(self._current_stroke)
        self._current_stroke = []
        self._was_drawing    = False

    # ----------------------------------------------------------
    # BLEND CANVAS ONTO FRAME
    # ----------------------------------------------------------
    def _blend(self, frame):
        """
        Blend the canvas onto the frame.

        We only blend pixels where the canvas is non-black.
        Black canvas pixels = transparent (show video through).
        Colored pixels = blend with video at canvas_alpha opacity.
        """
        # Create a mask of all canvas pixels that have any color
        # (i.e. not pure black)
        gray_canvas = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
        _, mask     = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)

        # Dilate the mask slightly to smooth blend edges
        kernel = np.ones((3, 3), np.uint8)
        mask   = cv2.dilate(mask, kernel, iterations=1)

        # Where mask > 0: blend canvas onto frame
        # Where mask = 0: keep original frame pixel
        result = frame.copy()
        mask3  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 3-channel mask

        # Blend only in the drawn regions
        blended = cv2.addWeighted(
            self._canvas, self.canvas_alpha,
            frame,        1.0 - self.canvas_alpha,
            0
        )

        # Apply blend only where mask says there's drawn content
        result = np.where(mask3 > 0, blended, frame)
        return result.astype(np.uint8)

    # ----------------------------------------------------------
    # VISUAL INDICATORS
    # ----------------------------------------------------------
    def _draw_eraser_indicator(self, frame, x, y):
        """Shows a white circle outline where the eraser is active."""
        cv2.circle(frame, (x, y), self.eraser_radius,
                   (255, 255, 255), 2)
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

    def _draw_pen_indicator(self, frame, x, y):
        """Shows a small colored dot at the pen tip."""
        cv2.circle(frame, (x, y), self.draw_thickness + 2,
                   self.draw_color, -1)

    # ----------------------------------------------------------
    # CANVAS CONTROLS (called from main.py key handlers)
    # ----------------------------------------------------------
    def clear(self):
        """Wipe the entire canvas and all stored strokes."""
        self._canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.strokes         = []
        self._current_stroke = []
        self._prev_point     = None
        self._was_drawing    = False
        print("[DrawingCanvas] Canvas cleared.")

    def save(self):
        """
        Save the current canvas as a PNG file.
        Saves to config data/drawings/ folder with a timestamp filename.
        """
        save_dir = os.path.join("data", "drawings")
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = os.path.join(save_dir, f"drawing_{timestamp}.png")

        # Save canvas with black background
        cv2.imwrite(filename, self._canvas)
        print(f"[DrawingCanvas] Saved to: {filename}")
        return filename

    def draw_hud(self, frame, gesture):
        """
        Shows drawing mode status on screen.
        """
        from utils.overlay_utils import draw_text_with_bg

        if gesture == "draw":
            label = "✏ DRAWING"
            color = self.draw_color
        elif gesture == "erase":
            label = "◯ ERASING"
            color = (255, 255, 255)
        else:
            return frame

        draw_text_with_bg(frame, label, (10, 95),
                          font_scale=0.6,
                          text_color=color,
                          bg_color=(0, 0, 0))
        return frame


# -------------------------------------------------------------
# STANDALONE TEST:  python modules/drawing_canvas.py
#
# - Point index finger → draw
# - Make a fist        → erase
# - Press 'c'          → clear canvas
# - Press 's'          → save drawing
# - Press 'q'          → quit
# -------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")

    from modules.camera        import Camera
    from modules.hand_tracker  import HandTracker
    from modules.gesture_engine import GestureEngine

    cam     = Camera()
    cam.open()

    tracker = HandTracker()
    engine  = GestureEngine()
    canvas  = DrawingCanvas(config.FRAME_WIDTH, config.FRAME_HEIGHT)

    print("Drawing canvas test.")
    print("  Index finger only → draw")
    print("  Closed fist       → erase")
    print("  c → clear  |  s → save  |  q → quit")

    while True:
        frame = cam.read()
        if frame is None:
            break

        landmarks = tracker.update(frame)
        gesture   = engine.update(landmarks)

        tracker.draw(frame)
        frame = canvas.update(frame, landmarks, gesture)
        canvas.draw_hud(frame, gesture)
        engine.draw_hud(frame)

        cv2.imshow("Drawing Canvas Test", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas.clear()
        elif key == ord('s'):
            canvas.save()

    cam.release()
    cv2.destroyAllWindows()


# =============================================================
# ShapeCanvas — extends DrawingCanvas with snapped shape drawing
#
# HOW SHAPE DRAWING WORKS:
#   - User enters draw mode (index finger up)
#   - Thumb+pinky pinch cycles shape mode (via gesture_engine)
#   - When shape_mode != "freehand":
#       → First point touched = shape anchor (start)
#       → As finger moves, a PREVIEW of the shape is shown
#       → When draw gesture ends (finger folds), shape is COMMITTED
#         to the canvas
#
# This gives you perfect geometric shapes drawn in the air.
# =============================================================

import math as _math


class ShapeCanvas(DrawingCanvas):
    """
    Drop-in replacement for DrawingCanvas that adds shape snapping.
    main.py uses this instead of DrawingCanvas from Phase 3 onwards.
    """

    def __init__(self, width, height):
        super().__init__(width, height)
        self._shape_anchor  = None    # First point of current shape
        self._shape_preview = None    # Current end point (live preview)
        self._in_shape      = False   # Are we currently drawing a shape?

    def update(self, frame, landmarks, gesture, shape_mode="freehand"):
        """
        Same signature as DrawingCanvas.update() plus shape_mode.
        shape_mode comes from engine.shape_mode each frame.
        """
        if landmarks is None:
            if self._in_shape:
                self._commit_shape(shape_mode)
            self._end_stroke()
            self._prev_point = None
            return self._blend(frame)

        finger_x = int(landmarks[8][0])
        finger_y = int(landmarks[8][1])

        if gesture == "draw":
            if shape_mode == "freehand":
                # Original freehand drawing
                self._in_shape     = False
                self._shape_anchor = None
                self._handle_draw(finger_x, finger_y)
            else:
                # Shape drawing
                if not self._in_shape or self._shape_anchor is None:
                    # First frame of shape — set anchor
                    self._shape_anchor  = (finger_x, finger_y)
                    self._shape_preview = (finger_x, finger_y)
                    self._in_shape      = True
                else:
                    # Update live end point
                    self._shape_preview = (finger_x, finger_y)

        elif gesture == "erase":
            if self._in_shape:
                # Cancel shape on erase
                self._in_shape     = False
                self._shape_anchor = None
            self._handle_erase(finger_x, finger_y)
            self._end_stroke()
            self._prev_point = None

        else:
            # Gesture changed → commit shape if one was in progress
            if self._in_shape and self._shape_anchor and self._shape_preview:
                self._commit_shape(shape_mode)
            self._in_shape     = False
            self._shape_anchor = None
            self._end_stroke()
            self._prev_point = None

        self._was_drawing = (gesture == "draw")

        # Blend committed canvas
        result = self._blend(frame)

        # Draw live shape preview on top (not committed yet)
        if self._in_shape and self._shape_anchor and self._shape_preview and gesture == "draw":
            self._draw_preview(result, shape_mode)

        # Draw indicators
        if gesture == "erase":
            self._draw_eraser_indicator(result, finger_x, finger_y)
        if gesture == "draw":
            self._draw_pen_indicator(result, finger_x, finger_y)

        return result

    def _commit_shape(self, shape_mode):
        """Draw the final shape onto the permanent canvas."""
        if self._shape_anchor is None or self._shape_preview is None:
            return

        x1, y1 = self._shape_anchor
        x2, y2 = self._shape_preview
        color   = self.draw_color
        thick   = self.draw_thickness

        if shape_mode == "line":
            cv2.line(self._canvas, (x1, y1), (x2, y2),
                     color, thick, cv2.LINE_AA)

        elif shape_mode == "rectangle":
            cv2.rectangle(self._canvas, (x1, y1), (x2, y2),
                          color, thick)

        elif shape_mode == "circle":
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            r  = int(_math.hypot(x2 - cx, y2 - cy))
            cv2.circle(self._canvas, (cx, cy), r,
                       color, thick, cv2.LINE_AA)

        elif shape_mode == "triangle":
            # Equilateral-ish triangle: anchor = top center, preview = bottom right
            # Three points derived from the bounding box
            tx, ty = (x1 + x2) // 2, y1          # Top center
            bl     = (x1, y2)                      # Bottom left
            br     = (x2, y2)                      # Bottom right
            pts    = np.array([tx, ty, bl[0], bl[1], br[0], br[1]],
                              dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(self._canvas, [pts], isClosed=True,
                          color=color, thickness=thick,
                          lineType=cv2.LINE_AA)

        # Store shape as a stroke for Phase 4
        self.strokes.append([self._shape_anchor, self._shape_preview])

        self._shape_anchor  = None
        self._shape_preview = None
        self._in_shape      = False
        print(f"[ShapeCanvas] Committed shape: {shape_mode}")

    def _draw_preview(self, frame, shape_mode):
        """Draw a semi-transparent preview of the shape being drawn."""
        x1, y1 = self._shape_anchor
        x2, y2 = self._shape_preview
        color   = self.draw_color
        thick   = max(1, self.draw_thickness - 1)

        # Dashed/thinner line for preview
        preview_color = tuple(min(255, c + 80) for c in color)  # Slightly brighter

        if shape_mode == "line":
            cv2.line(frame, (x1, y1), (x2, y2),
                     preview_color, thick, cv2.LINE_AA)

        elif shape_mode == "rectangle":
            cv2.rectangle(frame, (x1, y1), (x2, y2), preview_color, thick)
            # Anchor corner dot
            cv2.circle(frame, (x1, y1), 5, preview_color, -1)

        elif shape_mode == "circle":
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            r  = int(_math.hypot(x2 - cx, y2 - cy))
            cv2.circle(frame, (cx, cy), r, preview_color, thick, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, preview_color, -1)  # Center dot

        elif shape_mode == "triangle":
            tx, ty = (x1 + x2) // 2, y1
            bl     = (x1, y2)
            br     = (x2, y2)
            pts    = np.array([tx, ty, bl[0], bl[1], br[0], br[1]],
                              dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True,
                          color=preview_color, thickness=thick,
                          lineType=cv2.LINE_AA)

        # Distance label
        d = int(_math.hypot(x2 - x1, y2 - y1))
        cv2.putText(frame, f"{shape_mode} ({d}px)",
                    (min(x1,x2), min(y1,y2) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    preview_color, 1, cv2.LINE_AA)