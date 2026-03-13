# =============================================================
# modules/mouse_controller.py
#
# Maps hand landmark positions → actual OS mouse movements.
#
# WHAT THIS MODULE DOES:
#   - Takes the index fingertip position from landmarks
#   - Maps it from camera pixel coords → screen pixel coords
#   - Moves the system cursor via PyAutoGUI
#   - Handles click, drag based on gesture
#   - Applies smoothing so the cursor doesn't jump
#
# WHY COORDINATE MAPPING IS NEEDED:
#   Camera frame:  e.g. 1280 x 720 pixels
#   Screen:        e.g. 1920 x 1080 pixels
#   These are different sizes AND the usable tracking zone
#   (after margins) is smaller than the full frame.
#
#   We map: camera_zone → screen_size using linear interpolation
#   numpy.interp(value, [in_min, in_max], [out_min, out_max])
#
# MARGIN CONCEPT:
#   Without margins, reaching the screen corners requires moving
#   your finger to the very edge of the camera frame — almost
#   off-screen for the camera. Margins shrink the input zone so
#   "near-edge" finger positions still map to full screen edges.
# =============================================================
# =============================================================
# modules/mouse_controller.py  —  Phase 3 Update
#
# CHANGES FROM ORIGINAL:
#   - select_start: records screen position, begins a real drag
#   - select_end: releases drag at current position
#     → This produces an actual OS-level click+drag selection,
#       identical to holding left mouse button and dragging
#   - Visual: draws selection rectangle on the video frame
#     while selection is in progress
# =============================================================

import numpy as np
import pyautogui
import cv2
import config
from utils.smoother import EMAsmoother

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0.0


class MouseController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        print(f"[MouseController] Screen: {self.screen_w}x{self.screen_h}")

        self.frame_w  = config.FRAME_WIDTH
        self.frame_h  = config.FRAME_HEIGHT
        self.margin_x = config.MOUSE_MARGIN_X
        self.margin_y = config.MOUSE_MARGIN_Y

        self.x_smoother = EMAsmoother(alpha=0.4)
        self.y_smoother = EMAsmoother(alpha=0.4)

        self._dragging      = False
        self._mouse_active  = True

        # Selection state
        self._selecting       = False
        self._select_start_px = None   # Camera-space (x,y) of select start
        self._select_end_px   = None   # Camera-space (x,y) of select end (live)
        self._select_start_sc = None   # Screen-space (x,y) of select start

    # ----------------------------------------------------------
    def _to_screen(self, finger_x, finger_y):
        """Map camera pixel coords → screen pixel coords."""
        sx = np.interp(finger_x,
                       [self.margin_x, self.frame_w - self.margin_x],
                       [0, self.screen_w])
        sy = np.interp(finger_y,
                       [self.margin_y, self.frame_h - self.margin_y],
                       [0, self.screen_h])
        sx = self.x_smoother.update(sx, sy)[0]
        sy = self.y_smoother.update(sx, sy)[1]
        return int(max(0, min(self.screen_w - 1, sx))), \
               int(max(0, min(self.screen_h - 1, sy)))

    # ----------------------------------------------------------
    def update(self, landmarks, gesture, engine=None):
        """
        engine: GestureEngine instance — needed to write back
                select_start_pos / select_end_pos for other modules.
        """
        if not self._mouse_active:
            return
        if landmarks is None:
            if self._dragging:
                pyautogui.mouseUp()
                self._dragging = False
            return

        finger_x, finger_y = landmarks[8]   # index fingertip
        sx, sy = self._to_screen(finger_x, finger_y)

        # --------------------------------------------------
        if gesture == "click":
            pyautogui.click(sx, sy)

        elif gesture == "select_start":
            # Record start point, press and HOLD mouse button
            self._selecting       = True
            self._select_start_px = (int(finger_x), int(finger_y))
            self._select_start_sc = (sx, sy)
            self._select_end_px   = (int(finger_x), int(finger_y))
            pyautogui.moveTo(sx, sy)
            pyautogui.mouseDown()           # Hold left button down
            if engine:
                engine.select_start_pos = (sx, sy)
            print(f"[Mouse] Selection START at screen ({sx}, {sy})")

        elif gesture == "select_end":
            # Release mouse button → completes the drag selection
            if self._selecting:
                pyautogui.moveTo(sx, sy)
                pyautogui.mouseUp()         # Release left button
                self._selecting       = False
                self._select_end_px   = (int(finger_x), int(finger_y))
                if engine:
                    engine.select_end_pos = (sx, sy)
                print(f"[Mouse] Selection END at screen ({sx}, {sy})")

        elif gesture == "drag":
            if not self._dragging:
                pyautogui.mouseDown(sx, sy)
                self._dragging = True
            else:
                pyautogui.moveTo(sx, sy)

        elif gesture in ("idle", "draw", "erase", "filter_next",
                         "filter_prev", "open_palm"):
            if self._dragging:
                pyautogui.mouseUp()
                self._dragging = False
            pyautogui.moveTo(sx, sy)

        elif gesture == "none":
            if self._dragging:
                pyautogui.mouseUp()
                self._dragging = False

        # Update live end point during active selection
        if self._selecting and gesture not in ("select_start", "select_end"):
            self._select_end_px = (int(finger_x), int(finger_y))
            pyautogui.moveTo(sx, sy)   # Keep dragging to current position

    # ----------------------------------------------------------
    def draw_hud(self, frame, landmarks):
        """
        Draw cursor indicator + selection rectangle on frame.
        """
        if landmarks is None:
            return frame

        finger_x, finger_y = landmarks[8]
        cx, cy = int(finger_x), int(finger_y)

        # Cursor dot
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3,  (0, 255, 0), -1)

        # Selection rectangle (camera-space)
        if self._selecting and self._select_start_px is not None:
            x1, y1 = self._select_start_px
            x2, y2 = self._select_end_px if self._select_end_px else (cx, cy)

            # Semi-transparent fill
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 120, 255), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

            # Solid border
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)

            # Corner dots
            for pt in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                cv2.circle(frame, pt, 5, (0, 220, 255), -1)

            # Dimensions label
            w_px = abs(x2 - x1)
            h_px = abs(y2 - y1)
            cv2.putText(frame, f"{w_px}x{h_px}px",
                        (min(x1,x2), min(y1,y2) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 220, 255), 1, cv2.LINE_AA)

        return frame

    def toggle(self):
        self._mouse_active = not self._mouse_active
        print(f"[MouseController] {'ENABLED' if self._mouse_active else 'DISABLED'}")