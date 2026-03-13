# =============================================================
# modules/ar_3d.py
#
# Generates a pseudo-3D object from the strokes drawn on the
# canvas and renders it as an AR overlay on the video frame.
#
# WHY "PSEUDO-3D" AND NOT REAL 3D:
#   Real 3D requires a depth camera (like Intel RealSense) or
#   stereo cameras. We only have a regular webcam.
#   Instead we simulate depth using these techniques:
#     1. Extrusion: take 2D stroke paths and push them "back"
#        by an offset to create a shadow/depth copy
#     2. Perspective projection: points further "back" appear
#        slightly smaller and shifted (vanishing point illusion)
#     3. Rotation: rotate the entire object around Y and X axes
#        using a rotation matrix, then project back to 2D
#     4. Depth shading: faces further from viewer are darker
#
# THE PIPELINE:
#   strokes (2D pixel points)
#     → center and normalize to 3D space
#     → apply rotation matrix (spin the object)
#     → project back to 2D screen coords
#     → draw front face, back face, connecting edges
#
# CONTROLS (from main.py):
#   3D mode is toggled with the 't' key
#   When active, the last committed drawing is extruded into 3D
#   Object auto-rotates slowly, or can be rotated by hand gesture
# =============================================================
# =============================================================
# modules/ar_3d.py  — rotation fix
#
# ROTATION SYSTEM (fixed):
#   BEFORE: wrist-tilt ran every frame hand was visible → constant jitter
#   NOW:
#     - Auto-spin when NO hand detected (slow, nice idle animation)
#     - Manual rotation ONLY when gesture == "open_palm"
#       → tracks how much palm MOVED since last frame (delta-based)
#       → moving palm right  = rotate right on Y axis
#       → moving palm left   = rotate left  on Y axis
#       → moving palm up     = rotate up    on X axis
#       → moving palm down   = rotate down  on X axis
#     - Any other gesture = rotation freezes at current angle
#
# This gives full intentional control with zero accidental rotation.
# =============================================================
# =============================================================
# modules/ar_3d.py  — rotation fix
#
# ROTATION SYSTEM (fixed):
#   BEFORE: wrist-tilt ran every frame hand was visible → constant jitter
#   NOW:
#     - Auto-spin when NO hand detected (slow, nice idle animation)
#     - Manual rotation ONLY when gesture == "open_palm"
#       → tracks how much palm MOVED since last frame (delta-based)
#       → moving palm right  = rotate right on Y axis
#       → moving palm left   = rotate left  on Y axis
#       → moving palm up     = rotate up    on X axis
#       → moving palm down   = rotate down  on X axis
#     - Any other gesture = rotation freezes at current angle
#
# This gives full intentional control with zero accidental rotation.
# =============================================================

import cv2
import numpy as np
import math
import config


class AR3DEngine:
    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.cx     = width  // 2
        self.cy     = height // 2
        self.focal  = width

        self.rot_x  = 0.15
        self.rot_y  = 0.0

        # Auto-spin speed (active only when no hand in frame)
        self.auto_rot_y = 0.008
        self.auto_rot_x = 0.003

        # How sensitive manual rotation is
        # Higher = faster rotation per pixel of palm movement
        self.rot_sensitivity = 0.008

        # Track previous palm center for delta calculation
        self._prev_palm_x = None
        self._prev_palm_y = None

        self._obj_points = []
        self._obj_colors = []
        self._obj_active = False

        self._z_spread     = 30.0
        self._z_spread_max = 200.0
        self._z_spread_min = 5.0

        self._palette = [
            (0,   255, 255),
            (255, 100, 0  ),
            (100, 255, 100),
            (255, 50,  200),
            (50,  150, 255),
            (255, 220, 50 ),
        ]
        print(f"[AR3D] Initialized {width}x{height}")

    # ----------------------------------------------------------
    def build_from_strokes(self, strokes_2d):
        if not strokes_2d:
            print("[AR3D] No strokes to build from.")
            return

        self._obj_points = []
        self._obj_colors = []

        all_pts = [pt for stroke in strokes_2d for pt in stroke]
        if not all_pts:
            return

        cx = sum(p[0] for p in all_pts) / len(all_pts)
        cy = sum(p[1] for p in all_pts) / len(all_pts)

        num_strokes = len(strokes_2d)
        for i, stroke in enumerate(strokes_2d):
            if len(stroke) < 2:
                continue
            z_range   = self._z_spread * max(1, num_strokes - 1)
            z         = -z_range / 2 + i * self._z_spread
            stroke_3d = [(px - cx, py - cy, z) for (px, py) in stroke]
            self._obj_points.append(stroke_3d)
            self._obj_colors.append(self._palette[i % len(self._palette)])

        self._obj_active = True
        self.rot_x = 0.15
        self.rot_y = 0.0
        self._prev_palm_x = None
        self._prev_palm_y = None
        print(f"[AR3D] Built from {len(self._obj_points)} strokes.")

    # ----------------------------------------------------------
    def _Rx(self, a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

    def _Ry(self, a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)

    def _project(self, x, y, z):
        d = self.focal + z
        if abs(d) < 1e-6: d = 1e-6
        return int(self.focal * x / d + self.cx), \
               int(self.focal * y / d + self.cy)

    def _get_palm_center(self, landmarks):
        """
        Palm center = average of wrist + 4 finger bases.
        More stable than wrist alone — less affected by finger movement.
        """
        indices = [0, 5, 9, 13, 17]   # wrist + finger MCP joints
        xs = [landmarks[i][0] for i in indices]
        ys = [landmarks[i][1] for i in indices]
        return sum(xs)/len(xs), sum(ys)/len(ys)

    # ----------------------------------------------------------
    def update(self, frame, landmarks=None, gesture=None):
        """
        Update rotation and render.

        gesture == "open_palm" → manual rotation via palm movement
        gesture == None        → auto-spin
        anything else          → freeze rotation
        """
        if not self._obj_active:
            return frame

        # ── ROTATION LOGIC ──────────────────────────────────
        if landmarks is None or gesture is None:
            # No hand → auto-spin
            self.rot_x += self.auto_rot_x
            self.rot_y += self.auto_rot_y
            self._prev_palm_x = None
            self._prev_palm_y = None

        elif gesture == "drag":
            # Drag gesture = rotate in 3D stroke mode
            # Track palm center delta for smooth rotation
            palm_x, palm_y = self._get_palm_center(landmarks)

            if self._prev_palm_x is not None:
                dx = palm_x - self._prev_palm_x
                dy = palm_y - self._prev_palm_y
                self.rot_y += dx * self.rot_sensitivity
                self.rot_x += dy * self.rot_sensitivity

            self._prev_palm_x = palm_x
            self._prev_palm_y = palm_y

        else:
            # Any other gesture → freeze rotation
            self._prev_palm_x = None
            self._prev_palm_y = None

        self.rot_x %= (2 * math.pi)
        self.rot_y %= (2 * math.pi)

        R = self._Ry(self.rot_y) @ self._Rx(self.rot_x)

        # ── RENDER ──────────────────────────────────────────
        for si, stroke_3d in enumerate(self._obj_points):
            color = self._obj_colors[si]
            projected, depths = [], []

            for (x, y, z) in stroke_3d:
                pr = R @ np.array([x, y, z], dtype=np.float32)
                sx, sy = self._project(*pr)
                projected.append((sx, sy))
                depths.append(pr[2])

            if len(projected) < 2:
                continue

            avg_d      = sum(depths) / len(depths)
            depth_norm = max(0.0, min(1.0, (avg_d + 150) / 300))
            fade       = 1.0 - depth_norm * 0.6
            thickness  = max(1, int(2 * (1.0 - depth_norm * 0.5)))
            col        = tuple(int(c * fade) for c in color)

            for i in range(len(projected) - 1):
                p1, p2 = projected[i], projected[i+1]
                if (0<=p1[0]<self.width and 0<=p1[1]<self.height and
                    0<=p2[0]<self.width and 0<=p2[1]<self.height):
                    cv2.line(frame, p1, p2, col, thickness, cv2.LINE_AA)

            for (sx, sy) in projected:
                if 0<=sx<self.width and 0<=sy<self.height:
                    cv2.circle(frame, (sx,sy),
                               max(1, int(3*(1-depth_norm*0.4))), col, -1)

        return frame

    # ----------------------------------------------------------
    def reset_rotation(self):
        self.rot_x = 0.15
        self.rot_y = 0.0
        self._prev_palm_x = None
        self._prev_palm_y = None
        print("[AR3D] Rotation reset.")

    def increase_depth(self):
        self._z_spread = min(self._z_spread_max, self._z_spread + 10)
        # Rebuild won't happen automatically — user needs to re-convert.
        # Print a reminder.
        print(f"[AR3D] Z spread: {self._z_spread:.0f}  "
              f"(press t twice to re-convert with new depth)")

    def decrease_depth(self):
        self._z_spread = max(self._z_spread_min, self._z_spread - 10)
        print(f"[AR3D] Z spread: {self._z_spread:.0f}  "
              f"(press t twice to re-convert with new depth)")

    def clear(self):
        self._obj_points  = []
        self._obj_colors  = []
        self._obj_active  = False
        self._prev_palm_x = None
        self._prev_palm_y = None
        print("[AR3D] Cleared.")

    @property
    def is_active(self):
        return self._obj_active

    def draw_hud(self, frame):
        from utils.overlay_utils import draw_text_with_bg
        if self._obj_active:
            draw_text_with_bg(
                frame,
                "drag gesture = rotate  |  r=reset  z/x=depth",
                (10, frame.shape[0] - 30),
                font_scale=0.5,
                text_color=(0, 255, 255),
                bg_color=(0, 0, 0))
        return frame
