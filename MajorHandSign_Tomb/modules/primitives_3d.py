# =============================================================
# modules/primitives_3d.py
#
# 3D Primitive Shapes Engine
#
# SHAPES SUPPORTED:
#   - Sphere    (radius)
#   - Cuboid    (width, height, depth)
#   - Cone      (radius, height)
#   - Cylinder  (radius, height)
#
# HOW IT WORKS:
#   1. Each shape is a mesh of 3D vertices + edges
#   2. Vertices are generated mathematically (not from camera)
#   3. Each frame: rotation matrix applied → perspective project → draw
#   4. Shapes live in a scene list — add, select, delete individually
#
# CONTROLS (all from main.py):
#   - Shape type cycling
#   - Parameter adjustment (size, width, height, radius)
#   - Place shape at screen center
#   - Select shape (click near it)
#   - Delete selected shape
#   - Save scene to PNG
#
# COORDINATE SYSTEM:
#   Each shape has its own local (x,y,z) center in scene space.
#   Scene center = (0, 0, 0) = middle of screen.
# =============================================================

# =============================================================
# modules/primitives_3d.py  — clean rewrite
#
# Proper 3D wireframe primitives: Sphere, Cuboid, Cone,
# Cylinder, Square Pyramid.
#
# FIXES FROM PREVIOUS VERSION:
#   - Objects stay permanently — no auto-deletion
#   - Drag and drop: place object, then drag with index finger
#   - Per-object rotation via wrist tilt
#   - Global rotation via auto-spin when no hand
#   - Proper parameter panel
# =============================================================
# =============================================================
# modules/primitives_3d.py  — clean rewrite
#
# Proper 3D wireframe primitives: Sphere, Cuboid, Cone,
# Cylinder, Square Pyramid.
#
# FIXES FROM PREVIOUS VERSION:
#   - Objects stay permanently — no auto-deletion
#   - Drag and drop: place object, then drag with index finger
#   - Per-object rotation via wrist tilt
#   - Global rotation via auto-spin when no hand
#   - Proper parameter panel
# =============================================================
# =============================================================
# modules/primitives_3d.py  — clean rewrite
#
# Proper 3D wireframe primitives: Sphere, Cuboid, Cone,
# Cylinder, Square Pyramid.
#
# FIXES FROM PREVIOUS VERSION:
#   - Objects stay permanently — no auto-deletion
#   - Drag and drop: place object, then drag with index finger
#   - Per-object rotation via wrist tilt
#   - Global rotation via auto-spin when no hand
#   - Proper parameter panel
# =============================================================

import cv2
import numpy as np
import math
import os
import datetime


# ----------------------------------------------------------
# MESH GENERATORS
# ----------------------------------------------------------

def make_sphere(radius=80, lat=8, lon=12):
    verts, edges = [], []
    for la in range(lat + 1):
        theta = math.pi * la / lat
        for lo in range(lon):
            phi = 2 * math.pi * lo / lon
            verts.append((
                radius * math.sin(theta) * math.cos(phi),
                radius * math.cos(theta),
                radius * math.sin(theta) * math.sin(phi),
            ))
    for la in range(lat + 1):
        for lo in range(lon):
            edges.append((la*lon + lo, la*lon + (lo+1)%lon))
    for la in range(lat):
        for lo in range(lon):
            edges.append((la*lon + lo, (la+1)*lon + lo))
    return verts, edges


def make_cuboid(w=120, h=100, d=80):
    hw, hh, hd = w/2, h/2, d/2
    verts = [
        (-hw,-hh,-hd),(hw,-hh,-hd),(hw,hh,-hd),(-hw,hh,-hd),
        (-hw,-hh, hd),(hw,-hh, hd),(hw,hh, hd),(-hw,hh, hd),
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),   # front
        (4,5),(5,6),(6,7),(7,4),   # back
        (0,4),(1,5),(2,6),(3,7),   # sides
    ]
    return verts, edges


def make_cone(radius=70, height=140, steps=12):
    verts  = [(0, -height/2, 0)]   # apex index 0
    edges  = []
    for i in range(steps):
        a = 2 * math.pi * i / steps
        verts.append((radius*math.cos(a), height/2, radius*math.sin(a)))
    for i in range(steps):
        edges.append((1+i, 1+(i+1)%steps))      # base ring
    for i in range(0, steps, 2):
        edges.append((0, 1+i))                   # apex lines
    return verts, edges


def make_cylinder(radius=60, height=120, steps=12):
    verts, edges = [], []
    hh = height / 2
    for i in range(steps):
        a = 2 * math.pi * i / steps
        verts.append((radius*math.cos(a),  hh, radius*math.sin(a)))
    for i in range(steps):
        a = 2 * math.pi * i / steps
        verts.append((radius*math.cos(a), -hh, radius*math.sin(a)))
    for i in range(steps):
        edges.append((i, (i+1)%steps))            # bottom ring
        edges.append((steps+i, steps+(i+1)%steps))# top ring
    for i in range(0, steps, 2):
        edges.append((i, steps+i))                # verticals
    return verts, edges


def make_square_pyramid(base=120, height=140):
    hb = base / 2
    verts = [
        (-hb,  height/2, -hb),   # 0 base front-left
        ( hb,  height/2, -hb),   # 1 base front-right
        ( hb,  height/2,  hb),   # 2 base back-right
        (-hb,  height/2,  hb),   # 3 base back-left
        (  0, -height/2,   0),   # 4 apex
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # base square
        (0,4),(1,4),(2,4),(3,4),  # apex lines
    ]
    return verts, edges


SHAPE_CYCLE = ["sphere", "cuboid", "cone", "cylinder", "pyramid"]

SHAPE_GENERATORS = {
    "sphere":   make_sphere,
    "cuboid":   make_cuboid,
    "cone":     make_cone,
    "cylinder": make_cylinder,
    "pyramid":  make_square_pyramid,
}

# Parameters: default, min, max, step
SHAPE_PARAMS = {
    "sphere":   {"radius": [80,  20, 200, 10]},
    "cuboid":   {"width":  [120, 20, 300, 10],
                 "height": [100, 20, 300, 10],
                 "depth":  [80,  20, 300, 10]},
    "cone":     {"radius": [70,  20, 200, 10],
                 "height": [140, 30, 300, 10]},
    "cylinder": {"radius": [60,  20, 200, 10],
                 "height": [120, 30, 300, 10]},
    "pyramid":  {"base":   [120, 30, 300, 10],
                 "height": [140, 30, 300, 10]},
}

PALETTE = [
    (0,255,255),(255,100,50),(100,255,100),
    (255,80,200),(80,160,255),(255,220,60),
]


# ----------------------------------------------------------
class SceneObject:
    """One placed 3D shape in the scene."""
    def __init__(self, shape_type, param_vals, color, sx=0, sy=0):
        self.shape_type  = shape_type
        self.param_vals  = dict(param_vals)
        self.color       = color
        self.screen_x    = sx    # screen-space center X (for drag)
        self.screen_y    = sy    # screen-space center Y (for drag)
        self.selected    = False
        self.verts, self.edges = self._build()

    def _build(self):
        gen = SHAPE_GENERATORS[self.shape_type]
        p   = self.param_vals
        if self.shape_type == "sphere":
            return gen(radius=p["radius"])
        elif self.shape_type == "cuboid":
            return gen(w=p["width"], h=p["height"], d=p["depth"])
        elif self.shape_type in ("cone", "cylinder"):
            return gen(radius=p["radius"], height=p["height"])
        elif self.shape_type == "pyramid":
            return gen(base=p["base"], height=p["height"])
        return gen()


# ----------------------------------------------------------
class Primitives3DEngine:
    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.cx     = width  // 2
        self.cy     = height // 2
        self.focal  = width

        # Current shape being configured
        self.cur_shape  = "sphere"
        # Live param values for current shape [val, min, max, step]
        self.cur_params = {k: list(v) for k,v in SHAPE_PARAMS["sphere"].items()}
        self.active_param = "radius"

        # Scene
        self.scene        = []       # list of SceneObject
        self.selected_idx = None

        # Rotation (global, applies to all objects equally)
        self.rot_x           = 0.15
        self.rot_y           = 0.0
        self.auto_rot_y      = 0.008
        self.auto_rot_x      = 0.002
        self.rot_sensitivity = 0.008
        self._prev_palm_x    = None
        self._prev_palm_y    = None

        # Drag state
        self._dragging_idx   = None
        self._drag_offset_x  = 0
        self._drag_offset_y  = 0

        self._color_idx = 0
        print(f"[Primitives3D] Ready — {width}x{height}")

    # ── Shape / param controls ─────────────────────────────

    def cycle_shape(self):
        idx = SHAPE_CYCLE.index(self.cur_shape)
        self.cur_shape  = SHAPE_CYCLE[(idx+1) % len(SHAPE_CYCLE)]
        self.cur_params = {k: list(v) for k,v in SHAPE_PARAMS[self.cur_shape].items()}
        self.active_param = list(self.cur_params.keys())[0]
        print(f"[Primitives3D] Shape → {self.cur_shape.upper()}")

    def cycle_param(self):
        keys = list(self.cur_params.keys())
        idx  = keys.index(self.active_param) if self.active_param in keys else 0
        self.active_param = keys[(idx+1) % len(keys)]
        print(f"[Primitives3D] Param → {self.active_param} = {self.cur_params[self.active_param][0]}")

    def param_up(self):
        p = self.cur_params[self.active_param]   # [val, min, max, step]
        p[0] = min(p[2], p[0] + p[3])
        print(f"[Primitives3D] {self.active_param} = {p[0]}")

    def param_down(self):
        p = self.cur_params[self.active_param]
        p[0] = max(p[1], p[0] - p[3])
        print(f"[Primitives3D] {self.active_param} = {p[0]}")

    # ── Scene controls ─────────────────────────────────────

    def place_shape(self, screen_x=None, screen_y=None):
        """Place current shape at screen center (or given coords)."""
        color = PALETTE[self._color_idx % len(PALETTE)]
        self._color_idx += 1

        sx = screen_x if screen_x is not None else self.cx
        sy = screen_y if screen_y is not None else self.cy

        param_vals = {k: v[0] for k,v in self.cur_params.items()}

        obj = SceneObject(self.cur_shape, param_vals, color, sx, sy)
        self.scene.append(obj)
        self.selected_idx = len(self.scene) - 1
        print(f"[Primitives3D] Placed {self.cur_shape} at ({sx},{sy}). "
              f"Total: {len(self.scene)}")

    def delete_selected(self):
        if self.selected_idx is not None and self.scene:
            removed = self.scene.pop(self.selected_idx)
            self.selected_idx = len(self.scene)-1 if self.scene else None
            print(f"[Primitives3D] Deleted {removed.shape_type}. "
                  f"Remaining: {len(self.scene)}")

    def select_next(self):
        if not self.scene:
            return
        self.selected_idx = 0 if self.selected_idx is None \
                            else (self.selected_idx+1) % len(self.scene)
        print(f"[Primitives3D] Selected → {self.scene[self.selected_idx].shape_type}")

    def clear_scene(self):
        self.scene        = []
        self.selected_idx = None
        self._dragging_idx = None
        print("[Primitives3D] Scene cleared.")

    def reset_rotation(self):
        self.rot_x        = 0.15
        self.rot_y        = 0.0
        self._prev_palm_x = None
        self._prev_palm_y = None
        print("[Primitives3D] Rotation reset.")

    # ── Drag & drop ────────────────────────────────────────

    def start_drag(self, finger_x, finger_y):
        """
        If the index fingertip is near a shape's screen center,
        start dragging that shape.
        """
        best_dist = 80   # px threshold to grab a shape
        best_idx  = None
        for i, obj in enumerate(self.scene):
            d = math.hypot(finger_x - obj.screen_x, finger_y - obj.screen_y)
            if d < best_dist:
                best_dist = d
                best_idx  = i

        if best_idx is not None:
            self._dragging_idx  = best_idx
            self.selected_idx   = best_idx
            self._drag_offset_x = self.scene[best_idx].screen_x - finger_x
            self._drag_offset_y = self.scene[best_idx].screen_y - finger_y
            print(f"[Primitives3D] Drag start: "
                  f"{self.scene[best_idx].shape_type}")

    def update_drag(self, finger_x, finger_y):
        """Move dragged shape to finger position."""
        if self._dragging_idx is not None:
            obj = self.scene[self._dragging_idx]
            obj.screen_x = int(finger_x + self._drag_offset_x)
            obj.screen_y = int(finger_y + self._drag_offset_y)

    def end_drag(self):
        if self._dragging_idx is not None:
            print(f"[Primitives3D] Drag end.")
        self._dragging_idx = None

    # ── Rotation matrices ──────────────────────────────────

    def _Rx(self, a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

    def _Ry(self, a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)

    def _project(self, x, y, z, ox, oy):
        """Project 3D point → 2D, offset by object's screen position."""
        d = self.focal + z
        if abs(d) < 1e-6: d = 1e-6
        sx = int(self.focal * x / d + ox)
        sy = int(self.focal * y / d + oy)
        return sx, sy

    # ── Main update ────────────────────────────────────────

    def update(self, frame, landmarks=None, gesture=None, finger_xy=None):
        """
        Update rotation, handle drag, render all objects.

        Parameters:
            frame      : BGR frame
            landmarks  : hand landmarks or None
            gesture    : current gesture string
            finger_xy  : (x,y) of index fingertip in camera coords

        Returns modified frame.
        """
        h, w = frame.shape[:2]

        # Handle drag
        if gesture == "drag" and finger_xy:
            if self._dragging_idx is None:
                self.start_drag(*finger_xy)
            else:
                self.update_drag(*finger_xy)
        else:
            if self._dragging_idx is not None:
                self.end_drag()

        # Rotation — same system as AR3DEngine:
        #   open_palm gesture + move palm = rotate
        #   no hand                       = auto-spin
        #   any other gesture             = freeze
        if landmarks is None or gesture is None:
            self.rot_x += self.auto_rot_x
            self.rot_y += self.auto_rot_y
            self._prev_palm_x = None
            self._prev_palm_y = None
        elif gesture == "drag":
            # Drag gesture on empty space = rotate all objects
            # (drag near a shape = move that shape instead, handled below)
            if self._dragging_idx is None:
                indices = [0, 5, 9, 13, 17]
                palm_x  = sum(landmarks[i][0] for i in indices) / 5
                palm_y  = sum(landmarks[i][1] for i in indices) / 5
                if self._prev_palm_x is not None:
                    dx = palm_x - self._prev_palm_x
                    dy = palm_y - self._prev_palm_y
                    self.rot_y += dx * self.rot_sensitivity
                    self.rot_x += dy * self.rot_sensitivity
                self._prev_palm_x = palm_x
                self._prev_palm_y = palm_y
            else:
                self._prev_palm_x = None
                self._prev_palm_y = None
        else:
            self._prev_palm_x = None
            self._prev_palm_y = None

        self.rot_x %= (2 * math.pi)
        self.rot_y %= (2 * math.pi)

        R = self._Ry(self.rot_y) @ self._Rx(self.rot_x)

        # Render each scene object
        for i, obj in enumerate(self.scene):
            is_sel = (i == self.selected_idx)
            self._render(frame, obj, R, is_sel, w, h)

        # Draw drag indicator
        if self._dragging_idx is not None and finger_xy:
            cv2.circle(frame, (int(finger_xy[0]), int(finger_xy[1])),
                       12, (0,255,150), 2)

        return frame

    def _render(self, frame, obj, R, is_selected, w, h):
        projected, depths = [], []

        for (x, y, z) in obj.verts:
            pr = R @ np.array([x,y,z], dtype=np.float32)
            sx, sy = self._project(pr[0], pr[1], pr[2],
                                   obj.screen_x, obj.screen_y)
            projected.append((sx, sy))
            depths.append(pr[2])

        if not projected:
            return

        avg_d      = sum(depths) / len(depths)
        depth_norm = max(0.0, min(1.0, (avg_d + 200) / 400))
        fade       = 1.0 - depth_norm * 0.55
        thickness  = 3 if is_selected else 1

        if is_selected:
            col = tuple(min(255, int(c*1.3+40)) for c in obj.color)
        else:
            col = tuple(int(c * fade) for c in obj.color)

        for (i, j) in obj.edges:
            if i >= len(projected) or j >= len(projected):
                continue
            p1, p2 = projected[i], projected[j]
            if (0<=p1[0]<w and 0<=p1[1]<h and
                0<=p2[0]<w and 0<=p2[1]<h):
                cv2.line(frame, p1, p2, col, thickness, cv2.LINE_AA)

        for (sx, sy) in projected:
            if 0<=sx<w and 0<=sy<h:
                cv2.circle(frame, (sx,sy), 2, col, -1)

        # Label + center dot for selected
        if is_selected:
            cv2.circle(frame, (obj.screen_x, obj.screen_y), 6,
                       (255,255,255), -1)
            top = min(projected, key=lambda p: p[1])
            cv2.putText(frame, f"[{obj.shape_type}]",
                        (top[0]-25, max(15, top[1]-12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    # ── Save ───────────────────────────────────────────────

    def save_scene(self, frame):
        save_dir = os.path.join("data", "drawings")
        os.makedirs(save_dir, exist_ok=True)
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(save_dir, f"scene_3d_{ts}.png")
        cv2.imwrite(fname, frame)
        print(f"[Primitives3D] Saved: {fname}")
        return fname

    # ── HUD ────────────────────────────────────────────────

    def draw_hud(self, frame):
        px = frame.shape[1] - 245
        y  = 48

        # Panel background
        cv2.rectangle(frame, (px-8, y-22),
                      (frame.shape[1]-4, y+180),
                      (0,0,0), -1)
        cv2.rectangle(frame, (px-8, y-22),
                      (frame.shape[1]-4, y+180),
                      (0,200,200), 1)

        # Shape name
        cv2.putText(frame, f"Shape: {self.cur_shape.upper()}",
                    (px, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0,255,255), 1, cv2.LINE_AA)
        y += 24

        # Parameters
        for pname, pdata in self.cur_params.items():
            active = (pname == self.active_param)
            col    = (0,255,100) if active else (160,160,160)
            prefix = "► " if active else "  "
            cv2.putText(frame, f"{prefix}{pname}: {pdata[0]}",
                        (px, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, 1, cv2.LINE_AA)
            y += 20

        y += 6
        hints = [
            "n = next shape",
            "p  = next param",
            "=  = bigger  -=smaller",
            "Enter = place shape",
            "drag(empty)=rotate drag(shape)=move",
            "Tab= select  Del=delete",
            "r  = reset rotation",
            "s  = save  c = clear",
        ]
        for h in hints:
            cv2.putText(frame, h, (px, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (120,120,120), 1, cv2.LINE_AA)
            y += 16

        # Scene count
        col = (0,200,255) if self.scene else (80,80,80)
        cv2.putText(frame, f"Objects: {len(self.scene)}",
                    (px, y+4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, col, 1, cv2.LINE_AA)

        return frame
