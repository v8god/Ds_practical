# =============================================================
# modules/gesture_engine.py  —  Phase 3 Update
#
# CHANGES FROM ORIGINAL:
#   1. Open palm swipe: tracks palm x-movement over time
#      → swipe right = "filter_next"
#      → swipe left  = "filter_prev"
#
#   2. Select gesture: records start (x,y) on thumb+middle pinch,
#      records end (x,y) on second pinch
#      → gesture_engine now exposes select_start_pos, select_end_pos
#      → mouse_controller uses these for actual drag selection
#
#   3. Shape gestures (index finger only, but with wrist angle):
#      When draw mode is active, a second hand (or held pose) can
#      snap to shapes. We add shape_mode cycling via 'ok' gesture
#      (thumb + pinky tip meet).
#      Shapes: freehand → line → rectangle → circle → triangle
# GESTURE REFERENCE TABLE:
# ┌────────────────────────┬──────────────────────────────────────┐
# │ Gesture name           │ How it's detected                    │
# ├────────────────────────┼──────────────────────────────────────┤
# │ "none"                 │ No hand detected                     │
# │ "idle"                 │ Hand present but no specific gesture │
# │ "click"                │ Thumb tip + Index tip distance < threshold │
# │ "select_start"         │ Thumb tip + Middle tip meet (first)  │
# │ "select_end"           │ Thumb tip + Middle tip meet (again)  │
# │ "drag"                 │ Index + Middle fingers up, others down│
# │ "erase"                │ Closed fist (all fingers down)       │
# │ "draw"                 │ Only index finger extended           │
# │ "filter_next"          │ Open palm (all 5 fingers up)         │
# └────────────────────────┴──────────────────────────────────────┘
#
# =============================================================

import math
import time
import collections
import config
from modules.hand_tracker import FINGERTIP_IDS, FINGERBASE_IDS, FINGERPIP_IDS


def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


# Shape modes cycle order
SHAPE_MODES = ["freehand", "line", "rectangle", "circle", "triangle"]


class GestureEngine:
    def __init__(self):
        self.threshold = config.PINCH_THRESHOLD

        # --- Select state ---
        self._select_active    = False
        self._select_cooldown  = 0.6
        self._last_select_time = 0.0
        self.select_start_pos  = None   # (x,y) in SCREEN coords — set by mouse_controller
        self.select_end_pos    = None   # (x,y) in SCREEN coords

        # --- Click debounce ---
        self._click_cooldown  = 0.3
        self._last_click_time = 0.0

        # --- Swipe detection for open palm ---
        # Store the last N palm x-positions to detect directional movement
        self._palm_x_history  = collections.deque(maxlen=12)  # ~12 frames at 30fps = 0.4s window
        self._swipe_cooldown  = 0.8     # seconds between swipe triggers
        self._last_swipe_time = 0.0
        self._swipe_threshold = 80      # pixels of x-movement to count as a swipe

        # --- Shape mode ---
        self.shape_mode       = "freehand"   # current drawing shape
        self._shape_cooldown  = 0.8
        self._last_shape_time = 0.0

        # --- State ---
        self._prev_gesture = "none"
        self.gesture       = "none"

        # Expose swipe direction separately so main.py can act on it
        self.swipe_direction = None   # "left", "right", or None

    # ----------------------------------------------------------
    def update(self, landmarks):
        self.swipe_direction = None   # reset each frame

        if landmarks is None:
            self.gesture = "none"
            self._palm_x_history.clear()
            return self.gesture

        thumb_tip  = landmarks[FINGERTIP_IDS["thumb"]]
        index_tip  = landmarks[FINGERTIP_IDS["index"]]
        middle_tip = landmarks[FINGERTIP_IDS["middle"]]
        ring_tip   = landmarks[FINGERTIP_IDS["ring"]]
        pinky_tip  = landmarks[FINGERTIP_IDS["pinky"]]
        wrist      = landmarks[0]

        fingers_up = self._fingers_up(landmarks)

        thumb_index_dist  = distance(thumb_tip, index_tip)
        thumb_middle_dist = distance(thumb_tip, middle_tip)
        thumb_pinky_dist  = distance(thumb_tip, pinky_tip)

        now = time.time()

        # --------------------------------------------------
        # 1. ERASE — closed fist
        if sum(fingers_up) == 0:
            self.gesture = "erase"

        # 2. CLICK — thumb + index pinch
        elif thumb_index_dist < self.threshold:
            if now - self._last_click_time > self._click_cooldown:
                self._last_click_time = now
                self.gesture = "click"
            else:
                self.gesture = "idle"

        # 3. SELECT — thumb + middle pinch (stateful)
        elif thumb_middle_dist < self.threshold:
            if now - self._last_select_time > self._select_cooldown:
                self._last_select_time = now
                if not self._select_active:
                    self._select_active = True
                    self.gesture = "select_start"
                else:
                    self._select_active = False
                    self.gesture = "select_end"
            else:
                self.gesture = "idle"

        # 4. SHAPE MODE CYCLE — thumb + pinky pinch
        #    Cycles through freehand → line → rectangle → circle → triangle
        elif thumb_pinky_dist < self.threshold:
            if now - self._last_shape_time > self._shape_cooldown:
                self._last_shape_time = now
                idx = SHAPE_MODES.index(self.shape_mode)
                self.shape_mode = SHAPE_MODES[(idx + 1) % len(SHAPE_MODES)]
                print(f"[GestureEngine] Shape mode: {self.shape_mode.upper()}")
            self.gesture = "idle"

        # 5. OPEN PALM — all 5 fingers up → detect swipe direction
        elif sum(fingers_up) == 5:
            # Track palm center x (average of wrist and middle base)
            palm_x = (wrist[0] + landmarks[FINGERBASE_IDS["middle"]][0]) / 2
            self._palm_x_history.append(palm_x)
            self.gesture = "open_palm"

            # Need at least half the history window to judge a swipe
            if len(self._palm_x_history) >= 6:
                if now - self._last_swipe_time > self._swipe_cooldown:
                    x_start = self._palm_x_history[0]
                    x_end   = self._palm_x_history[-1]
                    delta   = x_end - x_start

                    if delta > self._swipe_threshold:
                        # Moved right → next filter
                        self.swipe_direction  = "right"
                        self._last_swipe_time = now
                        self._palm_x_history.clear()
                        self.gesture = "filter_next"

                    elif delta < -self._swipe_threshold:
                        # Moved left → previous filter
                        self.swipe_direction  = "left"
                        self._last_swipe_time = now
                        self._palm_x_history.clear()
                        self.gesture = "filter_prev"

        # 6. DRAW — only index finger extended
        elif fingers_up == [False, True, False, False, False]:
            self._palm_x_history.clear()
            self.gesture = "draw"

        # 7. DRAG — index + middle up
        elif fingers_up == [False, True, True, False, False]:
            self._palm_x_history.clear()
            self.gesture = "drag"

        # 8. IDLE
        else:
            self._palm_x_history.clear()
            self.gesture = "idle"

        self._prev_gesture = self.gesture
        return self.gesture

    # ----------------------------------------------------------
    def _fingers_up(self, landmarks):
        up = []
        thumb_tip  = landmarks[FINGERTIP_IDS["thumb"]]
        thumb_base = landmarks[FINGERBASE_IDS["thumb"]]
        up.append(thumb_tip[0] > thumb_base[0])

        for name in ["index", "middle", "ring", "pinky"]:
            tip = landmarks[FINGERTIP_IDS[name]]
            pip = landmarks[FINGERPIP_IDS[name]]
            up.append(tip[1] < pip[1])

        return up

    # ----------------------------------------------------------
    def draw_hud(self, frame):
        from utils.overlay_utils import draw_text_with_bg

        color_map = {
            "click":        (0, 255, 0),
            "select_start": (0, 165, 255),
            "select_end":   (0, 100, 255),
            "drag":         (255, 255, 0),
            "erase":        (0, 0, 255),
            "draw":         (255, 0, 255),
            "filter_next":  (0, 255, 255),
            "filter_prev":  (255, 200, 0),
            "open_palm":    (200, 200, 200),
            "idle":         (180, 180, 180),
            "none":         (80, 80, 80),
        }
        color = color_map.get(self.gesture, (255, 255, 255))

        label = f"Gesture: {self.gesture.upper()}"
        if self.gesture == "draw":
            label += f"  [{self.shape_mode}]"

        draw_text_with_bg(frame, label, (10, 65),
                          font_scale=0.6,
                          text_color=color,
                          bg_color=(0, 0, 0))

        # Show shape mode always when in draw context
        if self.shape_mode != "freehand":
            draw_text_with_bg(frame, f"Shape: {self.shape_mode.upper()}",
                              (10, 125),
                              font_scale=0.55,
                              text_color=(255, 180, 0),
                              bg_color=(0, 0, 0))
        return frame