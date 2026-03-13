# =============================================================
# modules/sign_detector.py
#
# Detects held hand signs from MediaPipe landmarks.
#
# HOW TO ADD A NEW SIGN:
#   Add one entry to SIGN_REGISTRY below.
#   Keys:
#     "fingers_up"   : list of 5 bools [thumb, index, middle, ring, pinky]
#                      True=up, False=down, None=don't care
#     "hold_frames"  : how many consecutive frames the sign must be
#                      held before it triggers (default 20 = ~0.67s)
#     "pinch"        : optional — (finger_a, finger_b, threshold_px)
#                      e.g. ("thumb","index", 35) for OK sign
#     "cross"        : optional — True means index crosses over middle
#                      (used for scissors / Domain Expansion)
#     "claw"         : optional — True means fingers half-curled
#
# SIGN_REGISTRY keys must match keys in sign_animator.py registry.
# =============================================================

import math
from modules.hand_tracker import FINGERTIP_IDS, FINGERBASE_IDS, FINGERPIP_IDS


# ----------------------------------------------------------
# SIGN REGISTRY
# Add your own signs here — nothing else needs to change.
# ----------------------------------------------------------
SIGN_REGISTRY = {

    "domain_expansion": {
        # Index + middle crossed (scissors held)
        "fingers_up":  [None, True, True, False, False],
        "cross":       True,    # index tip must be right of middle tip
        "hold_frames": 25,
    },

    "black_flash": {
        # Pinky only up
        "fingers_up":  [False, False, False, False, True],
        "hold_frames": 20,
    },

    "divergent_fist": {
        # Index + middle + ring up
        "fingers_up":  [False, True, True, True, False],
        "hold_frames": 20,
    },

    "red_blue": {
        # L-shape: thumb + index up (gun shape)
        "fingers_up":  [True, True, False, False, False],
        "hold_frames": 22,
    },

    "binding_vow": {
        # Claw: all fingers half-curled
        "fingers_up":  [False, False, False, False, False],
        "claw":        True,    # distinguishes from fist (erase gesture)
        "hold_frames": 25,
    },

    "reverse_cursed": {
        # OK sign: thumb + index pinch, others up
        "fingers_up":  [None, None, True, True, True],
        "pinch":       ("thumb", "index", 38),
        "hold_frames": 20,
    },

}


def _dist(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])


class SignDetector:
    def __init__(self):
        # Per-sign hold counters
        self._hold_counts = {name: 0 for name in SIGN_REGISTRY}
        self.active_sign  = None   # currently triggered sign name, or None
        self._cooldown    = 0      # frames to wait before next trigger
        self._COOLDOWN    = 180    # 6 seconds between triggers

    # ----------------------------------------------------------
    def update(self, landmarks):
        """
        Check landmarks against every registered sign.
        Sets self.active_sign when a sign is held long enough.
        Returns sign name string or None.
        """
        self.active_sign = None

        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        if landmarks is None:
            self._reset_all()
            return None

        for sign_name, rules in SIGN_REGISTRY.items():
            if self._check_sign(landmarks, rules):
                self._hold_counts[sign_name] += 1
                # Reset all others
                for other in self._hold_counts:
                    if other != sign_name:
                        self._hold_counts[other] = 0

                if self._hold_counts[sign_name] >= rules.get("hold_frames", 20):
                    self._hold_counts[sign_name] = 0
                    self._cooldown   = self._COOLDOWN
                    self.active_sign = sign_name
                    print(f"[SignDetector] Triggered: {sign_name}")
                    return sign_name
            else:
                self._hold_counts[sign_name] = max(0,
                    self._hold_counts[sign_name] - 1)

        return None

    # ----------------------------------------------------------
    def _check_sign(self, lm, rules):
        fingers_up = self._fingers_up(lm)

        # Check finger pattern
        pattern = rules.get("fingers_up", [None]*5)
        for i, expected in enumerate(pattern):
            if expected is not None and fingers_up[i] != expected:
                return False

        # Check pinch
        if "pinch" in rules:
            fa, fb, thresh = rules["pinch"]
            d = _dist(lm[FINGERTIP_IDS[fa]], lm[FINGERTIP_IDS[fb]])
            if d > thresh:
                return False

        # Check cross (index tip x > middle tip x = crossed)
        if rules.get("cross"):
            ix = lm[FINGERTIP_IDS["index"]][0]
            mx = lm[FINGERTIP_IDS["middle"]][0]
            if ix <= mx:   # not crossed
                return False

        # Check claw: fingers must be partially bent
        # tip.y > pip.y (not fully down) but tip.y > base.y (not fully up)
        # i.e. tip is between base and pip → curled
        if rules.get("claw"):
            claw = True
            for name in ["index","middle","ring","pinky"]:
                tip  = lm[FINGERTIP_IDS[name]][1]
                pip  = lm[FINGERPIP_IDS[name]][1]
                base = lm[FINGERBASE_IDS[name]][1]
                # In claw: tip should be between base and pip (y-wise)
                if not (base > tip > pip):
                    claw = False
                    break
            if not claw:
                return False

        return True

    # ----------------------------------------------------------
    def _fingers_up(self, lm):
        up = []
        # Thumb: x-axis
        up.append(lm[FINGERTIP_IDS["thumb"]][0] > lm[FINGERBASE_IDS["thumb"]][0])
        # Other fingers: y-axis (tip above pip = up)
        for name in ["index","middle","ring","pinky"]:
            up.append(lm[FINGERTIP_IDS[name]][1] < lm[FINGERPIP_IDS[name]][1])
        return up

    def _reset_all(self):
        for k in self._hold_counts:
            self._hold_counts[k] = 0

    # ----------------------------------------------------------
    def draw_hud(self, frame):
        """Show hold progress bar for the sign closest to triggering."""
        if not self._hold_counts:
            return frame

        best_sign  = max(self._hold_counts, key=self._hold_counts.get)
        best_count = self._hold_counts[best_sign]
        required   = SIGN_REGISTRY[best_sign].get("hold_frames", 20)

        if best_count < 3:   # don't show if barely started
            return frame

        # Progress bar
        bar_w    = 200
        filled   = int(bar_w * best_count / required)
        x, y     = 10, frame.shape[0] - 60

        cv2.rectangle(frame, (x, y), (x+bar_w, y+12), (40,40,40), -1)
        cv2.rectangle(frame, (x, y), (x+filled, y+12), (0,200,255), -1)
        cv2.rectangle(frame, (x, y), (x+bar_w, y+12), (0,200,255), 1)
        cv2.putText(frame, best_sign.replace("_"," ").upper(),
                    (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0,200,255), 1, cv2.LINE_AA)

        if self._cooldown > 0:
            cv2.putText(frame, "COOLDOWN",
                        (x+bar_w+8, y+10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (100,100,100), 1, cv2.LINE_AA)

        return frame


import cv2   # needed for draw_hud