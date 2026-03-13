# =============================================================
# modules/face_animator.py
#
# Plays visual animations when a known face is recognized.
#
# ANIMATIONS:
#   Known face   → glowing aura rings that pulse outward from face
#   Unknown face → red static overlay + "UNKNOWN" alert effect
#
# WHY A SEPARATE MODULE:
#   Keeps animation rendering logic out of face_recognition_module.py
#   (which only does detection/encoding). Single responsibility.
#
# ANIMATION STATES:
#   "idle"    → nothing playing
#   "known"   → aura animation (plays for AURA_DURATION frames)
#   "unknown" → warning flash (plays for WARN_DURATION frames)
# =============================================================

import cv2
import numpy as np
import math


AURA_DURATION = 90    # frames (~3 seconds at 30fps)
WARN_DURATION = 45    # frames (~1.5 seconds)


class FaceAnimator:
    def __init__(self):
        self.state          = "idle"
        self._frame_counter = 0
        self._name          = None
        self._face_box      = None    # (x1,y1,x2,y2) of the face to animate around

    def trigger_known(self, name, face_box):
        """Start the known-face aura animation."""
        self.state          = "known"
        self._frame_counter = AURA_DURATION
        self._name          = name
        self._face_box      = face_box
        print(f"[FaceAnimator] Aura animation: {name}")

    def trigger_unknown(self, face_box):
        """Start the unknown-face warning animation."""
        self.state          = "unknown"
        self._frame_counter = WARN_DURATION
        self._name          = "UNKNOWN"
        self._face_box      = face_box

    def update(self, frame):
        """
        Draw current animation frame onto frame.
        Call every frame — handles its own countdown.
        """
        if self.state == "idle" or self._frame_counter <= 0:
            self.state = "idle"
            return frame

        self._frame_counter -= 1

        if self.state == "known":
            frame = self._draw_aura(frame)
        elif self.state == "unknown":
            frame = self._draw_warning(frame)

        return frame

    # ----------------------------------------------------------
    def _draw_aura(self, frame):
        """
        Pulsing cyan aura rings that expand outward from the face.
        3 rings at different phases create a continuous pulse effect.
        """
        if self._face_box is None:
            return frame

        x1, y1, x2, y2 = self._face_box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        base_r = max(x2-x1, y2-y1) // 2 + 10

        # Progress 0→1 over the animation duration
        progress = 1.0 - self._frame_counter / AURA_DURATION

        # Draw 3 rings at offset phases
        for ring in range(3):
            phase     = (progress + ring / 3.0) % 1.0
            r         = int(base_r + phase * 80)
            alpha     = 1.0 - phase             # fade as ring expands
            thickness = max(1, int(3 * (1.0 - phase)))

            color = (
                int(0   * alpha),
                int(255 * alpha),
                int(255 * alpha),
            )

            if alpha > 0.05:
                cv2.circle(frame, (cx, cy), r, color, thickness, cv2.LINE_AA)

        # Pulsing name label above face
        pulse   = 0.7 + 0.3 * math.sin(progress * math.pi * 6)
        label   = f"✓ {self._name.upper()}"
        font_sc = 0.65 * pulse
        lw, lh  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_sc, 2)[0]
        tx      = cx - lw // 2
        ty      = y1 - 25
        col     = (int(0*pulse), int(255*pulse), int(200*pulse))
        cv2.putText(frame, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_sc, col, 2, cv2.LINE_AA)

        # Subtle bright flash on first few frames
        if self._frame_counter > AURA_DURATION - 8:
            intensity = (self._frame_counter - (AURA_DURATION - 8)) * 6
            flash     = np.zeros_like(frame, dtype=np.uint8)
            flash[:,:,1] = intensity   # green channel flash
            flash[:,:,0] = intensity // 2
            frame = cv2.add(frame, flash)

        return frame

    # ----------------------------------------------------------
    def _draw_warning(self, frame):
        """
        Red pulsing border + UNKNOWN alert for unrecognized faces.
        """
        progress = 1.0 - self._frame_counter / WARN_DURATION
        pulse    = abs(math.sin(progress * math.pi * 4))

        # Red border around entire frame
        border   = int(6 * pulse)
        h, w     = frame.shape[:2]
        red      = (0, 0, int(200 * pulse))
        cv2.rectangle(frame, (0, 0), (w-1, h-1), red, border)

        if self._face_box:
            x1, y1, x2, y2 = self._face_box
            # Red overlay on face region
            overlay         = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 150), -1)
            cv2.addWeighted(overlay, 0.25 * pulse, frame, 1 - 0.25*pulse, 0, frame)

        return frame

    @property
    def is_playing(self):
        return self.state != "idle"