# =============================================================
# modules/hand_tracker.py
#
# Wraps MediaPipe Hand Landmarker (Tasks API — v0.10.x)
#
# MediaPipe 0.10.x removed mp.solutions.hands and replaced it
# with the Tasks API. This file uses the correct new API.
#
# MEDIAPIPE LANDMARK INDEX MAP:
#
#        8   12  16  20        ← fingertips
#        |    |   |   |
#        7   11  15  19
#        |    |   |   |
#        6   10  14  18
#        |    |   |   |
#   4    5    9  13  17
#   |    |
#   3    |
#   |   (wrist = 0)
#   2
#   |
#   1
#   |
#   0 = WRIST
#
# Finger tip indices:  THUMB=4, INDEX=8, MIDDLE=12, RING=16, PINKY=20
# Finger base indices: THUMB=2, INDEX=5, MIDDLE=9,  RING=13, PINKY=17
# =============================================================

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os
import config
from utils.smoother import MultiPointSmoother
from utils.overlay_utils import draw_landmark_point, draw_connection_line


# Hand skeleton connections
MP_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

FINGERTIP_IDS  = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGERBASE_IDS = {"thumb": 2, "index": 5, "middle":  9, "ring": 13, "pinky": 17}
FINGERPIP_IDS  = {"index": 6, "middle": 10, "ring":  14, "pinky": 18}

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def _ensure_model():
    """Download the hand landmarker .task model file if not present."""
    if not os.path.exists(MODEL_PATH):
        print(f"[HandTracker] Downloading model to '{MODEL_PATH}' ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[HandTracker] Download complete.")
    else:
        print(f"[HandTracker] Model found: {MODEL_PATH}")


class HandTracker:
    def __init__(self):
        _ensure_model()

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=config.MAX_HANDS,
            min_hand_detection_confidence=config.DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )

        self.detector       = mp_vision.HandLandmarker.create_from_options(options)
        self.smoother       = MultiPointSmoother(num_points=21)
        self.show_landmarks = config.SHOW_LANDMARKS
        self.landmarks      = None
        self.raw_landmarks  = None

    def update(self, frame):
        """
        Process one frame. Returns list of 21 (x, y) pixel-coord tuples,
        or None if no hand detected.
        """
        h, w = frame.shape[:2]

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            self.smoother.reset()
            self.landmarks     = None
            self.raw_landmarks = None
            return None

        # result.hand_landmarks[0] = list of 21 NormalizedLandmark
        hand = result.hand_landmarks[0]
        raw  = [(lm.x * w, lm.y * h) for lm in hand]

        self.raw_landmarks = raw
        self.landmarks     = self.smoother.update(raw)
        return self.landmarks

    def draw(self, frame):
        """
        Draw skeleton and landmark dots onto frame.
        Only draws if show_landmarks is True.
        """
        if not self.show_landmarks or self.landmarks is None:
            return frame

        lm = self.landmarks

        for (a, b) in MP_HAND_CONNECTIONS:
            draw_connection_line(frame, lm[a], lm[b], color=(80, 80, 80), thickness=1)

        for i, (x, y) in enumerate(lm):
            if i in FINGERTIP_IDS.values():
                draw_landmark_point(frame, x, y, color=(0, 255, 255), radius=6)
            else:
                draw_landmark_point(frame, x, y, color=(180, 180, 180), radius=4)

        return frame

    def get_landmark(self, name):
        """Get a specific landmark by finger name or integer index."""
        if self.landmarks is None:
            return None
        if isinstance(name, int):
            return self.landmarks[name]
        if name in FINGERTIP_IDS:
            return self.landmarks[FINGERTIP_IDS[name]]
        return None

    def toggle_landmarks(self):
        self.show_landmarks = not self.show_landmarks
        print(f"[HandTracker] Landmarks: {'ON' if self.show_landmarks else 'OFF'}")


# -------------------------------------------------------------
# STANDALONE TEST:  python modules/hand_tracker.py
# -------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")

    cap     = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    tracker = HandTracker()

    print("Hand tracker test. Show your hand. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame     = cv2.flip(frame, 1)
        landmarks = tracker.update(frame)
        tracker.draw(frame)

        if landmarks:
            ix, iy = landmarks[8]
            cv2.putText(frame, f"Index: ({int(ix)}, {int(iy)})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Hand Tracker Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()