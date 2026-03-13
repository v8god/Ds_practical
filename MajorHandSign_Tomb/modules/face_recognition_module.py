# =============================================================
# modules/face_recognition_module.py
#
# Face detection + recognition using OpenCV DNN detector
# and the face_recognition library for encoding + matching.
#
# WHY NOT haar cascade?
#   DNN detector (res10_300x300_ssd) is far more accurate,
#   handles tilted faces, partial occlusion, and works at
#   multiple scales without tuning. Haar fails on any tilt.
#
# ARCHITECTURE:
#   - OpenCV DNN detects face bounding boxes every N frames
#   - face_recognition encodes detected faces (128-d vector)
#   - Encoding compared to stored known encodings via L2 distance
#   - Result: name string or "unknown"
#
# HOW TO ADD YOUR FACE:
#   1. Put a clear photo (jpg/png) in assets/known_faces/
#   2. Filename = person's name  e.g.  pratham.jpg
#   3. Restart — encodings are auto-generated on startup
#
# PERFORMANCE:
#   face_recognition is heavy. We run it every N frames only
#   (config.FACE_RECOGNITION_EVERY_N_FRAMES, default=5).
#   Between runs, the last result is displayed — imperceptible.
#
# ANIMATION TRIGGER:
#   When a known face is recognized, self.recognized_name is set.
#   ar_overlay reads this and plays the aura animation.
#   Unknown face → self.recognized_name = "unknown"
#   No face      → self.recognized_name = None
# =============================================================

import cv2
import numpy as np
import os
import urllib.request
import config

# face_recognition is optional install — handle gracefully
try:
    import face_recognition as fr
    FR_AVAILABLE = True
except ImportError:
    FR_AVAILABLE = False
    print("[FaceRecog] WARNING: face_recognition not installed.")
    print("  Run:  pip install face-recognition")
    print("  Face recognition will be disabled until installed.")


# ── DNN Model paths ──────────────────────────────────────────
# OpenCV ships with a res10 SSD face detector.
# We download it if not present (small files, ~10MB total).
PROTOTXT_URL = ("https://raw.githubusercontent.com/opencv/opencv/master/"
                "samples/dnn/face_detector/deploy.prototxt")
MODEL_URL    = ("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples"
                "_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")
PROTOTXT_PATH = "face_detector.prototxt"
MODEL_PATH    = "face_detector.caffemodel"


def _ensure_dnn_model():
    """Download DNN face detector files if not present."""
    for path, url in [(PROTOTXT_PATH, PROTOTXT_URL),
                      (MODEL_PATH,    MODEL_URL)]:
        if not os.path.exists(path):
            print(f"[FaceRecog] Downloading {path} ...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"[FaceRecog] Downloaded {path}")
            except Exception as e:
                print(f"[FaceRecog] Download failed: {e}")
                return False
    return True


class FaceRecognizer:
    def __init__(self):
        self.enabled          = FR_AVAILABLE
        self.net              = None          # OpenCV DNN net
        self.known_encodings  = []            # list of 128-d arrays
        self.known_names      = []            # list of name strings

        # State
        self.recognized_name  = None         # last result: str or None
        self.face_boxes       = []            # [(x1,y1,x2,y2), ...] last detected
        self._frame_count     = 0
        self._run_every       = config.FACE_RECOGNITION_EVERY_N_FRAMES
        self._tolerance       = config.FACE_TOLERANCE

        # Animation trigger
        self.trigger_animation = False        # set True for one cycle when known face found
        self._prev_name        = None

        if not self.enabled:
            return

        # Load DNN face detector
        if _ensure_dnn_model():
            try:
                self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
                print("[FaceRecog] DNN face detector loaded.")
            except Exception as e:
                print(f"[FaceRecog] Failed to load DNN model: {e}")
                self.net = None

        # Load known face encodings from assets/known_faces/
        self._load_known_faces()

    # ----------------------------------------------------------
    def _load_known_faces(self):
        """
        Scan assets/known_faces/ for images.
        Filename (without extension) = person's name.
        Generate face_recognition encoding for each.
        """
        if not FR_AVAILABLE:
            return

        faces_dir = config.KNOWN_FACES_DIR
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir, exist_ok=True)
            print(f"[FaceRecog] Created {faces_dir}/ — add face images here.")
            return

        loaded = 0
        for fname in os.listdir(faces_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            name  = os.path.splitext(fname)[0]
            fpath = os.path.join(faces_dir, fname)

            img = fr.load_image_file(fpath)
            encs = fr.face_encodings(img)

            if not encs:
                print(f"[FaceRecog] No face found in {fname} — skipping.")
                continue

            self.known_encodings.append(encs[0])
            self.known_names.append(name)
            loaded += 1
            print(f"[FaceRecog] Loaded: {name}")

        print(f"[FaceRecog] {loaded} known face(s) loaded.")

    # ----------------------------------------------------------
    def update(self, frame):
        """
        Run face detection + recognition on this frame.
        Only runs full recognition every N frames for performance.

        Returns nothing — results stored in self.recognized_name
        and self.face_boxes.
        """
        if not self.enabled or self.net is None:
            return

        self._frame_count += 1
        self.trigger_animation = False

        # Only run every N frames
        if self._frame_count % self._run_every != 0:
            return

        h, w = frame.shape[:2]

        # ── Step 1: DNN face detection ───────────────────────
        # Prepare blob: resize to 300x300, subtract mean BGR values
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0)   # ImageNet BGR mean
        )
        self.net.setInput(blob)
        detections = self.net.forward()   # shape: (1,1,200,7)

        self.face_boxes = []
        face_images     = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.6:          # Confidence threshold
                continue

            # Box coords are normalized (0-1) — scale to frame size
            x1 = int(detections[0,0,i,3] * w)
            y1 = int(detections[0,0,i,4] * h)
            x2 = int(detections[0,0,i,5] * w)
            y2 = int(detections[0,0,i,6] * h)

            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            self.face_boxes.append((x1, y1, x2, y2))

            # Crop face region for encoding
            face_crop = frame[y1:y2, x1:x2]
            face_images.append(face_crop)

        # ── Step 2: face_recognition encoding + matching ─────
        if not face_images or not self.known_encodings:
            if not self.face_boxes:
                self.recognized_name = None
            else:
                self.recognized_name = "unknown"
            return

        # Convert to RGB (face_recognition needs RGB, cv2 gives BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get face locations in face_recognition format: (top,right,bottom,left)
        locations = [(y1, x2, y2, x1) for (x1,y1,x2,y2) in self.face_boxes]

        # Encode detected faces
        encodings = fr.face_encodings(rgb_frame, locations)

        names = []
        for enc in encodings:
            # Compare to all known encodings
            distances = fr.face_distance(self.known_encodings, enc)
            best_idx  = int(np.argmin(distances))

            if distances[best_idx] <= self._tolerance:
                names.append(self.known_names[best_idx])
            else:
                names.append("unknown")

        # Use first detected face result
        self.recognized_name = names[0] if names else "unknown"

        # Trigger animation when a known person is first recognized
        if (self.recognized_name != "unknown" and
                self.recognized_name != self._prev_name):
            self.trigger_animation = True
            print(f"[FaceRecog] Recognized: {self.recognized_name}")

        self._prev_name = self.recognized_name

    # ----------------------------------------------------------
    def draw(self, frame):
        """
        Draw face boxes and name labels on frame.
        Green box + name = known person.
        Red box + "unknown" = unrecognized face.
        """
        if not self.face_boxes:
            return frame

        for i, (x1, y1, x2, y2) in enumerate(self.face_boxes):
            name  = self.recognized_name if i == 0 else "unknown"
            known = (name != "unknown" and name is not None)

            color = (0, 255, 80) if known else (0, 60, 255)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Corner accents (more stylish than plain rectangle)
            cl = 16   # corner length
            cv2.line(frame, (x1, y1), (x1+cl, y1), color, 3)
            cv2.line(frame, (x1, y1), (x1, y1+cl), color, 3)
            cv2.line(frame, (x2, y1), (x2-cl, y1), color, 3)
            cv2.line(frame, (x2, y1), (x2, y1+cl), color, 3)
            cv2.line(frame, (x1, y2), (x1+cl, y2), color, 3)
            cv2.line(frame, (x1, y2), (x1, y2-cl), color, 3)
            cv2.line(frame, (x2, y2), (x2-cl, y2), color, 3)
            cv2.line(frame, (x2, y2), (x2, y2-cl), color, 3)

            # Name label
            label = name.upper() if name else "UNKNOWN"
            lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1-lh-12), (x1+lw+8, y1), color, -1)
            cv2.putText(frame, label, (x1+4, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,0,0) if known else (255,255,255),
                        2, cv2.LINE_AA)

            # Glow ring for known face
            if known:
                cx = (x1+x2)//2
                cy = (y1+y2)//2
                r  = max((x2-x1), (y2-y1))//2 + 15
                cv2.circle(frame, (cx, cy), r, color, 1, cv2.LINE_AA)

        return frame