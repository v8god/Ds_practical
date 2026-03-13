# =============================================================
# modules/camera.py
#
# Handles everything related to the webcam:
#   - Opening the camera
#   - Reading frames
#   - Flipping (mirror mode — feels natural for AR)
#   - Resizing to the target resolution
#   - Releasing cleanly on exit
#
# WHY A CLASS instead of just cap = cv2.VideoCapture(0) in main?
# Encapsulating the camera means main.py doesn't care HOW the
# frame is captured. Later if you switch to an IP camera or a
# video file for testing, you only change this file.
# =============================================================

import cv2
import config


class Camera:
    def __init__(self):
        self.index  = config.CAMERA_INDEX
        self.width  = config.FRAME_WIDTH
        self.height = config.FRAME_HEIGHT
        self.cap    = None          # cv2.VideoCapture object, set in open()
        self.is_open = False

    def open(self):
        """
        Opens the camera and sets resolution.
        Call this once before your main loop.
        Raises RuntimeError if camera cannot be opened.
        """
        # CAP_DSHOW is Windows-specific — it uses DirectShow backend.
        # This is faster to open on Windows than the default backend
        # and avoids the common "could not open camera" error on Windows 10/11.
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self.index}. "
                "Check that your webcam is connected and not used by another app."
            )

        # Request the resolution we want
        # (camera may not support it — it'll pick the closest it can)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

        # Read back what the camera actually gave us
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Opened index={self.index} | "
              f"Requested: {self.width}x{self.height} | "
              f"Actual: {actual_w}x{actual_h}")

        self.is_open = True

    def read(self):
        """
        Reads one frame from the camera.

        Returns:
            frame : np.ndarray (BGR) — the frame, ready to process
                    Returns None if reading failed.

        What flip(1) does:
            Mirrors the image horizontally so it behaves like a mirror.
            Without this, raising your RIGHT hand appears on the LEFT
            side of the screen, which feels wrong for AR interaction.
        """
        if not self.is_open:
            return None

        ret, frame = self.cap.read()

        if not ret or frame is None:
            print("[Camera] WARNING: Failed to read frame.")
            return None

        # Mirror flip (1 = horizontal axis)
        frame = cv2.flip(frame, 1)

        # Resize to exact target size in case camera gave us different dims
        # (some webcams only support certain resolutions)
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        return frame

    def release(self):
        """
        Releases the camera and frees the hardware resource.
        Always call this on exit — otherwise the camera LED stays on
        and the next run may fail to open it.
        """
        if self.cap is not None:
            self.cap.release()
            self.is_open = False
            print("[Camera] Released.")


# -------------------------------------------------------------
# STANDALONE TEST — run directly to verify camera works:
#   python modules/camera.py
# Press 'q' to quit.
# -------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")   # So config.py is findable when run from modules/

    cam = Camera()
    cam.open()

    print("Camera test running. Press 'q' to quit.")

    while True:
        frame = cam.read()
        if frame is None:
            print("No frame received. Exiting.")
            break

        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()