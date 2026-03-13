# =============================================================
# utils/fps_counter.py
#
# Calculates real FPS using the time between frames.
# Why not use cv2.CAP_PROP_FPS? That returns the *requested* FPS,
# not the actual FPS your system is achieving. This measures real.
# =============================================================

import time
import cv2


class FPSCounter:
    def __init__(self):
        self._prev_time = time.time()   # Time of last frame
        self._fps = 0.0                 # Last calculated FPS value

    def update(self):
        """
        Call this once per frame in your main loop.
        Returns the current FPS as a float.
        """
        current_time = time.time()
        elapsed = current_time - self._prev_time

        # Avoid division by zero if two frames arrive at the same instant
        if elapsed > 0:
            self._fps = 1.0 / elapsed

        self._prev_time = current_time
        return self._fps

    def draw(self, frame):
        """
        Draws the FPS value onto the top-left corner of the frame.
        Modifies frame in-place (no return needed, but returns it anyway
        so you can chain: frame = counter.draw(frame)).
        """
        fps_text = f"FPS: {int(self._fps)}"

        # Dark background rectangle so text is readable on any video
        cv2.rectangle(frame, (8, 8), (110, 36), (0, 0, 0), -1)

        cv2.putText(
            frame,
            fps_text,
            (12, 28),               # Bottom-left corner of text
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,                    # Font scale
            (0, 255, 0),            # Green text (BGR)
            2,                      # Thickness
            cv2.LINE_AA             # Anti-aliased (smoother edges)
        )
        return frame

    @property
    def fps(self):
        return self._fps


# -------------------------------------------------------------
# STANDALONE TEST — run this file directly to verify it works:
#   python utils/fps_counter.py
# -------------------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    counter = FPSCounter()

    print("FPS counter test running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read from camera.")
            break

        counter.update()
        counter.draw(frame)

        cv2.imshow("FPS Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()