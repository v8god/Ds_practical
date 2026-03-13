# =============================================================
# utils/smoother.py
#
# Smooths noisy coordinate values from hand tracking.
#
# WHY THIS IS NEEDED:
# MediaPipe returns raw pixel coordinates every frame.
# Even when your hand is perfectly still, the values jitter
# by a few pixels due to camera noise + model uncertainty.
# Without smoothing, the mouse cursor shakes and drawn lines
# look jagged. With smoothing, movement feels intentional.
#
# METHOD: Exponential Moving Average (EMA)
# new_smooth = alpha * raw + (1 - alpha) * previous_smooth
#
# alpha in config.py (SMOOTHING_ALPHA):
#   - 0.3 = very smooth but sluggish (slow to respond)
#   - 0.5 = good balance (recommended start)
#   - 0.8 = very responsive but still a little jittery
#   - 1.0 = no smoothing at all (raw values)
# =============================================================

import config


class EMAsmoother:
    """
    Exponential Moving Average smoother for (x, y) coordinates.
    One instance per point you want to smooth (e.g. index fingertip).
    """

    def __init__(self, alpha=None):
        """
        alpha: smoothing factor. If None, reads from config.SMOOTHING_ALPHA.
        """
        self.alpha = alpha if alpha is not None else config.SMOOTHING_ALPHA
        self._sx = None   # Smoothed x (None until first value arrives)
        self._sy = None   # Smoothed y

    def update(self, x, y):
        """
        Feed in a new raw (x, y) point.
        Returns the smoothed (x, y) as a tuple of floats.

        First call: returns the raw value as-is (nothing to smooth yet).
        """
        if self._sx is None:
            # First data point — initialize with raw value
            self._sx = float(x)
            self._sy = float(y)
        else:
            self._sx = self.alpha * x + (1.0 - self.alpha) * self._sx
            self._sy = self.alpha * y + (1.0 - self.alpha) * self._sy

        return self._sx, self._sy

    def reset(self):
        """
        Call this when tracking is lost (hand leaves frame).
        Otherwise old smoothed values will "pull" the next position.
        """
        self._sx = None
        self._sy = None

    @property
    def value(self):
        """Returns last smoothed value, or (0, 0) if not yet initialized."""
        if self._sx is None:
            return (0.0, 0.0)
        return (self._sx, self._sy)


class MultiPointSmoother:
    """
    Smooths all 21 hand landmarks at once.
    Internally creates one EMAsmoother per landmark point.
    """

    def __init__(self, num_points=21, alpha=None):
        self.smoothers = [EMAsmoother(alpha) for _ in range(num_points)]

    def update(self, landmarks_list):
        """
        landmarks_list: list of (x, y) tuples, one per landmark (21 total).
        Returns: list of smoothed (x, y) tuples.
        """
        return [
            self.smoothers[i].update(landmarks_list[i][0], landmarks_list[i][1])
            for i in range(len(landmarks_list))
        ]

    def reset(self):
        """Reset all smoothers when hand tracking is lost."""
        for s in self.smoothers:
            s.reset()