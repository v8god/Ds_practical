# =============================================================
# config.py
# ALL settings for the entire AR Studio project live here.
# Never hardcode values inside modules — change them here only.
# =============================================================

# -------------------------------------------------------------
# CAMERA
# -------------------------------------------------------------
CAMERA_INDEX = 0          # 0 = default webcam. Change to 1 if you have external cam.
# FRAME_WIDTH  = 1280
FRAME_WIDTH  = 500       # Capture width in pixels
# FRAME_HEIGHT = 720
FRAME_HEIGHT = 282        # Capture height in pixels
TARGET_FPS   = 30         # Requested FPS from camera (not guaranteed, depends on hardware)

# -------------------------------------------------------------
# DISPLAY
# -------------------------------------------------------------
WINDOW_NAME  = "AR Studio"
SHOW_FPS     = True       # Toggle FPS counter on screen
SHOW_LANDMARKS = True     # Toggle hand landmark dots on screen

# -------------------------------------------------------------
# FILTERS
# List of filter names in the order they cycle through.
# The index into this list is what gets stored as active_filter.
# -------------------------------------------------------------
FILTER_NAMES = [
    "none",         # 0 - Raw camera feed
    "bw",           # 1 - Black and White
    "sepia",        # 2 - Warm sepia tone
    "paris_blue",   # 3 - Cool blue tone (Paris filter)
    "sketch",       # 4 - Pencil sketch effect
    "edge",         # 5 - Edge detection (Canny)
    "cartoon",      # 6 - Cartoon / cel-shaded look
]
DEFAULT_FILTER = 0        # Start with no filter

# Key to press to cycle filters (keyboard fallback, before gesture switching)
FILTER_KEY = ord('f')     # Press 'f' to cycle filters

# -------------------------------------------------------------
# HAND TRACKING  (Phase 2 — defined here early so config is complete)
# -------------------------------------------------------------
MAX_HANDS            = 1      # Detect 1 hand (increase to 2 later if needed)
DETECTION_CONFIDENCE = 0.7    # Min confidence to detect a hand (0.0 - 1.0)
TRACKING_CONFIDENCE  = 0.6    # Min confidence to keep tracking (0.0 - 1.0)

# Pinch distance threshold (in pixels) — below this = fingers are touching
PINCH_THRESHOLD = 35

# Smoothing factor for EMA (Exponential Moving Average) — 0=no smoothing, 1=frozen
# 0.5 is a good balance between responsiveness and stability
SMOOTHING_ALPHA = 0.5

# -------------------------------------------------------------
# MOUSE CONTROL  (Phase 2)
# -------------------------------------------------------------
# The camera frame is mapped to the full screen.
# These margins (in pixels) trim the edges of the camera frame
# so the cursor can actually reach the screen corners.
MOUSE_MARGIN_X = 100
MOUSE_MARGIN_Y = 60

# -------------------------------------------------------------
# DRAWING CANVAS  (Phase 3)
# -------------------------------------------------------------
DRAW_COLOR        = (0, 255, 255)   # Cyan lines (BGR)
DRAW_THICKNESS    = 5               # Line thickness in pixels
ERASER_RADIUS     = 40              # Radius of eraser circle in pixels
CANVAS_ALPHA      = 0.6             # Opacity of drawing overlay on video (0.0-1.0)

# -------------------------------------------------------------
# FACE RECOGNITION  (Phase 5)
# -------------------------------------------------------------
FACE_RECOGNITION_EVERY_N_FRAMES = 5   # Only run face recognition every N frames
                                       # (saves CPU — face doesn't change frame to frame)
FACE_TOLERANCE = 0.55                  # Lower = stricter match. 0.6 is default.
KNOWN_FACES_DIR = "assets/known_faces" # Folder with reference face images

# -------------------------------------------------------------
# SYMBOL DETECTION  (Phase 6)
# -------------------------------------------------------------
SYMBOL_DETECTION_EVERY_N_FRAMES = 10  # Run symbol detection every N frames
SYMBOLS_DIR = "assets/symbols"        # Folder with target symbol images
ORB_MATCH_THRESHOLD = 15              # Min good matches to consider symbol detected

# -------------------------------------------------------------
# AR OVERLAY  (Phase 4-6)
# -------------------------------------------------------------
ANIMATION_DIR = "assets/animations"  # Folder with animation frames / PNGs

# -------------------------------------------------------------
# KEYS  (keyboard shortcuts active throughout)
# -------------------------------------------------------------
KEY_QUIT          = ord('q')   # Press 'q' to exit
KEY_CYCLE_FILTER  = ord('f')   # Press 'f' to cycle filters
KEY_CLEAR_CANVAS  = ord('c')   # Press 'c' to clear drawing canvas
KEY_SAVE_DRAWING  = ord('s')   # Press 's' to save current drawing
KEY_TOGGLE_LANDMARKS = ord('l')  # Press 'l' to show/hide landmarks