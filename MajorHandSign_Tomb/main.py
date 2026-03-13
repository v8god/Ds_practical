# =============================================================
# main.py  —  Phase 3
#
# Added in this phase:
#   - DrawingCanvas : air drawing + eraser overlay
#
# Gesture → action mapping so far:
#   draw          → draw on canvas with index fingertip
#   erase         → erase canvas content near fingertip
#   click         → mouse click
#   drag          → mouse drag
#   select_start  → begin selection area
#   select_end    → end selection area
#   filter_next   → cycle to next filter
#
# Keyboard shortcuts:
#   f → cycle filters
#   l → toggle landmarks
#   m → toggle mouse control
#   c → clear canvas
#   s → save drawing
#   q → quit
# =============================================================
# =============================================================
# main.py  —  Phase 3 (updated)
#
# Changes in this update:
#   - Uses ShapeCanvas instead of DrawingCanvas
#   - Open palm swipe left/right → prev/next filter
#   - Thumb+middle select now does real OS drag selection
#   - Thumb+pinky pinch cycles shape mode
#   - Shape mode shows live preview, commits on gesture end
# =============================================================

# =============================================================
# main.py  —  Phase 4
#
# Added in this phase:
#   - AR3DEngine : extrudes drawn strokes into a rotating 3D object
#
# New keyboard shortcut:
#   t → build 3D object from current drawing + toggle 3D mode
#   r → reset / clear 3D object
#
# Workflow:
#   1. Draw something with your finger (index only = draw gesture)
#   2. Press 't' — your drawing is extruded into a 3D object
#   3. The object auto-rotates slowly
#   4. Use drag gesture (index+middle) while 3D mode is active
#      to manually rotate the object with your hand
#   5. Press 't' again to toggle 3D off (drawing stays on canvas)
#   6. Press 'r' to clear the 3D object entirely
# =============================================================
# =============================================================
# main.py  —  Phase 4 (updated — 3D Primitives)
#
# THREE MODES now exist:
#   "canvas"     → 2D drawing / shapes (default)
#   "3d_stroke"  → stroke-based pseudo-3D (press t)
#   "3d_prims"   → proper 3D primitives (press g)
#
# 3D PRIMITIVES MODE KEYS:
#   g       → toggle primitives mode on/off
#   n       → next shape  (sphere→cuboid→cone→cylinder)
#   p       → next parameter  (radius / width / height / depth)
#   =       → increase active parameter value
#   -       → decrease active parameter value
#   Enter   → place shape in scene
#   Delete  → delete selected shape
#   Tab     → cycle selection through placed shapes
#   s       → save current scene as PNG (in primitives mode)
#   c       → clear entire scene
#
# EXISTING KEYS STILL WORK:
#   t → stroke-3D toggle | r → reset rotation
#   z/x → depth spread   | f → filter | l → landmarks
#   m → mouse toggle      | q → quit
# =============================================================
# =============================================================
# main.py  —  Phase 4 (final clean version)
#
# THREE MODES:
#   "canvas"    → 2D drawing + shapes  (default)
#   "3d_stroke" → stroke-to-3D         (press t)
#   "3d_prims"  → 3D primitives        (press g)
#
# KEY REFERENCE:
# ─────────────────────────────────────────────────
#  UNIVERSAL:
#   q  = quit
#   f  = cycle filter (keyboard)
#   l  = toggle landmark dots on/off
#   m  = toggle mouse control on/off
#
#  CANVAS MODE:
#   c  = clear canvas
#   s  = save drawing
#   t  = convert drawing to stroke-3D
#
#  3D STROKE MODE  (press t to enter):
#   t  = back to canvas
#   r  = reset rotation
#   z  = increase depth spread
#   x  = decrease depth spread
#   c  = clear 3D + back to canvas
#
#  3D PRIMITIVES MODE  (press g to enter):
#   g  = back to canvas
#   n  = next shape (sphere/cuboid/cone/cylinder/pyramid)
#   p  = cycle active parameter
#   =  = increase parameter
#   -  = decrease parameter
#   Enter     = place shape at center
#   Tab       = select next placed shape
#   Del / BS  = delete selected shape
#   r  = reset rotation
#   c  = clear entire scene
#   s  = save scene as PNG
#   drag gesture (index+middle up) = grab & move shape
# =============================================================
# =============================================================
# main.py  —  Phase 4 (final clean version)
#
# THREE MODES:
#   "canvas"    → 2D drawing + shapes  (default)
#   "3d_stroke" → stroke-to-3D         (press t)
#   "3d_prims"  → 3D primitives        (press g)
#
# KEY REFERENCE:
# ─────────────────────────────────────────────────
#  UNIVERSAL:
#   q  = quit
#   f  = cycle filter (keyboard)
#   l  = toggle landmark dots on/off
#   m  = toggle mouse control on/off
#
#  CANVAS MODE:
#   c  = clear canvas
#   s  = save drawing
#   t  = convert drawing to stroke-3D
#
#  3D STROKE MODE  (press t to enter):
#   t  = back to canvas
#   r  = reset rotation
#   z  = increase depth spread
#   x  = decrease depth spread
#   c  = clear 3D + back to canvas
#
#  3D PRIMITIVES MODE  (press g to enter):
#   g  = back to canvas
#   n  = next shape (sphere/cuboid/cone/cylinder/pyramid)
#   p  = cycle active parameter
#   =  = increase parameter
#   -  = decrease parameter
#   Enter     = place shape at center
#   Tab       = select next placed shape
#   Del / BS  = delete selected shape
#   r  = reset rotation
#   c  = clear entire scene
#   s  = save scene as PNG
#   drag gesture (index+middle up) = grab & move shape
# =============================================================
# =============================================================
# main.py  —  Phase 5
#
# Added in this phase:
#   - FaceRecognizer  : detects + identifies faces every N frames
#   - FaceAnimator    : plays aura animation for known faces,
#                       warning animation for unknown faces
#
# HOW TO ADD YOUR FACE:
#   1. Put a clear photo in  assets/known_faces/yourname.jpg
#   2. Run the app — encoding generated automatically on startup
#   3. Stand in front of camera — your name appears above your face
#
# KEY REFERENCE:
#   UNIVERSAL:  q=quit  f=filter  l=landmarks  m=mouse
#   CANVAS:     c=clear  s=save  t=stroke-3D
#   3D STROKE:  t=back  r=reset  z/x=depth  drag=rotate
#   3D PRIMS:   g=back  n=shape  p=param  =/- adjust
#               Enter=place  Tab=select  Del=delete
#               drag(empty)=rotate  drag(shape)=move
#               r=reset  c=clear  s=save
# =============================================================
# =============================================================
# main.py  —  Phase 5
#
# Added in this phase:
#   - FaceRecognizer  : detects + identifies faces every N frames
#   - FaceAnimator    : plays aura animation for known faces,
#                       warning animation for unknown faces
#
# HOW TO ADD YOUR FACE:
#   1. Put a clear photo in  assets/known_faces/yourname.jpg
#   2. Run the app — encoding generated automatically on startup
#   3. Stand in front of camera — your name appears above your face
#
# KEY REFERENCE:
#   UNIVERSAL:  q=quit  f=filter  l=landmarks  m=mouse
#   CANVAS:     c=clear  s=save  t=stroke-3D
#   3D STROKE:  t=back  r=reset  z/x=depth  drag=rotate
#   3D PRIMS:   g=back  n=shape  p=param  =/- adjust
#               Enter=place  Tab=select  Del=delete
#               drag(empty)=rotate  drag(shape)=move
#               r=reset  c=clear  s=save
# =============================================================

import cv2
import config

from modules.camera                import Camera
from modules.filters               import FilterEngine
from modules.hand_tracker          import HandTracker
from modules.gesture_engine        import GestureEngine
from modules.mouse_controller      import MouseController
from modules.drawing_canvas        import ShapeCanvas
from modules.ar_3d                 import AR3DEngine
from modules.ar_overlay            import AROverlay
from modules.primitives_3d         import Primitives3DEngine
from modules.face_recognition_module import FaceRecognizer
from modules.face_animator         import FaceAnimator
from modules.sign_detector         import SignDetector
from modules.sign_animator         import SignAnimator
from utils.fps_counter             import FPSCounter


def main():
    # ----------------------------------------------------------
    # SETUP
    # ----------------------------------------------------------
    cam = Camera()
    cam.open()

    filters    = FilterEngine()
    fps        = FPSCounter()
    tracker    = HandTracker()
    engine     = GestureEngine()
    controller = MouseController()
    canvas     = ShapeCanvas(config.FRAME_WIDTH, config.FRAME_HEIGHT)
    ar3d       = AR3DEngine(config.FRAME_WIDTH, config.FRAME_HEIGHT)
    overlay    = AROverlay(config.FRAME_WIDTH, config.FRAME_HEIGHT, ar3d)
    prims      = Primitives3DEngine(config.FRAME_WIDTH, config.FRAME_HEIGHT)
    face_recog = FaceRecognizer()
    face_anim   = FaceAnimator()
    sign_detect = SignDetector()
    sign_anim   = SignAnimator()
    face_recog_enabled = True

    mode       = "canvas"
    last_frame = None

    print(f"\n{'='*55}")
    print(" AR Studio — Phase 5")
    print(f"{'='*55}")
    print("  Add face photo to assets/known_faces/name.jpg")
    print("  t=stroke3D  g=primitives  q=quit  f=filter")
    print("  drag=rotate(3D modes)  drag(near shape)=move shape")
    print(f"{'='*55}\n")

    # ----------------------------------------------------------
    # MAIN LOOP
    # ----------------------------------------------------------
    while True:

        # 1. CAPTURE
        frame = cam.read()
        if frame is None:
            break

        # 2. HAND TRACKING
        landmarks = tracker.update(frame)

        # 3. GESTURE
        gesture   = engine.update(landmarks)

        # Index fingertip for drag/primitives
        finger_xy = None
        if landmarks is not None:
            finger_xy = (int(landmarks[8][0]), int(landmarks[8][1]))

        # 4. FILTER SWIPE — canvas mode only
        if mode == "canvas":
            if gesture == "filter_next":
                filters.cycle()
            elif gesture == "filter_prev":
                filters.prev()

        # 5. MOUSE — canvas mode, not drawing/erasing
        if mode == "canvas" and gesture not in ("draw", "erase"):
            controller.update(landmarks, gesture, engine)

        # 6. FACE RECOGNITION — toggle with 'v', runs every N frames
        if face_recog_enabled:
            face_recog.update(frame)

        # Trigger animation when recognition result changes
        if face_recog.trigger_animation:
            if face_recog.face_boxes:
                face_anim.trigger_known(face_recog.recognized_name,
                                        face_recog.face_boxes[0])
        elif (face_recog.recognized_name == "unknown"
              and face_recog.face_boxes
              and not face_anim.is_playing):
            face_anim.trigger_unknown(face_recog.face_boxes[0])

        # 6b. SIGN DETECTION + ANIMATION TRIGGER
        triggered_sign = sign_detect.update(landmarks)
        if triggered_sign and not sign_anim.is_playing:
            # Hand center for animation origin
            if landmarks is not None:
                hcx = int(sum(lm[0] for lm in landmarks)/len(landmarks))
                hcy = int(sum(lm[1] for lm in landmarks)/len(landmarks))
            else:
                hcx, hcy = config.FRAME_WIDTH//2, config.FRAME_HEIGHT//2
            sign_anim.trigger(triggered_sign, hcx, hcy)

        # 7. APPLY FILTER
        frame = filters.apply(frame)

        # 8. MODE RENDER
        if mode == "canvas":
            frame = canvas.update(frame, landmarks, gesture,
                                  shape_mode=engine.shape_mode)

        elif mode == "3d_stroke":
            frame = overlay.update(frame, landmarks, gesture)

        elif mode == "3d_prims":
            frame = (frame.astype(float) * 0.72).clip(0, 255).astype('uint8')
            frame = prims.update(frame, landmarks, gesture, finger_xy)
            prims.draw_hud(frame)

        # 9. FACE OVERLAYS
        if face_recog_enabled:
            face_recog.draw(frame)
            face_anim.update(frame)

        # 9b. SIGN ANIMATION (fullscreen — on top of everything)
        if sign_anim.is_playing:
            frame = sign_anim.update(frame)
            sign_anim.draw_hud(frame)
        else:
            sign_detect.draw_hud(frame)

        # 10. HAND OVERLAYS
        tracker.draw(frame)
        engine.draw_hud(frame)

        if mode == "canvas":
            canvas.draw_hud(frame, gesture)
            overlay.draw_hud(frame)
            if gesture not in ("draw", "erase"):
                controller.draw_hud(frame, landmarks)

        elif mode == "3d_stroke":
            overlay.draw_hud(frame)
            ar3d.draw_hud(frame)

        elif mode == "3d_prims":
            cv2.putText(frame, "MODE: 3D PRIMITIVES  [g=exit]",
                        (frame.shape[1]//2 - 140, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 200), 2, cv2.LINE_AA)

        # 11. HUD
        if config.SHOW_FPS:
            fps.update()
            fps.draw(frame)
        filters.draw_hud(frame)

        last_frame = frame.copy()

        # 12. SHOW
        cv2.imshow(config.WINDOW_NAME, frame)

        # 13. KEYS
        key = cv2.waitKey(1) & 0xFF

        if   key == config.KEY_QUIT:              break
        elif key == config.KEY_CYCLE_FILTER:      filters.cycle()
        elif key == config.KEY_TOGGLE_LANDMARKS:  tracker.toggle_landmarks()
        elif key == ord('v'):
            face_recog_enabled = not face_recog_enabled
            print(f"[main] Face recognition: {'ON' if face_recog_enabled else 'OFF'}")
        elif key == ord('m'):                     controller.toggle()

        elif key == ord('t'):
            if mode == "canvas":
                if overlay.convert_to_3d(canvas.strokes):
                    mode = "3d_stroke"
            elif mode == "3d_stroke":
                overlay.back_to_canvas()
                mode = "canvas"

        elif key == ord('g'):
            mode = "3d_prims" if mode != "3d_prims" else "canvas"
            print(f"[main] Mode → {mode}")

        elif mode == "canvas":
            if   key == config.KEY_CLEAR_CANVAS:  canvas.clear(); ar3d.clear()
            elif key == config.KEY_SAVE_DRAWING:  canvas.save()

        elif mode == "3d_stroke":
            if   key == ord('r'): ar3d.reset_rotation()
            elif key == ord('z'): ar3d.increase_depth()
            elif key == ord('x'): ar3d.decrease_depth()
            elif key == config.KEY_CLEAR_CANVAS:
                ar3d.clear(); overlay.back_to_canvas(); mode = "canvas"

        elif mode == "3d_prims":
            if   key == ord('n'):                 prims.cycle_shape()
            elif key == ord('p'):                 prims.cycle_param()
            elif key in (ord('='), ord('+')):     prims.param_up()
            elif key == ord('-'):                 prims.param_down()
            elif key == 13:                       prims.place_shape()
            elif key in (127, 8):                 prims.delete_selected()
            elif key == 9:                        prims.select_next()
            elif key == ord('r'):                 prims.reset_rotation()
            elif key == config.KEY_CLEAR_CANVAS:  prims.clear_scene()
            elif key == config.KEY_SAVE_DRAWING:
                if last_frame is not None:        prims.save_scene(last_frame)

    # ----------------------------------------------------------
    # CLEANUP
    # ----------------------------------------------------------
    cam.release()
    cv2.destroyAllWindows()
    print("[main] Shutdown complete.")


if __name__ == "__main__":
    main()
