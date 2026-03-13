# =============================================================
# modules/sign_animator.py
#
# All animation rendering for hand sign triggers.
#
# ── HOW TO ADD A NEW MATHEMATICAL ANIMATION ─────────────────
#   1. Add a method:  def _anim_yourname(self, frame, t, cx, cy)
#        t   = float 0.0→1.0 (animation progress)
#        cx,cy = hand center in pixels
#        return frame
#   2. Register it in __init__:
#        self.registry["your_sign_key"] = Animation(
#            name="yourname", duration=90)
#
# ── HOW TO USE A VIDEO FILE INSTEAD ─────────────────────────
#   self.registry["your_sign_key"] = Animation(
#       name="yourname", duration=90,
#       video_path="assets/animations/your_video.mp4")
#
# ── HOW TO REPLACE AN EXISTING SIGN'S ANIMATION WITH VIDEO ──
#   Find the sign in __init__ and add video_path=... to it.
#   Example:
#       self.registry["domain_expansion"] = Animation(
#           name="domain_expansion", duration=120,
#           video_path="assets/animations/domain.mp4")
#
# ── VIDEO BLENDING ───────────────────────────────────────────
#   Videos are overlaid on the live camera frame using
#   blend_mode (see Animation dataclass below):
#     "overlay"  → video drawn on top at video_alpha opacity
#     "screen"   → additive blend (bright pixels show through)
#     "multiply" → darkening blend
#   Default is "overlay" at alpha=0.85.
#   Change per-animation by setting blend_mode and video_alpha.
#
# ── ANIMATION DURATION ───────────────────────────────────────
#   duration = number of frames the animation plays.
#   At 30fps: 30=1s, 60=2s, 90=3s, 120=4s.
# =============================================================

import cv2
import numpy as np
import math
import random
import os
from dataclasses import dataclass, field
from typing import Optional


# ----------------------------------------------------------
@dataclass
class Animation:
    name:        str
    duration:    int              = 90
    video_path:  Optional[str]   = None    # None = use math animation
    blend_mode:  str             = "overlay"
    video_alpha: float           = 0.85
    _cap:        object          = field(default=None, repr=False, compare=False)
    _loaded:     bool            = field(default=False, repr=False, compare=False)

    def load_video(self):
        """Load video capture if video_path is set and file exists."""
        if self.video_path and os.path.exists(self.video_path):
            self._cap    = cv2.VideoCapture(self.video_path)
            self._loaded = True
            print(f"[SignAnimator] Video loaded: {self.video_path}")
        elif self.video_path:
            print(f"[SignAnimator] Video NOT found: {self.video_path} "
                  f"— falling back to math animation.")
            self._loaded = False

    def reset_video(self):
        """Rewind video to start for next play."""
        if self._cap and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get_video_frame(self, target_w, target_h):
        """
        Read next video frame, resized to target dimensions.
        Returns None if video ended or not available.
        """
        if not self._loaded or not self._cap:
            return None
        ret, vframe = self._cap.read()
        if not ret:
            return None
        return cv2.resize(vframe, (target_w, target_h))

    def use_video(self):
        return self._loaded and self._cap is not None


# ----------------------------------------------------------
class SignAnimator:
    """
    Plays the animation associated with a triggered hand sign.

    Usage:
        animator.trigger("domain_expansion", hand_cx, hand_cy)
        frame = animator.update(frame)   # call every frame
    """

    def __init__(self):
        # ── ANIMATION REGISTRY ──────────────────────────────
        # To replace any math animation with a video:
        #   Add   video_path="assets/animations/yourfile.mp4"
        # To change duration: change duration=N (frames)
        self.registry = {

            "domain_expansion": Animation(
                name="domain_expansion",
                duration=120,
                # video_path="assets/animations/domain_expansion.mp4",
            ),

            "black_flash": Animation(
                name="black_flash",
                duration=75,
                # video_path="assets/animations/black_flash.mp4",
            ),

            "divergent_fist": Animation(
                name="divergent_fist",
                duration=80,
                # video_path="assets/animations/divergent_fist.mp4",
            ),

            "red_blue": Animation(
                name="red_blue",
                duration=100,
                # video_path="assets/animations/red_blue.mp4",
            ),

            "binding_vow": Animation(
                name="binding_vow",
                duration=110,
                # video_path="assets/animations/binding_vow.mp4",
            ),

            "reverse_cursed": Animation(
                name="reverse_cursed",
                duration=90,
                # video_path="assets/animations/reverse_cursed.mp4",
            ),
        }

        # Load video captures for any that have video_path set
        for anim in self.registry.values():
            anim.load_video()

        # Playback state
        self._active_sign   = None
        self._frame_counter = 0
        self._hand_cx       = 0
        self._hand_cy       = 0
        self._rng_seed      = 0   # fixed per-animation for consistency

    # ----------------------------------------------------------
    def trigger(self, sign_name, hand_cx, hand_cy):
        """Start playing the animation for sign_name."""
        if sign_name not in self.registry:
            print(f"[SignAnimator] Unknown sign: {sign_name}")
            return

        anim = self.registry[sign_name]
        self._active_sign   = sign_name
        self._frame_counter = anim.duration
        self._hand_cx       = hand_cx
        self._hand_cy       = hand_cy
        self._rng_seed      = random.randint(0, 9999)

        if anim.use_video():
            anim.reset_video()

        print(f"[SignAnimator] Playing: {sign_name} "
              f"({'video' if anim.use_video() else 'math'})")

    # ----------------------------------------------------------
    def update(self, frame):
        """
        Render current animation frame onto frame.
        Call every frame.
        """
        if self._active_sign is None or self._frame_counter <= 0:
            self._active_sign = None
            return frame

        anim     = self.registry[self._active_sign]
        total    = anim.duration
        elapsed  = total - self._frame_counter
        t        = elapsed / max(1, total - 1)   # 0.0 → 1.0

        self._frame_counter -= 1

        h, w = frame.shape[:2]
        cx   = self._hand_cx
        cy   = self._hand_cy

        # ── VIDEO PLAYBACK ───────────────────────────────────
        if anim.use_video():
            vframe = anim.get_video_frame(w, h)
            if vframe is not None:
                frame = self._blend_video(frame, vframe,
                                          anim.blend_mode,
                                          anim.video_alpha)
            else:
                # Video ended early — stop animation
                self._active_sign = None
            return frame

        # ── MATH ANIMATION ───────────────────────────────────
        method_name = f"_anim_{anim.name}"
        method      = getattr(self, method_name, None)
        if method:
            frame = method(frame, t, cx, cy)
        else:
            print(f"[SignAnimator] No method: {method_name}")
            self._active_sign = None

        return frame

    # ----------------------------------------------------------
    @property
    def is_playing(self):
        return self._active_sign is not None and self._frame_counter > 0

    # ----------------------------------------------------------
    # VIDEO BLEND MODES
    # ----------------------------------------------------------
    def _blend_video(self, cam_frame, vid_frame, mode, alpha):
        """Composite video frame onto camera frame."""
        if mode == "overlay":
            return cv2.addWeighted(cam_frame, 1.0 - alpha,
                                   vid_frame,       alpha, 0)

        elif mode == "screen":
            # Screen: 1-(1-a)*(1-b) — bright pixels show through
            cam_f = cam_frame.astype(np.float32) / 255.0
            vid_f = vid_frame.astype(np.float32) / 255.0
            result = 1.0 - (1.0-cam_f)*(1.0-vid_f*alpha)
            return (result.clip(0,1)*255).astype(np.uint8)

        elif mode == "multiply":
            cam_f = cam_frame.astype(np.float32) / 255.0
            vid_f = vid_frame.astype(np.float32) / 255.0
            result = cam_f * (1.0 - alpha + vid_f*alpha)
            return (result.clip(0,1)*255).astype(np.uint8)

        return cv2.addWeighted(cam_frame, 1.0-alpha, vid_frame, alpha, 0)

    # ==========================================================
    # MATH ANIMATIONS
    # Each method: (self, frame, t, cx, cy) → frame
    # t = 0.0 at start, 1.0 at end
    # cx,cy = hand center in pixels
    # ==========================================================

    # ----------------------------------------------------------
    def _anim_domain_expansion(self, frame, t, cx, cy):
        """
        DOMAIN EXPANSION — Unlimited Void / Hollow Purple
        Phase 1 (t=0→0.3): Screen cracks and tears. Dark vignette.
        Phase 2 (t=0.3→0.7): Void portal expands from hand. Purple+black.
        Phase 3 (t=0.7→1.0): Hollow Purple orbs collide at center. Flash.
        """
        h, w = frame.shape[:2]

        # ── Phase 1: vignette + cracks ───────────────────────
        if t < 0.3:
            p = t / 0.3   # 0→1 within phase

            # Dark vignette
            frame = self._vignette(frame, strength=p * 0.7)

            # Screen shake (pixel shift)
            shake = int(p * 8 * math.sin(t * 80))
            M     = np.float32([[1,0,shake],[0,1,shake//2]])
            frame = cv2.warpAffine(frame, M, (w,h))

            # Crack lines from hand center
            random.seed(self._rng_seed)
            for _ in range(int(p * 12)):
                angle  = random.uniform(0, 2*math.pi)
                length = random.uniform(80, 300) * p
                x2     = int(cx + math.cos(angle)*length)
                y2     = int(cy + math.sin(angle)*length)
                thick  = random.randint(1,2)
                col    = (random.randint(60,120),
                          0,
                          random.randint(80,160))
                self._jagged_line(frame, (cx,cy), (x2,y2), col, thick, 4)

        # ── Phase 2: void portal ─────────────────────────────
        elif t < 0.7:
            p = (t - 0.3) / 0.4

            frame = self._vignette(frame, strength=0.7 + p*0.2)

            # Expanding void circle — black inside
            r = int(p * max(w,h) * 0.7)
            overlay = frame.copy()
            cv2.circle(overlay, (cx,cy), r, (0,0,0), -1)
            cv2.addWeighted(overlay, p*0.9, frame, 1-p*0.9, 0, frame)

            # Purple glow ring at void edge
            for ring in range(3):
                rr    = r - ring*8
                alpha = 1.0 - ring*0.3
                col   = (int(180*alpha), 0, int(255*alpha))
                if rr > 0:
                    cv2.circle(frame, (cx,cy), rr, col, 2+ring, cv2.LINE_AA)

            # Stars/particles inside void
            random.seed(self._rng_seed + int(t*100))
            for _ in range(30):
                px = random.randint(max(0,cx-r), min(w-1,cx+r))
                py = random.randint(max(0,cy-r), min(h-1,cy+r))
                if math.hypot(px-cx,py-cy) < r:
                    cv2.circle(frame,(px,py),random.randint(1,3),(200,100,255),-1)

        # ── Phase 3: Hollow Purple collision ─────────────────
        else:
            p = (t - 0.7) / 0.3

            frame = self._vignette(frame, strength=0.9*(1-p))

            # Two orbs converging to center
            red_x  = int(w*0.1  + (w*0.5 - w*0.1) * p)
            blue_x = int(w*0.9  - (w*0.9 - w*0.5) * p)
            orb_r  = int(60 - p*40)

            if orb_r > 0:
                self._glow_circle(frame, (red_x,  h//2), orb_r, (0,50,255), 3)
                self._glow_circle(frame, (blue_x, h//2), orb_r, (255,80,0), 3)

            # On collision (p > 0.85): white flash
            if p > 0.85:
                intensity = int((p-0.85)/0.15 * 255)
                flash     = np.full_like(frame, intensity)
                cv2.addWeighted(flash, 0.7, frame, 0.3, 0, frame)
                # Shockwave
                sr = int((p-0.85)/0.15 * w * 0.8)
                cv2.circle(frame,(w//2,h//2),sr,(180,0,255),3,cv2.LINE_AA)

        # Title text
        self._slam_text(frame, "DOMAIN EXPANSION", t, (w//2, 80),
                        (180, 0, 255))
        return frame

    # ----------------------------------------------------------
    def _anim_black_flash(self, frame, t, cx, cy):
        """
        BLACK FLASH — electric cursed energy discharge
        Phase 1: Screen goes black suddenly
        Phase 2: Recursive electric lightning bursts from fingertip
        Phase 3: Afterimage fade
        """
        h, w = frame.shape[:2]
        random.seed(self._rng_seed + int(t * 50))

        if t < 0.15:
            # Sudden blackout
            p     = t / 0.15
            black = np.zeros_like(frame)
            cv2.addWeighted(black, p, frame, 1-p, 0, frame)

        elif t < 0.75:
            p = (t - 0.15) / 0.6

            # Keep frame dark
            dark = (frame.astype(float) * 0.15).astype(np.uint8)
            frame = dark

            # Lightning bursts — redrawn every 2 frames for flicker
            num_bolts = 6 + int(p * 6)
            for _ in range(num_bolts):
                angle  = random.uniform(0, 2*math.pi)
                length = random.uniform(100, min(w,h)*0.6) * (0.4 + p*0.6)
                ex     = int(cx + math.cos(angle)*length)
                ey     = int(cy + math.sin(angle)*length)
                self._lightning(frame, (cx,cy), (ex,ey), depth=3)

            # Impact glow at hand center
            glow_r = int(20 + p*40)
            self._glow_circle(frame, (cx,cy), glow_r, (200,200,255), 4)

            # Screen edge electric arcs
            for _ in range(4):
                sx = random.choice([0, w-1])
                sy = random.randint(0, h-1)
                ex = random.randint(w//4, 3*w//4)
                ey = random.randint(h//4, 3*h//4)
                self._lightning(frame, (sx,sy), (ex,ey), depth=2)

        else:
            # Fade back
            p     = (t - 0.75) / 0.25
            black = np.zeros_like(frame)
            dark  = (frame.astype(float)*0.15).astype(np.uint8)
            cv2.addWeighted(frame, p, dark, 1-p, 0, frame)

        self._slam_text(frame, "BLACK FLASH", t, (w//2, 80), (150, 150, 255))
        return frame

    # ----------------------------------------------------------
    def _anim_divergent_fist(self, frame, t, cx, cy):
        """
        DIVERGENT FIST — concentric shockwave rings + pixel distortion
        """
        h, w = frame.shape[:2]

        # Distortion map: sine-wave pixel displacement
        if t < 0.8:
            strength = math.sin(t * math.pi) * 25
            map_x    = np.zeros((h,w), dtype=np.float32)
            map_y    = np.zeros((h,w), dtype=np.float32)
            for row in range(h):
                for col in range(w):
                    dx = col - cx
                    dy = row - cy
                    d  = max(1, math.hypot(dx,dy))
                    wave = math.sin(d * 0.08 - t*15) * strength / d * 30
                    map_x[row,col] = col + dx/d * wave
                    map_y[row,col] = row + dy/d * wave
            frame = cv2.remap(frame, map_x, map_y,
                              cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

        # Shockwave rings (5 rings at different phases)
        num_rings = 5
        max_r     = int(math.hypot(w,h))
        for i in range(num_rings):
            phase   = (t + i/num_rings) % 1.0
            r       = int(phase * max_r)
            alpha   = 1.0 - phase
            thick   = max(1, int(3*(1-phase)))
            r_col   = int(255*(1-phase))
            g_col   = int(150*(1-phase))
            col     = (0, g_col, r_col)
            if alpha > 0.05 and r < max_r:
                cv2.circle(frame,(cx,cy),r,col,thick,cv2.LINE_AA)

        # Impact flash at start
        if t < 0.1:
            intensity = int((1-t/0.1)*200)
            flash     = np.full_like(frame,intensity)
            cv2.addWeighted(flash, 0.6, frame, 0.4, 0, frame)

        self._slam_text(frame, "DIVERGENT FIST", t, (w//2,80),(255,150,50))
        return frame

    # ----------------------------------------------------------
    def _anim_red_blue(self, frame, t, cx, cy):
        """
        RED / BLUE — Convergence: two cursed energy orbs collide
        Red orb from left, Blue orb from right, collision at center.
        """
        h, w = frame.shape[:2]

        # Color tint based on which side
        tint       = np.zeros_like(frame, dtype=np.float32)
        tint_str   = math.sin(t * math.pi) * 0.3

        # Red tint left half, blue tint right half
        tint[:, :w//2, 2] = 60 * tint_str   # red left
        tint[:, w//2:, 0] = 60 * tint_str   # blue right
        frame = cv2.add(frame, tint.astype(np.uint8))

        # Orb positions: start at edges, meet at center
        red_x  = int(w*0.05 + (w*0.5 - w*0.05)*t)
        blue_x = int(w*0.95 - (w*0.95 - w*0.5)*t)
        orb_y  = h//2
        orb_r  = int(50 + math.sin(t*math.pi*3)*10)

        # Draw orbs with glow
        self._glow_circle(frame, (red_x,  orb_y), orb_r, (0,  50, 255), 5)
        self._glow_circle(frame, (blue_x, orb_y), orb_r, (255, 50,   0), 5)

        # Energy trail
        for i in range(8):
            trail_t = max(0, t - i*0.03)
            tx_r    = int(w*0.05 + (w*0.5 - w*0.05)*trail_t)
            tx_b    = int(w*0.95 - (w*0.95 - w*0.5)*trail_t)
            alpha   = (1 - i/8) * 0.4
            tr      = int(orb_r * (1-i/8) * 0.7)
            if tr > 2:
                overlay = frame.copy()
                cv2.circle(overlay,(tx_r,orb_y),tr,(0,30,200),-1)
                cv2.circle(overlay,(tx_b,orb_y),tr,(200,30,0),-1)
                cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

        # Collision effect when orbs meet (t > 0.85)
        if t > 0.85:
            p = (t-0.85)/0.15
            # Hollow Purple: blended purple flash
            sr = int(p * max(w,h))
            cv2.circle(frame,(w//2,h//2),sr,(160,0,200),4,cv2.LINE_AA)
            intensity = int(p * 180)
            flash     = np.zeros_like(frame)
            flash[:,:,0] = intensity//2
            flash[:,:,2] = intensity
            cv2.addWeighted(flash,0.6,frame,0.4,0,frame)

        self._slam_text(frame,"CONVERGENCE",t,(w//2,80),(150,80,255))
        return frame

    # ----------------------------------------------------------
    def _anim_binding_vow(self, frame, t, cx, cy):
        """
        BINDING VOW — dark chains with cursed seals wrap the frame border.
        """
        h, w = frame.shape[:2]

        # Dark overlay that grows in
        dark_alpha = min(0.75, t*1.5)
        dark       = (frame.astype(float)*( 1-dark_alpha*0.6)).astype(np.uint8)
        frame      = dark

        # Purple-black tint
        tint         = np.zeros_like(frame,dtype=np.uint8)
        tint[:,:,0]  = int(30*dark_alpha)
        tint[:,:,2]  = int(20*dark_alpha)
        frame        = cv2.add(frame,tint)

        # Chains along border: perimeter = 2*(w+h)
        perimeter = 2*(w+h)
        chain_len = int(t * perimeter * 1.1)   # grows with t
        link_size = 28

        for pos in range(0, min(chain_len, perimeter), link_size):
            # Convert perimeter position to (x,y)
            if pos < w:
                px,py = pos, 0
            elif pos < w+h:
                px,py = w-1, pos-w
            elif pos < 2*w+h:
                px,py = w-1-(pos-w-h), h-1
            else:
                px,py = 0, h-1-(pos-2*w-h)

            # Chain link = rotated rectangle
            angle = (pos/link_size) * 45   # alternating angle
            self._chain_link(frame, (px,py), link_size//2, angle)

        # Cursed seal symbols at corners
        if t > 0.4:
            corners = [(30,30),(w-30,30),(30,h-30),(w-30,h-30)]
            for (sx,sy) in corners:
                self._cursed_seal(frame,(sx,sy),
                                  int(20*(t-0.4)/0.6))

        # Chains also from hand center to edges
        if t > 0.2:
            p = (t-0.2)/0.8
            random.seed(self._rng_seed)
            for _ in range(4):
                angle  = random.uniform(0,2*math.pi)
                length = p * random.uniform(100,300)
                ex     = int(cx+math.cos(angle)*length)
                ey     = int(cy+math.sin(angle)*length)
                self._draw_chain(frame,(cx,cy),(ex,ey))

        self._slam_text(frame,"BINDING VOW",t,(w//2,80),(120,0,180))
        return frame

    # ----------------------------------------------------------
    def _anim_reverse_cursed(self, frame, t, cx, cy):
        """
        REVERSE CURSED TECHNIQUE — green healing energy floods screen.
        Frame brightens, green radial glow, healing particles.
        """
        h, w = frame.shape[:2]

        # Brightness ramp up then down
        brightness = math.sin(t * math.pi) * 80
        frame      = cv2.convertScaleAbs(frame, alpha=1.0,
                                          beta=int(brightness))

        # Green radial glow from hand center
        Y, X    = np.ogrid[:h, :w]
        dist    = np.sqrt((X-cx)**2 + (Y-cy)**2).astype(np.float32)
        max_d   = math.hypot(w,h)
        glow_r  = max_d * (0.1 + t * 0.9)
        glow    = np.clip(1.0 - dist/glow_r, 0, 1)
        glow    = (glow * math.sin(t*math.pi) * 120).astype(np.uint8)

        overlay        = np.zeros_like(frame)
        overlay[:,:,1] = glow   # green channel
        cv2.addWeighted(overlay, 0.8, frame, 1.0, 0, frame)

        # Healing particles — green dots moving outward
        random.seed(self._rng_seed)
        num_particles = 60
        for i in range(num_particles):
            angle  = random.uniform(0, 2*math.pi)
            speed  = random.uniform(50, 250)
            px     = int(cx + math.cos(angle) * speed * t)
            py     = int(cy + math.sin(angle) * speed * t)
            size   = random.randint(2,6)
            alpha  = 1.0 - t
            if (0<=px<w and 0<=py<h and alpha>0.1):
                col = (0, int(200*alpha), int(100*alpha))
                cv2.circle(frame,(px,py),size,col,-1,cv2.LINE_AA)

        # Concentric green rings pulsing outward
        for ring in range(4):
            phase = (t + ring*0.25) % 1.0
            r     = int(phase * max_d * 0.8)
            alpha = 1.0 - phase
            col   = (0, int(200*alpha), int(100*alpha))
            if r > 0:
                cv2.circle(frame,(cx,cy),r,col,2,cv2.LINE_AA)

        self._slam_text(frame,"REVERSE CURSED TECHNIQUE",
                        t,(w//2,80),(0,255,120))
        return frame

    # ==========================================================
    # SHARED HELPER METHODS
    # (used by multiple animations — do not call directly)
    # ==========================================================

    def _vignette(self, frame, strength=0.6):
        """Dark vignette — edges go black."""
        h, w   = frame.shape[:2]
        Y, X   = np.ogrid[:h, :w]
        cx, cy = w//2, h//2
        dist   = np.sqrt((X-cx)**2+(Y-cy)**2) / math.hypot(cx,cy)
        mask   = np.clip(dist * strength, 0, 1)
        mask   = mask[:,:,np.newaxis]
        return (frame.astype(float)*(1-mask)).clip(0,255).astype(np.uint8)

    def _glow_circle(self, frame, center, radius, color, layers=4):
        """Draw a glowing circle (multiple fading rings)."""
        for i in range(layers):
            r     = radius + i*6
            alpha = 1.0 - i/layers
            col   = tuple(int(c*alpha) for c in color)
            thick = max(1, layers-i)
            cv2.circle(frame, center, r, col, thick, cv2.LINE_AA)

    def _lightning(self, frame, p1, p2, depth=3, color=None):
        """
        Recursive jagged lightning bolt between two points.
        depth controls how many times it splits (more = more complex).
        """
        if depth == 0 or math.hypot(p2[0]-p1[0],p2[1]-p1[1]) < 5:
            col = color or (200+random.randint(0,55),
                            200+random.randint(0,55),
                            255)
            cv2.line(frame, p1, p2, col, 1, cv2.LINE_AA)
            return

        # Midpoint with random perpendicular offset
        mx = (p1[0]+p2[0])//2 + random.randint(-20,20)
        my = (p1[1]+p2[1])//2 + random.randint(-20,20)
        mid = (mx,my)

        self._lightning(frame, p1,  mid, depth-1, color)
        self._lightning(frame, mid, p2,  depth-1, color)

    def _jagged_line(self, frame, p1, p2, color, thickness, segments):
        """Draw a jagged crack line with random offsets."""
        pts = [p1]
        for i in range(1, segments):
            frac = i/segments
            mx   = int(p1[0]+(p2[0]-p1[0])*frac + random.randint(-15,15))
            my   = int(p1[1]+(p2[1]-p1[1])*frac + random.randint(-15,15))
            pts.append((mx,my))
        pts.append(p2)
        for i in range(len(pts)-1):
            cv2.line(frame, pts[i], pts[i+1], color, thickness, cv2.LINE_AA)

    def _chain_link(self, frame, center, size, angle_deg):
        """Draw one chain link (rotated rectangle) at center."""
        x,y = center
        col = (80,0,100)
        cv2.rectangle(frame,
                      (x-size//2, y-size//4),
                      (x+size//2, y+size//4),
                      col, 2)
        # Inner oval suggestion
        cv2.ellipse(frame,(x,y),(size//3,size//5),
                    angle_deg,0,360,(120,0,160),1)

    def _draw_chain(self, frame, p1, p2):
        """Draw chain links along a line."""
        dx   = p2[0]-p1[0]
        dy   = p2[1]-p1[1]
        dist = max(1,math.hypot(dx,dy))
        steps= max(1,int(dist//20))
        for i in range(steps):
            t  = i/steps
            px = int(p1[0]+dx*t)
            py = int(p1[1]+dy*t)
            self._chain_link(frame,(px,py),10,i*45)

    def _cursed_seal(self, frame, center, size):
        """Draw a simple cursed seal symbol (overlapping geometry)."""
        if size < 3:
            return
        x,y = center
        col = (140,0,180)
        cv2.circle(frame,(x,y),size,col,1)
        pts = np.array([
            [x,       y-size],
            [x+size,  y+size],
            [x-size,  y+size],
        ],np.int32)
        cv2.polylines(frame,[pts],True,col,1)
        cv2.line(frame,(x-size,y),(x+size,y),col,1)
        cv2.line(frame,(x,y-size),(x,y+size),col,1)

    def _slam_text(self, frame, text, t, pos, color):
        """
        Text that slams in from large→normal size,
        then fades out near end.
        """
        if t < 0.05 or t > 0.95:
            return

        # Scale: starts large, settles to normal
        if t < 0.15:
            scale = 0.5 + (1.5*(t/0.15))
        else:
            scale = 2.0

        alpha = 1.0
        if t > 0.8:
            alpha = (1.0-t)/0.2

        col = tuple(int(c*alpha) for c in color)

        lw, lh = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                  scale*0.6, 2)[0]
        tx = pos[0] - lw//2
        ty = pos[1] + lh//2

        # Shadow
        cv2.putText(frame, text, (tx+2,ty+2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale*0.6,
                    (0,0,0), 3, cv2.LINE_AA)
        # Text
        cv2.putText(frame, text, (tx,ty),
                    cv2.FONT_HERSHEY_SIMPLEX, scale*0.6,
                    col, 2, cv2.LINE_AA)

    # ----------------------------------------------------------
    def draw_hud(self, frame):
        if self.is_playing:
            from utils.overlay_utils import draw_text_with_bg
            sign = self._active_sign.replace("_"," ").upper()
            draw_text_with_bg(frame, f"ANIMATION: {sign}",
                              (10, frame.shape[0]-80),
                              font_scale=0.5,
                              text_color=(255,200,0),
                              bg_color=(0,0,0))
        return frame
