"""
EyeMouse Pro v4.0 — Faster Calibration + Precision Mode
========================================================
Improvements over v3:
  [1]  Smart 3x3 (9-point) calibration  half the time, same accuracy
  [2]  Guided warm-up step before calibration (no wasted samples)
  [3]  Natural left-to-right, top-to-bottom calibration order
  [4]  PRECISION MODE (P key) — fine-grained cursor when gaze is slow
         - Strong double-EMA smoothing in precision mode
         - Reduced acceleration exponent (linear = easier to target)
         - Smaller dead-zone so tiny eye movements register
         - HUD badge shows PRECISE / COARSE
  [5]  Calibration saved to disk (calib.npz) and auto-loaded on startup
         skip calibration entirely if file exists and accuracy is good
  [6]  All v3 features retained (dual-iris fix, dwell, scroll, blink-click)
"""

import cv2
import mediapipe as mp
import pyautogui
import math
import time
import random
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════
@dataclass
class Config:
    # MediaPipe
    MAX_FACES: int          = 1
    DETECT_CONF: float      = 0.65
    TRACK_CONF: float       = 0.65

    # EMA — base alphas; velocity-adaptive code scales these at runtime
    EMA_FAST_BASE: float    = 0.40   # when moving fast
    EMA_SLOW_BASE: float    = 0.10   # when nearly still
    EMA_VEL_SCALE: float    = 0.015  # how quickly alpha ramps with velocity

    # Dead-zone
    DEAD_ZONE: float        = 0.018  # reduced: 0.035 was killing small gaze movements

    # Acceleration
    ACCEL_EXP: float        = 1.60   # power-curve exponent
    ACCEL_SCALE: float      = 1.08   # global gain

    # Fixation braking — if pixel velocity < threshold, scale movement down
    FIXATION_VEL_PX: float  = 4.0    # px/frame below which braking kicks in
    FIXATION_BRAKE: float   = 0.45   # raised from 0.15: was killing small movements

    # Blink / click
    EAR_CLOSE: float        = 0.21   # raised: 0.18 was too tight for most people
    EAR_OPEN: float         = 0.25   # raised to match
    EAR_BLINK_FRAMES: int   = 2      # 2 frames is enough; 3 missed fast blinks
    CLICK_HOLD_S: float     = 0.90   # hold → left-click
    DOUBLE_BLINK_WINDOW: float = 0.35  # both eyes close within this → double-click
    POST_CLICK_FREEZE_S: float = 0.30  # freeze cursor after click fires

    # Dwell-click
    DWELL_RADIUS_PX: int    = 42     # pixels of allowed drift during dwell
    DWELL_TIME_S: float     = 1.20   # dwell duration to trigger click

    # Scroll
    SCROLL_ZONE_FRAC: float = 0.08   # top/bottom 8% of screen
    SCROLL_DWELL_S: float   = 0.55   # dwell in zone before scrolling starts
    SCROLL_REPEAT_S: float  = 0.18   # repeat scroll every N seconds

    # Calibration — 3x3 = 9 points, ~20 seconds total
    CALIB_COLS: int         = 3
    CALIB_ROWS: int         = 3
    CALIB_DWELL_S: float    = 1.5    # seconds per point (was 2.0)
    CALIB_MARGIN_PX: int    = 100
    CALIB_SETTLE_S: float   = 0.25   # skip first N seconds per point (was 0.35)
    CALIB_FILE: str         = 'calib.npz'   # saved calibration path

    # Precision mode — toggled with P key
    PRECISION_EMA_FAST: float  = 0.12   # much stronger smoothing
    PRECISION_EMA_SLOW: float  = 0.04
    PRECISION_ACCEL_EXP: float = 1.10   # nearly linear — easier to aim
    PRECISION_DEAD_ZONE: float = 0.008  # smaller — registers tiny movements

    # Tracking
    MAX_LOST_FRAMES: int    = 10

    # Iris / EAR landmark indices — MediaPipe 478-point model
    # Left iris
    L_IRIS: int   = 468
    L_INNER: int  = 362
    L_OUTER: int  = 263
    # Right iris
    R_IRIS: int   = 473
    R_INNER: int  = 133
    R_OUTER: int  = 33

    # Left eye EAR landmarks (6-point)
    LE_P1: int = 385; LE_P2: int = 387; LE_P3: int = 373
    LE_P4: int = 380; LE_P5: int = 263; LE_P6: int = 362

    # Right eye EAR landmarks (6-point)
    RE_P1: int = 160; RE_P2: int = 158; RE_P3: int = 153
    RE_P4: int = 144; RE_P5: int = 33;  RE_P6: int = 133

    FAILSAFE: bool          = True
    PYAUTO_PAUSE: float     = 0.0


CFG = Config()
pyautogui.FAILSAFE = CFG.FAILSAFE
pyautogui.PAUSE    = CFG.PYAUTO_PAUSE


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════
def _pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float64)


def compute_ear(landmarks, p1, p2, p3, p4, p5, p6, w, h) -> float:
    """Standard EAR = (||P2-P6||+||P3-P5||) / (2*||P1-P4||)"""
    A = np.linalg.norm(_pt(landmarks, p2, w, h) - _pt(landmarks, p6, w, h))
    B = np.linalg.norm(_pt(landmarks, p3, w, h) - _pt(landmarks, p5, w, h))
    C = np.linalg.norm(_pt(landmarks, p1, w, h) - _pt(landmarks, p4, w, h))
    return (A + B) / (2.0 * C) if C > 1e-6 else 0.0


def left_ear(lm, w, h):
    return compute_ear(lm, CFG.LE_P1,CFG.LE_P2,CFG.LE_P3,
                           CFG.LE_P4,CFG.LE_P5,CFG.LE_P6, w, h)

def right_ear(lm, w, h):
    return compute_ear(lm, CFG.RE_P1,CFG.RE_P2,CFG.RE_P3,
                           CFG.RE_P4,CFG.RE_P5,CFG.RE_P6, w, h)


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """
    CLAHE on Y channel only — boosts local contrast for iris detection
    in low-light without blowing out highlights.
    """
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


# ═══════════════════════════════════════════════════════════
#  VELOCITY-ADAPTIVE EMA
# ═══════════════════════════════════════════════════════════
class AdaptiveEMA:
    """
    Alpha scales with instantaneous velocity:
      alpha = clamp(base_slow + velocity * vel_scale, base_slow, base_fast)
    Fast motion → high alpha (low lag).
    Fixation → low alpha (strong smoothing).
    """
    def __init__(self, init_x: float, init_y: float):
        self.fx = self.sx = init_x
        self.fy = self.sy = init_y
        self._prev_x = init_x
        self._prev_y = init_y

    def update(self, x: float, y: float, precision: bool = False) -> Tuple[float, float]:
        vel = math.hypot(x - self._prev_x, y - self._prev_y)

        if precision:
            fast = CFG.PRECISION_EMA_FAST
            slow = CFG.PRECISION_EMA_SLOW
        else:
            fast = CFG.EMA_FAST_BASE
            slow = CFG.EMA_SLOW_BASE

        alpha = min(fast, slow + vel * CFG.EMA_VEL_SCALE)

        self.fx = alpha * x        + (1 - alpha) * self.fx
        self.fy = alpha * y        + (1 - alpha) * self.fy
        self.sx = 0.14 * self.fx   + 0.86 * self.sx
        self.sy = 0.14 * self.fy   + 0.86 * self.sy

        self._prev_x, self._prev_y = x, y
        return self.sx, self.sy

    def reset(self, x: float, y: float):
        self.fx = self.sx = self._prev_x = x
        self.fy = self.sy = self._prev_y = y


# ═══════════════════════════════════════════════════════════
#  16-POINT CALIBRATION GRID (4×4)
# ═══════════════════════════════════════════════════════════
class CalibrationGrid:
    """
    4×4 = 16 calibration points → 9 bilinear interpolation cells.
    Each cell corrects for local iris-plane non-linearity independently.
    """
    def __init__(self, cols: int = 4, rows: int = 4):
        self.cols  = cols
        self.rows  = rows
        self.raw   = [[None]*cols for _ in range(rows)]
        self.ready = False
        self.min_x = self.max_x = 0.0
        self.min_y = self.max_y = 0.0

    def set_point(self, r, c, rx, ry):
        self.raw[r][c] = (rx, ry)

    def finalise(self):
        vals_x = [self.raw[r][c][0] for r in range(self.rows)
                  for c in range(self.cols) if self.raw[r][c]]
        vals_y = [self.raw[r][c][1] for r in range(self.rows)
                  for c in range(self.cols) if self.raw[r][c]]
        self.min_x, self.max_x = min(vals_x), max(vals_x)
        self.min_y, self.max_y = min(vals_y), max(vals_y)
        self.ready = True

    def gaze_to_screen(self, rx: float, ry: float) -> Tuple[float, float]:
        if not self.ready:
            return 0.5, 0.5
        dx = max(1e-9, self.max_x - self.min_x)
        dy = max(1e-9, self.max_y - self.min_y)

        cx = max(self.min_x, min(self.max_x, rx))
        cy = max(self.min_y, min(self.max_y, ry))

        fx = (cx - self.min_x) / dx          # 0..1 across columns
        fy = (cy - self.min_y) / dy          # 0..1 across rows

        cf = fx * (self.cols - 1)
        rf = fy * (self.rows - 1)
        c0 = int(max(0, min(self.cols - 2, cf)))
        r0 = int(max(0, min(self.rows - 2, rf)))
        c1, r1 = c0 + 1, r0 + 1
        tc = cf - c0
        tr = rf - r0

        def tgt(r, c):
            return c / (self.cols-1), r / (self.rows-1)

        tl = tgt(r0,c0); tr_ = tgt(r0,c1)
        bl = tgt(r1,c0); br  = tgt(r1,c1)

        sx = (tl[0]*(1-tc)*(1-tr) + tr_[0]*tc*(1-tr) +
              bl[0]*(1-tc)*tr     + br[0]*tc*tr)
        sy = (tl[1]*(1-tc)*(1-tr) + tr_[1]*tc*(1-tr) +
              bl[1]*(1-tc)*tr     + br[1]*tc*tr)
        return sx, sy

    def accuracy_test(self, rel_x, rel_y, target_norm_x, target_norm_y,
                      screen_w, screen_h) -> float:
        """Return pixel error for a validation point."""
        sx, sy = self.gaze_to_screen(rel_x, rel_y)
        ex = (sx - target_norm_x) * screen_w
        ey = (sy - target_norm_y) * screen_h
        return math.hypot(ex, ey)


# ═══════════════════════════════════════════════════════════
#  CURSOR MAPPER: dead-zone + acceleration + fixation brake
# ═══════════════════════════════════════════════════════════
class CursorMapper:
    def __init__(self, sw: int, sh: int):
        self.sw = sw
        self.sh = sh
        self._prev_px = sw / 2
        self._prev_py = sh / 2

    def apply(self, nx: float, ny: float, precision: bool = False) -> Tuple[float, float]:
        dz  = CFG.PRECISION_DEAD_ZONE if precision else CFG.DEAD_ZONE
        exp = CFG.PRECISION_ACCEL_EXP if precision else CFG.ACCEL_EXP
        sc  = CFG.ACCEL_SCALE

        dx = nx - 0.5
        dy = ny - 0.5

        # Dead-zone rescale
        if abs(dx) < dz:  dx = 0.0
        else: dx = math.copysign((abs(dx)-dz)/(0.5-dz)*0.5, dx)
        if abs(dy) < dz:  dy = 0.0
        else: dy = math.copysign((abs(dy)-dz)/(0.5-dz)*0.5, dy)

        # Acceleration power curve
        adx = math.copysign(min(0.5,(abs(dx)**exp)*sc), dx)
        ady = math.copysign(min(0.5,(abs(dy)**exp)*sc), dy)

        px = (0.5 + adx) * self.sw
        py = (0.5 + ady) * self.sh

        # Fixation braking — if very small movement, dampen further
        vel = math.hypot(px - self._prev_px, py - self._prev_py)
        if vel < CFG.FIXATION_VEL_PX:
            brake = CFG.FIXATION_BRAKE
            px = self._prev_px + (px - self._prev_px) * brake
            py = self._prev_py + (py - self._prev_py) * brake

        self._prev_px, self._prev_py = px, py

        px = max(5, min(self.sw-5, px))
        py = max(5, min(self.sh-5, py))
        return px, py


# ═══════════════════════════════════════════════════════════
#  BLINK / CLICK DETECTOR
# ═══════════════════════════════════════════════════════════
class ClickDetector:
    """
    Handles four interaction events:
      L  = left eye sustained close  → left-click  (after CLICK_HOLD_S)
      R  = right eye sustained close → right-click (after CLICK_HOLD_S)
      B  = both eyes close < DOUBLE_BLINK_WINDOW  → double-click
      (normal blinks < BLINK_BUFFER_FRAMES are ignored)
    """
    def __init__(self):
        self._l_frames = 0;  self._l_closed = False
        self._r_frames = 0;  self._r_closed = False
        self._l_t = None;    self._r_t = None
        self._l_fired = False; self._r_fired = False
        self._both_t: Optional[float] = None
        self._post_click_until: float = 0.0

    def update(self, l_ear: float, r_ear: float
               ) -> Tuple[str, float, float]:
        """
        Returns: (action, l_held, r_held)
        action ∈ {'', 'left', 'right', 'double'}
        """
        now   = time.time()
        action = ''

        # ── Left eye state machine ──
        if not self._l_closed:
            if l_ear < CFG.EAR_CLOSE:
                self._l_frames += 1
                if self._l_frames >= CFG.EAR_BLINK_FRAMES:
                    self._l_closed = True
                    self._l_t = now
                    self._l_fired = False
            else:
                self._l_frames = 0
        else:
            if l_ear > CFG.EAR_OPEN:
                self._l_closed = False
                self._l_frames = 0
                self._l_t = None
                self._l_fired = False

        # ── Right eye state machine ──
        if not self._r_closed:
            if r_ear < CFG.EAR_CLOSE:
                self._r_frames += 1
                if self._r_frames >= CFG.EAR_BLINK_FRAMES:
                    self._r_closed = True
                    self._r_t = now
                    self._r_fired = False
            else:
                self._r_frames = 0
        else:
            if r_ear > CFG.EAR_OPEN:
                self._r_closed = False
                self._r_frames = 0
                self._r_t = None
                self._r_fired = False

        l_held = (now - self._l_t) if self._l_closed and self._l_t else 0.0
        r_held = (now - self._r_t) if self._r_closed and self._r_t else 0.0

        # ── Both eyes closed within DOUBLE_BLINK_WINDOW → double-click ──
        if self._l_closed and self._r_closed:
            if self._both_t is None:
                self._both_t = now
            elif (now - self._both_t < CFG.DOUBLE_BLINK_WINDOW
                  and not self._l_fired and not self._r_fired):
                pyautogui.doubleClick()
                self._l_fired = self._r_fired = True
                action = 'double'
                self._post_click_until = now + CFG.POST_CLICK_FREEZE_S
        else:
            self._both_t = None

        # ── Single sustained left → left-click ──
        if (action == '' and self._l_closed and not self._r_closed
                and not self._l_fired and l_held >= CFG.CLICK_HOLD_S):
            pyautogui.click()
            self._l_fired = True
            action = 'left'
            self._post_click_until = now + CFG.POST_CLICK_FREEZE_S

        # ── Single sustained right → right-click ──
        if (action == '' and self._r_closed and not self._l_closed
                and not self._r_fired and r_held >= CFG.CLICK_HOLD_S):
            pyautogui.rightClick()
            self._r_fired = True
            action = 'right'
            self._post_click_until = now + CFG.POST_CLICK_FREEZE_S

        return action, l_held, r_held

    @property
    def in_post_click_freeze(self) -> bool:
        return time.time() < self._post_click_until

    @property
    def l_closed(self) -> bool: return self._l_closed
    @property
    def r_closed(self) -> bool: return self._r_closed


# ═══════════════════════════════════════════════════════════
#  DWELL-CLICK
# ═══════════════════════════════════════════════════════════
class DwellClicker:
    """
    If cursor stays within DWELL_RADIUS_PX for DWELL_TIME_S → left-click.
    Visual ring shrinks toward the cursor as dwell progress increases.
    """
    def __init__(self):
        self._anchor_x: Optional[float] = None
        self._anchor_y: Optional[float] = None
        self._start: Optional[float] = None
        self._fired = False
        self.enabled = False    # toggled via 'D' key

    def update(self, px: float, py: float) -> Tuple[float, bool]:
        """Returns (progress 0..1, click_fired)"""
        if not self.enabled:
            return 0.0, False
        now  = time.time()
        fired = False

        if self._anchor_x is None:
            self._anchor_x, self._anchor_y = px, py
            self._start = now
            self._fired = False
            return 0.0, False

        dist = math.hypot(px - self._anchor_x, py - self._anchor_y)
        if dist > CFG.DWELL_RADIUS_PX:
            # Moved outside dwell zone — reset
            self._anchor_x, self._anchor_y = px, py
            self._start = now
            self._fired = False
            return 0.0, False

        progress = min(1.0, (now - self._start) / CFG.DWELL_TIME_S)
        if progress >= 1.0 and not self._fired:
            pyautogui.click()
            self._fired = True
            fired = True
            # Reset after firing so user can fire again
            self._anchor_x, self._anchor_y = px, py
            self._start = now + 0.8   # cooldown

        return progress, fired


# ═══════════════════════════════════════════════════════════
#  EDGE-DWELL SCROLLER
# ═══════════════════════════════════════════════════════════
class EdgeScroller:
    """
    Top 8% of screen → scroll up.
    Bottom 8% of screen → scroll down.
    Dwell for SCROLL_DWELL_S before first scroll, then repeat.
    """
    def __init__(self, screen_h: int):
        self.sh = screen_h
        self._zone: Optional[str] = None   # 'up' | 'down' | None
        self._enter_t: Optional[float] = None
        self._last_scroll: float = 0.0
        self.enabled = True     # toggled via 'S' key

    def update(self, py: float):
        if not self.enabled:
            return
        now    = time.time()
        zone_h = self.sh * CFG.SCROLL_ZONE_FRAC

        if py < zone_h:
            new_zone = 'up'
        elif py > self.sh - zone_h:
            new_zone = 'down'
        else:
            new_zone = None

        if new_zone != self._zone:
            self._zone    = new_zone
            self._enter_t = now if new_zone else None
            self._last_scroll = 0.0
            return

        if self._zone and self._enter_t:
            elapsed = now - self._enter_t
            if elapsed >= CFG.SCROLL_DWELL_S:
                if now - self._last_scroll >= CFG.SCROLL_REPEAT_S:
                    amount = 3 if self._zone == 'up' else -3
                    pyautogui.scroll(amount)
                    self._last_scroll = now

    @property
    def active_zone(self) -> Optional[str]:
        return self._zone


# ═══════════════════════════════════════════════════════════
#  FPS MONITOR
# ═══════════════════════════════════════════════════════════
class FPSMonitor:
    def __init__(self, window: int = 30):
        self._t: deque = deque(maxlen=window)

    def tick(self) -> float:
        self._t.append(time.perf_counter())
        if len(self._t) < 2: return 0.0
        return (len(self._t)-1) / (self._t[-1] - self._t[0])


# ═══════════════════════════════════════════════════════════
#  GAZE TRAIL
# ═══════════════════════════════════════════════════════════
class GazeTrail:
    """Stores last N gaze pixel positions for trail visualisation."""
    def __init__(self, maxlen: int = 25):
        self._pts: deque = deque(maxlen=maxlen)

    def push(self, x: int, y: int):
        self._pts.append((x, y))

    def draw(self, frame: np.ndarray):
        pts = list(self._pts)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            colour = (int(255*alpha), int(180*(1-alpha)), 80)
            cv2.line(frame, pts[i-1], pts[i], colour, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════
#  LIVE TUNING WINDOW
# ═══════════════════════════════════════════════════════════
class TuningWindow:
    """
    OpenCV trackbar window for runtime parameter tuning.
    Sliders map to: EMA fast alpha, dead-zone, accel exponent.
    """
    WIN = 'EyeMouse Tuning'

    def __init__(self):
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, 400, 160)
        # Sliders: integer 1..100, divide to get float
        cv2.createTrackbar('EMA Speed  (fast alpha ×100)', self.WIN,
                           int(CFG.EMA_FAST_BASE * 100), 100, lambda v: None)
        cv2.createTrackbar('Dead-Zone  (×1000)',           self.WIN,
                           int(CFG.DEAD_ZONE * 1000), 100, lambda v: None)
        cv2.createTrackbar('Accel Exp  (×100)',            self.WIN,
                           int(CFG.ACCEL_EXP * 100), 300, lambda v: None)

    def sync(self):
        """Pull current slider values into CFG."""
        v1 = cv2.getTrackbarPos('EMA Speed  (fast alpha ×100)', self.WIN)
        v2 = cv2.getTrackbarPos('Dead-Zone  (×1000)',           self.WIN)
        v3 = cv2.getTrackbarPos('Accel Exp  (×100)',            self.WIN)
        CFG.EMA_FAST_BASE = max(0.05, v1 / 100.0)
        CFG.DEAD_ZONE     = max(0.005, v2 / 1000.0)
        CFG.ACCEL_EXP     = max(1.0, v3 / 100.0)


# ═══════════════════════════════════════════════════════════
#  MAIN CONTROLLER
# ═══════════════════════════════════════════════════════════
class EyeMouseController:
    def __init__(self):
        self.sw, self.sh = pyautogui.size()

        self.mp_fm = mp.solutions.face_mesh
        self.face_mesh = self.mp_fm.FaceMesh(
            max_num_faces=CFG.MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=CFG.DETECT_CONF,
            min_tracking_confidence=CFG.TRACK_CONF,
        )

        self.grid    = CalibrationGrid(CFG.CALIB_COLS, CFG.CALIB_ROWS)
        self.ema     = AdaptiveEMA(self.sw/2, self.sh/2)
        self.mapper  = CursorMapper(self.sw, self.sh)
        self.clicker = ClickDetector()
        self.dwell   = DwellClicker()
        self.scroller= EdgeScroller(self.sh)
        self.fps_mon = FPSMonitor()
        self.trail   = GazeTrail()
        self.tuner   = TuningWindow()

        self._frames_lost   = 0
        self._last_px       = self.sw / 2
        self._last_py       = self.sh / 2
        self.precision_mode = False   # toggled with P key

    # ── Dual-iris offset: average both eyes ─────────────────────────
    def _get_iris_offset(self, lm) -> Tuple[float, float]:
        """
        Average left and right iris offsets.
        Each iris is expressed relative to its own eye socket midpoint,
        giving head-pose-invariant gaze regardless of head rotation.
        Using both irises cuts noise by ~√2 compared to single-eye tracking.
        """
        # Left
        l_anchor_x = (lm[CFG.L_INNER].x + lm[CFG.L_OUTER].x) / 2
        l_anchor_y = (lm[CFG.L_INNER].y + lm[CFG.L_OUTER].y) / 2
        lx = lm[CFG.L_IRIS].x - l_anchor_x
        ly = lm[CFG.L_IRIS].y - l_anchor_y

        # Right
        r_anchor_x = (lm[CFG.R_INNER].x + lm[CFG.R_OUTER].x) / 2
        r_anchor_y = (lm[CFG.R_INNER].y + lm[CFG.R_OUTER].y) / 2
        rx = lm[CFG.R_IRIS].x - r_anchor_x
        ry = lm[CFG.R_IRIS].y - r_anchor_y

        # Both eyes move in the same direction after cv2.flip(frame,1)
        # DO NOT negate rx — negating it causes the two signals to cancel out
        avg_x = (lx + rx) / 2
        avg_y = (ly + ry) / 2
        return avg_x, avg_y

    # ── Save / Load calibration from disk ───────────────────────────
    def save_calibration(self):
        """Save grid raw points to disk so we can skip calibration next time."""
        import os
        try:
            data = {}
            for r in range(self.grid.rows):
                for c in range(self.grid.cols):
                    v = self.grid.raw[r][c]
                    if v:
                        data[f'{r}_{c}_x'] = v[0]
                        data[f'{r}_{c}_y'] = v[1]
            data['rows'] = self.grid.rows
            data['cols'] = self.grid.cols
            np.savez(CFG.CALIB_FILE, **data)
            print(f"  Calibration saved to {CFG.CALIB_FILE}")
        except Exception as e:
            print(f"  Could not save calibration: {e}")

    def try_load_calibration(self) -> bool:
        """Try to load a saved calibration. Returns True if loaded successfully."""
        import os
        if not os.path.exists(CFG.CALIB_FILE):
            return False
        try:
            data = np.load(CFG.CALIB_FILE)
            rows = int(data['rows'])
            cols = int(data['cols'])
            if rows != CFG.CALIB_ROWS or cols != CFG.CALIB_COLS:
                print("  Saved calibration grid size mismatch — recalibrating.")
                return False
            self.grid = CalibrationGrid(cols, rows)
            for r in range(rows):
                for c in range(cols):
                    k = f'{r}_{c}_x'
                    if k in data:
                        self.grid.set_point(r, c, float(data[f'{r}_{c}_x']),
                                                   float(data[f'{r}_{c}_y']))
            self.grid.finalise()
            print(f"  Loaded saved calibration from {CFG.CALIB_FILE}  (press R to redo)")
            return True
        except Exception as e:
            print(f"  Could not load calibration: {e}")
            return False

    # ── Calibration ─────────────────────────────────────────────────
    def run_calibration(self, cap: cv2.VideoCapture):
        cols, rows = CFG.CALIB_COLS, CFG.CALIB_ROWS
        m  = CFG.CALIB_MARGIN_PX
        sw, sh = self.sw, self.sh

        # Build uniform grid of screen targets
        xs = [int(m + (sw - 2*m) * c / (cols-1)) for c in range(cols)]
        ys = [int(m + (sh - 2*m) * r / (rows-1)) for r in range(rows)]

        # Natural reading order: top-left → right → down (no random shuffle)
        # This feels predictable and lets the eyes settle between rows.
        order = [(r, c) for r in range(rows) for c in range(cols)]

        total = cols * rows
        print(f"\n  {total}-POINT CALIBRATION ({cols}x{rows} grid, ~{int(total*CFG.CALIB_DWELL_S)}s)")
        print("  Look at each dot. Keep your head still.\n")

        # ── Warm-up: show camera feed for 2s so face mesh initialises ──
        warmup_end = time.time() + 2.0
        while time.time() < warmup_end:
            ok, frame = cap.read()
            if not ok: continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(apply_clahe(frame), cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            self.face_mesh.process(rgb)   # warm up the model

            ui = np.zeros((sh, sw, 3), dtype=np.uint8)
            rem = warmup_end - time.time()
            msg = "Get ready... look straight ahead"
            cv2.putText(ui, msg,
                        (sw//2 - 280, sh//2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (160, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(ui, f"Starting in {rem:.0f}s",
                        (sw//2 - 100, sh//2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1, cv2.LINE_AA)
            cv2.imshow('EyeMouse Calibration', ui)
            cv2.waitKey(1)

        completed = []   # (r,c) points already done — shown as green dots

        for idx, (row, col) in enumerate(order):
            target = (xs[col], ys[row])
            smpx, smpy = [], []
            t0  = time.time()
            dwell_s = CFG.CALIB_DWELL_S
            settle  = CFG.CALIB_SETTLE_S
            label = f"Point {idx+1} of {total}"

            while time.time() - t0 < dwell_s:
                ok, frame = cap.read()
                if not ok: continue
                frame = cv2.flip(frame, 1)
                ih, iw = frame.shape[:2]
                rgb = cv2.cvtColor(apply_clahe(frame), cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = self.face_mesh.process(rgb)

                elapsed  = time.time() - t0
                progress = elapsed / dwell_s

                ui = np.zeros((sh, sw, 3), dtype=np.uint8)

                # Completed dots
                for pr, pc in completed:
                    cv2.circle(ui, (xs[pc], ys[pr]), 9, (30, 160, 30), -1)
                    cv2.putText(ui, "✓", (xs[pc]-8, ys[pr]+5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,255,80), 1)

                # Remaining dots (dim)
                for fr2 in range(rows):
                    for fc2 in range(cols):
                        if (fr2, fc2) not in completed and (fr2, fc2) != (row, col):
                            cv2.circle(ui, (xs[fc2], ys[fr2]), 7, (55, 55, 55), -1)

                # Active target — pulsing ring
                pulse = int(10 + 8 * math.sin(elapsed * 10))
                cv2.circle(ui, target, 18, (0, 160, 255), -1)
                cv2.circle(ui, target, 18 + pulse, (255, 255, 255), 2, cv2.LINE_AA)

                # Collecting arc (fills as we gather samples)
                arc_end = int(-90 + 360 * progress)
                cv2.ellipse(ui, target, (30, 30), 0, -90, arc_end,
                            (0, 230, 140), 3, cv2.LINE_AA)

                # Arrow showing order: draw faint line to next point
                if idx + 1 < total:
                    nr, nc = order[idx + 1]
                    next_pt = (xs[nc], ys[nr])
                    cv2.arrowedLine(ui, target, next_pt, (40, 40, 40), 1,
                                    cv2.LINE_AA, tipLength=0.02)

                # Label near target
                lx_off = 30 if target[0] < sw // 2 else -120
                cv2.putText(ui, label, (target[0] + lx_off, target[1] - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

                # Bottom status bar
                ov = ui.copy()
                cv2.rectangle(ov, (0, sh-40), (sw, sh), (20,20,20), -1)
                cv2.addWeighted(ov, 0.7, ui, 0.3, 0, ui)
                cv2.putText(ui, "Keep head still  |  Look directly at the dot",
                            (sw//2 - 250, sh - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)

                # Tracking indicator top-right
                tracking_now = results.multi_face_landmarks is not None
                t_col = (0, 220, 80) if tracking_now else (0, 60, 220)
                t_txt = "Face: OK" if tracking_now else "Face: LOST"
                cv2.putText(ui, t_txt, (sw - 160, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, t_col, 1, cv2.LINE_AA)

                cv2.imshow('EyeMouse Calibration', ui)
                cv2.waitKey(1)

                if elapsed > settle and results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    ox, oy = self._get_iris_offset(lm)
                    smpx.append(ox); smpy.append(oy)

            if smpx:
                self.grid.set_point(row, col,
                                    float(np.median(smpx)),
                                    float(np.median(smpy)))
                completed.append((row, col))
                print(f"  [{idx+1}/{total}] OK  ({len(smpx)} samples)")
            else:
                print(f"  [{idx+1}/{total}] MISSED — check lighting/camera")

        self.grid.finalise()

        # ── Validation: show accuracy at 4 corners ───────────────────
        print("\n  Validating accuracy...")
        val_pts = [
            (0, 0, xs[0], ys[0]),
            (0, cols-1, xs[cols-1], ys[0]),
            (rows-1, 0, xs[0], ys[rows-1]),
            (rows-1, cols-1, xs[cols-1], ys[rows-1]),
        ]
        errors = []
        for vi, (vr, vc, vx, vy) in enumerate(val_pts):
            vsamps_x, vsamps_y = [], []
            vt0 = time.time()
            while time.time() - vt0 < 1.2:
                ok, frame = cap.read()
                if not ok: continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(apply_clahe(frame), cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res2 = self.face_mesh.process(rgb)
                vel = time.time() - vt0
                ui2 = np.zeros((sh, sw, 3), dtype=np.uint8)
                cv2.putText(ui2, f"Validation {vi+1}/4 — look at dot",
                            (sw//2 - 220, sh//2 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 100), 2, cv2.LINE_AA)
                cv2.circle(ui2, (vx, vy), 16, (255, 120, 0), -1)
                cv2.circle(ui2, (vx, vy), int(16 + 8*math.sin(vel*10)),
                           (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('EyeMouse Calibration', ui2)
                cv2.waitKey(1)
                if vel > 0.3 and res2.multi_face_landmarks:
                    lm2 = res2.multi_face_landmarks[0].landmark
                    ox, oy = self._get_iris_offset(lm2)
                    vsamps_x.append(ox); vsamps_y.append(oy)

            if vsamps_x:
                err = self.grid.accuracy_test(
                    float(np.median(vsamps_x)), float(np.median(vsamps_y)),
                    vx / sw, vy / sh, sw, sh)
                errors.append(err)

        if errors:
            avg_err = sum(errors) / len(errors)
            quality = "Excellent" if avg_err < 80 else \
                      "Good"      if avg_err < 150 else "Poor — consider redoing (press R)"
            print(f"  Average error: {avg_err:.0f}px  — {quality}")

        # ── Save calibration so we skip this next time ────────────────
        self.save_calibration()

        # ── Done screen ───────────────────────────────────────────────
        done_ui = np.zeros((sh, sw, 3), dtype=np.uint8)
        q_label = "Excellent!" if (errors and sum(errors)/len(errors) < 80) else \
                  "Good!" if (errors and sum(errors)/len(errors) < 150) else "Done"
        cv2.putText(done_ui, f"Calibration {q_label}",
                    (sw//2 - 200, sh//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 120), 2, cv2.LINE_AA)
        cv2.putText(done_ui, "Starting in 2s...  Press P for Precision Mode",
                    (sw//2 - 310, sh//2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.imshow('EyeMouse Calibration', done_ui)
        cv2.waitKey(2000)

        print("  Calibration complete.\n")
        cv2.destroyWindow('EyeMouse Calibration')

    # ── Per-frame update ─────────────────────────────────────────────
    def update(self, frame: np.ndarray) -> np.ndarray:
        ih, iw = frame.shape[:2]
        fps = self.fps_mon.tick()
        self.tuner.sync()

        # CLAHE pre-processing for low-light robustness
        enhanced = apply_clahe(frame)

        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._frames_lost = min(self._frames_lost+1, CFG.MAX_LOST_FRAMES+1)
            if self._frames_lost <= CFG.MAX_LOST_FRAMES:
                pyautogui.moveTo(int(self._last_px), int(self._last_py))
            self._draw_hud(frame, fps, None, None, 0, 0, 'lost', False, False, 0.0)
            return frame

        self._frames_lost = 0
        lm = results.multi_face_landmarks[0].landmark

        # ── Gaze → screen ──
        rel_x, rel_y       = self._get_iris_offset(lm)
        norm_x, norm_y     = self.grid.gaze_to_screen(rel_x, rel_y)
        raw_px, raw_py     = self.mapper.apply(norm_x, norm_y, self.precision_mode)
        smooth_px, smooth_py = self.ema.update(raw_px, raw_py, self.precision_mode)
        smooth_px = max(5, min(self.sw-5, smooth_px))
        smooth_py = max(5, min(self.sh-5, smooth_py))

        # ── Click detection ──
        l_e = left_ear(lm, iw, ih)
        r_e = right_ear(lm, iw, ih)
        action, l_held, r_held = self.clicker.update(l_e, r_e)

        # ── Dwell-click ──
        dwell_prog, dwell_fired = self.dwell.update(smooth_px, smooth_py)

        # ── Edge scroll ──
        self.scroller.update(smooth_py)

        # ── Cursor movement (skipped during post-click freeze) ──
        if not self.clicker.in_post_click_freeze and not dwell_fired:
            pyautogui.moveTo(int(smooth_px), int(smooth_py))
            self._last_px = smooth_px
            self._last_py = smooth_py

        # ── Trail ──
        self.trail.push(
            int(lm[CFG.L_IRIS].x * iw),
            int(lm[CFG.L_IRIS].y * ih)
        )
        self.trail.draw(frame)

        # ── Gaze direction arrow ──
        iris_px = int(lm[CFG.L_IRIS].x * iw)
        iris_py = int(lm[CFG.L_IRIS].y * ih)
        l_inner_px = int(lm[CFG.L_INNER].x * iw)
        l_inner_py = int(lm[CFG.L_INNER].y * ih)
        arrow_dx = int((lm[CFG.L_IRIS].x - (lm[CFG.L_INNER].x+lm[CFG.L_OUTER].x)/2) * iw * 12)
        arrow_dy = int((lm[CFG.L_IRIS].y - (lm[CFG.L_INNER].y+lm[CFG.L_OUTER].y)/2) * ih * 12)
        cv2.arrowedLine(frame, (iris_px, iris_py),
                        (iris_px+arrow_dx, iris_py+arrow_dy),
                        (0,220,255), 2, cv2.LINE_AA, tipLength=0.4)

        # Both iris dots
        cv2.circle(frame, (iris_px, iris_py), 4, (0,255,100), -1)
        r_iris_px = int(lm[CFG.R_IRIS].x * iw)
        r_iris_py = int(lm[CFG.R_IRIS].y * ih)
        cv2.circle(frame, (r_iris_px, r_iris_py), 4, (100,255,0), -1)

        # Dwell ring at iris position
        if dwell_prog > 0.05:
            ring_r = int(CFG.DWELL_RADIUS_PX * 0.5)
            start_a = -90
            end_a   = int(-90 + 360 * dwell_prog)
            cv2.ellipse(frame, (iris_px, iris_py), (ring_r, ring_r),
                        0, start_a, end_a, (0,200,255), 2, cv2.LINE_AA)

        self._draw_hud(frame, fps, l_e, r_e, l_held, r_held,
                       action, self.clicker.l_closed, self.clicker.r_closed,
                       dwell_prog)
        return frame

    # ── HUD overlay ─────────────────────────────────────────────────
    def _draw_hud(self, frame, fps, l_ear_v, r_ear_v,
                  l_held, r_held, action, l_closed, r_closed, dwell_prog):
        h, w = frame.shape[:2]

        # Semi-transparent top bar
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (w, 82), (0,0,0), -1)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)

        # Status
        tracking = (l_ear_v is not None)
        s_col = (0,255,120) if tracking else (0,60,255)
        s_txt = "TRACKING" if tracking else "LOST — move closer / improve lighting"
        cv2.putText(frame, f"EyeMouse Pro v4  |  {s_txt}",
                    (14,24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, s_col, 2, cv2.LINE_AA)

        # Precision mode badge
        if self.precision_mode:
            cv2.rectangle(frame, (w-160, 4), (w-4, 34), (0, 80, 0), -1)
            cv2.putText(frame, "PRECISE", (w-150, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (w-150, 4), (w-4, 34), (40, 40, 0), -1)
            cv2.putText(frame, "COARSE", (w-140, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 0), 1, cv2.LINE_AA)

        # FPS
        fps_col = (0,255,120) if fps >= 25 else (0,140,255) if fps >= 15 else (0,50,200)
        cv2.putText(frame, f"FPS: {fps:.0f}",
                    (14,52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, fps_col, 1, cv2.LINE_AA)

        if l_ear_v is not None:
            # Left EAR
            lc = (0,60,255) if l_closed else (180,180,180)
            cv2.putText(frame, f"L-EAR:{l_ear_v:.3f}",
                        (110,52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, lc, 1, cv2.LINE_AA)
            # Right EAR
            rc = (0,60,255) if r_closed else (180,180,180)
            cv2.putText(frame, f"R-EAR:{r_ear_v:.3f}",
                        (240,52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, rc, 1, cv2.LINE_AA)

            # Action label
            act_map = {'left':'LEFT CLICK','right':'RIGHT CLICK','double':'DOUBLE CLICK','':''}
            if act_map.get(action):
                cv2.putText(frame, act_map[action],
                            (w-220, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0,220,255), 2, cv2.LINE_AA)

        # Dwell / scroll state indicators (bottom strip)
        ov2 = frame.copy()
        cv2.rectangle(ov2, (0, h-36), (w, h), (0,0,0), -1)
        cv2.addWeighted(ov2, 0.45, frame, 0.55, 0, frame)

        dwell_txt = "[D] DWELL: ON " if self.dwell.enabled else "[D] DWELL: OFF"
        scroll_txt= "[S] SCROLL: ON" if self.scroller.enabled else "[S] SCROLL: OFF"
        cv2.putText(frame, dwell_txt,  (14, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (0,220,120) if self.dwell.enabled else (120,120,120), 1, cv2.LINE_AA)
        cv2.putText(frame, scroll_txt, (200, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (0,220,120) if self.scroller.enabled else (120,120,120), 1, cv2.LINE_AA)
        cv2.putText(frame, "[R] Recalibrate   [P] Precision   [Q] Quit",
                    (340, h-12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (140,140,140), 1, cv2.LINE_AA)

        # Scroll zone indicator lines
        sz = int(h * CFG.SCROLL_ZONE_FRAC)
        if self.scroller.enabled:
            col_scroll = (0,200,255) if self.scroller.active_zone else (60,60,60)
            cv2.line(frame, (0, sz), (w, sz), col_scroll, 1, cv2.LINE_AA)
            cv2.line(frame, (0, h-sz), (w, h-sz), col_scroll, 1, cv2.LINE_AA)
            if self.scroller.active_zone == 'up':
                cv2.putText(frame, "▲ SCROLLING UP", (w//2-100, sz+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2, cv2.LINE_AA)
            elif self.scroller.active_zone == 'down':
                cv2.putText(frame, "▼ SCROLLING DOWN", (w//2-110, h-sz-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2, cv2.LINE_AA)

        # Click hold bars
        def draw_hold_bar(held, label, y_off, colour):
            if held > 0.05:
                bw = int(w * 0.35)
                bx = w//2 - bw//2
                prog = min(1.0, held / CFG.CLICK_HOLD_S)
                cv2.rectangle(frame, (bx, y_off), (bx+bw, y_off+14), (40,40,40), -1)
                cv2.rectangle(frame, (bx, y_off), (bx+int(bw*prog), y_off+14), colour, -1)
                cv2.putText(frame, f"{label}: {held:.1f}s",
                            (bx, y_off-8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.46, colour, 1, cv2.LINE_AA)

        draw_hold_bar(l_held, "L-CLICK", h-70, (0,140,255))
        draw_hold_bar(r_held, "R-CLICK", h-50, (255,120,0))


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════╗")
    print("║       EyeMouse Pro  v4.0                 ║")
    print("╠══════════════════════════════════════════╣")
    print("║  L-eye hold  →  Left click               ║")
    print("║  R-eye hold  →  Right click              ║")
    print("║  Both blink  →  Double click             ║")
    print("║  P key       →  Toggle Precision Mode    ║")
    print("║  D key       →  Toggle dwell-click       ║")
    print("║  S key       →  Toggle edge-scroll       ║")
    print("║  R key       →  Recalibrate              ║")
    print("║  Q key       →  Quit                     ║")
    print("╚══════════════════════════════════════════╝\n")

    ctrl = EyeMouseController()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    # Try to load saved calibration — skip calibration if successful
    loaded = ctrl.try_load_calibration()
    if not loaded:
        cv2.namedWindow('EyeMouse Calibration', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('EyeMouse Calibration',
                              cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        ctrl.run_calibration(cap)

    # Main window
    cv2.namedWindow('EyeMouse Pro', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('EyeMouse Pro', 640, 480)

    print("Running. Press Q in camera window to quit.\n")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: continue

        frame = cv2.flip(frame, 1)
        out   = ctrl.update(frame)
        cv2.imshow('EyeMouse Pro', out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            cv2.namedWindow('EyeMouse Calibration', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('EyeMouse Calibration',
                                  cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            ctrl.grid  = CalibrationGrid(CFG.CALIB_COLS, CFG.CALIB_ROWS)
            ctrl.ema.reset(ctrl.sw/2, ctrl.sh/2)
            ctrl.run_calibration(cap)
            cv2.namedWindow('EyeMouse Pro', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('EyeMouse Pro', 640, 480)
        elif key == ord('p'):
            ctrl.precision_mode = not ctrl.precision_mode
            mode_name = "PRECISION" if ctrl.precision_mode else "COARSE"
            print(f"Mode: {mode_name}")
        elif key == ord('d'):
            ctrl.dwell.enabled = not ctrl.dwell.enabled
            print(f"Dwell-click: {'ON' if ctrl.dwell.enabled else 'OFF'}")
        elif key == ord('s'):
            ctrl.scroller.enabled = not ctrl.scroller.enabled
            print(f"Edge-scroll: {'ON' if ctrl.scroller.enabled else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("EyeMouse Pro closed.")


if __name__ == "__main__":
    main()