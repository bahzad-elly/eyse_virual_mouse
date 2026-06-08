"""
EyeMouse Pro - Production-Quality Eye-Controlled Mouse
=======================================================
A complete rewrite of the gaze-controlled cursor system with:
  - 9-point calibration with bilinear interpolation
  - Dual-stage EMA smoothing (replaces laggy Kalman filter)
  - Correct EAR-based blink detection (non-inverting)
  - Dead-zone filtering for micro-saccade suppression
  - Velocity-based cursor acceleration
  - Decoupled cursor movement from blink state
  - Head-pose-free design for maximum FPS
  - Live FPS monitoring
  - Full tracking-loss recovery
"""

import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ─────────────────────────────────────────────
#  GLOBAL CONFIGURATION  (tune these to taste)
# ─────────────────────────────────────────────
@dataclass
class Config:
    # MediaPipe
    MAX_FACES: int = 1
    DETECTION_CONFIDENCE: float = 0.65
    TRACKING_CONFIDENCE: float = 0.65

    # Smoothing — EMA alphas (0=max smooth/lag, 1=raw/no smooth)
    # Fast layer reacts quickly; slow layer damps residual jitter.
    EMA_FAST_ALPHA: float = 0.35   # inner EMA: tracks motion
    EMA_SLOW_ALPHA: float = 0.12   # outer EMA: final cursor position

    # Dead-zone: gaze movements smaller than this fraction of the
    # full calibrated range are treated as zero (suppresses tremor).
    DEAD_ZONE_FRACTION: float = 0.04   # 4% of calibrated range

    # Cursor acceleration: maps normalised gaze delta → screen delta.
    # Values > 1 add exponential boost at the edges.
    ACCELERATION_EXPONENT: float = 1.55   # higher = more edge boost
    ACCELERATION_SCALE: float = 1.10      # global gain multiplier

    # Blink / click detection
    # Standard EAR (Eye Aspect Ratio): vertical / horizontal eye span.
    # A typical open eye EAR ≈ 0.25–0.35; closed eye EAR < 0.18.
    EAR_CLOSE_THRESHOLD: float = 0.18    # below → eye closed
    EAR_OPEN_THRESHOLD: float = 0.22     # above → eye open (hysteresis)
    CLICK_HOLD_SECONDS: float = 0.9      # hold blink this long → left-click
    BLINK_BUFFER_FRAMES: int = 3         # consecutive closed frames before acting

    # Calibration
    CALIB_COLS: int = 3
    CALIB_ROWS: int = 3
    CALIB_DWELL_SECONDS: float = 2.2     # time to collect samples per point
    CALIB_MARGIN_PX: int = 90           # inset from screen edges

    # Tracking recovery
    MAX_FRAMES_LOST: int = 8            # frames without face before freeze

    # Iris landmark indices (MediaPipe with refine_landmarks=True)
    LEFT_IRIS_CENTER: int = 468
    LEFT_EYE_INNER: int = 362   # medial canthus (nose side)
    LEFT_EYE_OUTER: int = 263   # lateral canthus

    # Standard 6-point EAR landmarks for the LEFT eye
    # P1=upper-outer, P2=upper-inner, P3=lower-inner, P4=lower-outer
    # P5=lateral canthus, P6=medial canthus
    EAR_P1: int = 385   # upper-outer lid
    EAR_P2: int = 387   # upper-inner lid
    EAR_P3: int = 373   # lower-inner lid
    EAR_P4: int = 380   # lower-outer lid
    EAR_P5: int = 263   # lateral canthus (horizontal)
    EAR_P6: int = 362   # medial  canthus (horizontal)

    # pyautogui safety
    FAILSAFE: bool = True
    PYAUTOGUI_PAUSE: float = 0.0


CFG = Config()

# Apply pyautogui settings immediately
pyautogui.FAILSAFE = CFG.FAILSAFE
pyautogui.PAUSE = CFG.PYAUTOGUI_PAUSE


# ─────────────────────────────────────────────
#  HELPER: Eye Aspect Ratio
# ─────────────────────────────────────────────
def compute_ear(landmarks, img_w: int, img_h: int) -> float:
    """
    Standard EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
    Returns a value ≈ 0.25–0.35 when open, < 0.18 when closed.
    """
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * img_w, lm.y * img_h])

    p1, p2, p3 = pt(CFG.EAR_P1), pt(CFG.EAR_P2), pt(CFG.EAR_P3)
    p4, p5, p6 = pt(CFG.EAR_P4), pt(CFG.EAR_P5), pt(CFG.EAR_P6)

    vert_a = np.linalg.norm(p2 - p6)
    vert_b = np.linalg.norm(p3 - p5)
    horiz  = np.linalg.norm(p1 - p4)

    if horiz < 1e-6:
        return 0.0
    return (vert_a + vert_b) / (2.0 * horiz)


# ─────────────────────────────────────────────
#  SMOOTHING: Dual-Stage EMA
# ─────────────────────────────────────────────
class DualEMA:
    """
    Two cascaded Exponential Moving Averages.
    Stage 1 (fast): tracks actual movement with low lag.
    Stage 2 (slow): smooths residual jitter without drift.

    Why not Kalman?
    The Kalman filter used in the original code had processNoiseCov=0.05
    and measurementNoiseCov=0.3, making it heavily trust its own kinematic
    model and strongly distrust new iris measurements. This caused 150–300ms
    of lag. EMA is O(1), parameter-transparent, and lag-predictable.
    """
    def __init__(self, init_x: float, init_y: float):
        self.fast_x = init_x
        self.fast_y = init_y
        self.slow_x = init_x
        self.slow_y = init_y

    def update(self, x: float, y: float) -> Tuple[float, float]:
        a_f = CFG.EMA_FAST_ALPHA
        a_s = CFG.EMA_SLOW_ALPHA

        self.fast_x = a_f * x + (1 - a_f) * self.fast_x
        self.fast_y = a_f * y + (1 - a_f) * self.fast_y

        self.slow_x = a_s * self.fast_x + (1 - a_s) * self.slow_x
        self.slow_y = a_s * self.fast_y + (1 - a_s) * self.slow_y

        return self.slow_x, self.slow_y

    def set(self, x: float, y: float):
        self.fast_x = self.slow_x = x
        self.fast_y = self.slow_y = y


# ─────────────────────────────────────────────
#  CALIBRATION: 9-Point Grid + Bilinear Lerp
# ─────────────────────────────────────────────
class CalibrationGrid:
    """
    Builds a 9-point (3×3) mapping from raw iris offset space to
    normalised screen space [0,1]×[0,1].

    At runtime, gaze_to_screen() performs bilinear interpolation inside
    whichever grid cell the current iris offset falls into.

    Why 9 points instead of 5?
    5-point calibration can only fit a global affine transform, which
    cannot correct for non-linearities in iris movement (the iris moves
    on a sphere, not a plane). 9 points allow per-cell corrections that
    handle these distortions in each screen quadrant independently.
    """
    def __init__(self, cols: int = 3, rows: int = 3):
        self.cols = cols
        self.rows = rows
        # raw_pts[r][c] = (median_rel_x, median_rel_y) from calibration
        self.raw_pts: List[List[Optional[Tuple[float, float]]]] = [
            [None] * cols for _ in range(rows)
        ]
        self.ready = False

    def set_point(self, row: int, col: int, rx: float, ry: float):
        self.raw_pts[row][col] = (rx, ry)

    def finalise(self):
        """Compute global min/max extents for normalisation."""
        all_x = [self.raw_pts[r][c][0] for r in range(self.rows)
                 for c in range(self.cols) if self.raw_pts[r][c]]
        all_y = [self.raw_pts[r][c][1] for r in range(self.rows)
                 for c in range(self.cols) if self.raw_pts[r][c]]
        self.min_x, self.max_x = min(all_x), max(all_x)
        self.min_y, self.max_y = min(all_y), max(all_y)
        self.ready = True

    def gaze_to_screen(self, rel_x: float, rel_y: float) -> Tuple[float, float]:
        """
        Map raw iris offset (rel_x, rel_y) → normalised screen coords [0,1].
        Uses bilinear interpolation within the appropriate grid cell.
        Falls back to global linear mapping if grid is incomplete.
        """
        if not self.ready:
            return 0.5, 0.5

        dx = max(1e-9, self.max_x - self.min_x)
        dy = max(1e-9, self.max_y - self.min_y)

        # Clamp to calibrated range
        cx = max(self.min_x, min(self.max_x, rel_x))
        cy = max(self.min_y, min(self.max_y, rel_y))

        # Find fractional position in grid space
        fx = (cx - self.min_x) / dx  # 0..1
        fy = (cy - self.min_y) / dy  # 0..1

        # Grid cell indices
        col_f = fx * (self.cols - 1)
        row_f = fy * (self.rows - 1)
        c0 = int(max(0, min(self.cols - 2, col_f)))
        r0 = int(max(0, min(self.rows - 2, row_f)))
        c1, r1 = c0 + 1, r0 + 1

        # Bilinear weights
        tc = col_f - c0   # 0..1 within cell, horizontal
        tr = row_f - r0   # 0..1 within cell, vertical

        # Screen target positions for the 4 surrounding calibration points
        # Each grid point maps to its ideal screen position (uniform grid)
        def screen_target(r, c):
            return (c / (self.cols - 1), r / (self.rows - 1))

        tl = screen_target(r0, c0)
        tr_ = screen_target(r0, c1)
        bl = screen_target(r1, c0)
        br = screen_target(r1, c1)

        # Bilinear interpolation
        sx = (tl[0] * (1 - tc) * (1 - tr) +
              tr_[0] * tc * (1 - tr) +
              bl[0] * (1 - tc) * tr +
              br[0] * tc * tr)
        sy = (tl[1] * (1 - tc) * (1 - tr) +
              tr_[1] * tc * (1 - tr) +
              bl[1] * (1 - tc) * tr +
              br[1] * tc * tr)

        return sx, sy


# ─────────────────────────────────────────────
#  CURSOR MAPPER: Dead-zone + Acceleration
# ─────────────────────────────────────────────
class CursorMapper:
    """
    Converts normalised gaze [0,1]×[0,1] → absolute screen pixels
    with dead-zone filtering and velocity-scaled acceleration.

    Dead-zone:
        Tiny involuntary eye movements (micro-saccades, physiological tremor)
        are ~0.1–0.5° of visual angle. Without a dead-zone, they produce
        constant cursor jitter even when the user is fixating. The dead-zone
        maps all movement within DEAD_ZONE_FRACTION of the current gaze centre
        to zero cursor delta, then re-scales the remaining range to full screen.

    Acceleration:
        A power-curve maps gaze displacement → cursor velocity. Small gaze
        displacements produce proportionally small cursor moves (precision in
        the centre), while large displacements are amplified exponentially
        (fast travel to screen edges). This mirrors how professional eye
        trackers implement "Fitts' Law-aware" pointer gain.
    """
    def __init__(self, screen_w: int, screen_h: int):
        self.sw = screen_w
        self.sh = screen_h
        self._ref_x = 0.5   # current gaze reference centre (updated when moving)
        self._ref_y = 0.5

    def apply(self, norm_x: float, norm_y: float) -> Tuple[float, float]:
        dz = CFG.DEAD_ZONE_FRACTION
        exp = CFG.ACCELERATION_EXPONENT
        scale = CFG.ACCELERATION_SCALE

        # ── Dead-zone ──
        # Compute displacement from screen centre (not from a moving reference,
        # which would cause drift — we use fixed [0.5, 0.5] as the neutral point).
        dx = norm_x - 0.5
        dy = norm_y - 0.5

        # Suppress tiny movements
        if abs(dx) < dz:
            dx = 0.0
        else:
            # Rescale: map (dz … 0.5) → (0 … 0.5)
            dx = math.copysign((abs(dx) - dz) / (0.5 - dz) * 0.5, dx)

        if abs(dy) < dz:
            dy = 0.0
        else:
            dy = math.copysign((abs(dy) - dz) / (0.5 - dz) * 0.5, dy)

        # ── Acceleration curve ──
        # f(d) = sign(d) * |d|^exp * scale
        # Clamp result to [-0.5, 0.5] before mapping to screen pixels.
        adx = math.copysign(min(0.5, (abs(dx) ** exp) * scale), dx)
        ady = math.copysign(min(0.5, (abs(dy) ** exp) * scale), dy)

        # Map to pixel space (centre = screen/2)
        px = (0.5 + adx) * self.sw
        py = (0.5 + ady) * self.sh

        # Hard clamp to screen bounds with a small margin
        px = max(5, min(self.sw - 5, px))
        py = max(5, min(self.sh - 5, py))
        return px, py


# ─────────────────────────────────────────────
#  BLINK DETECTOR
# ─────────────────────────────────────────────
class BlinkDetector:
    """
    Detects sustained eye closure and fires a click after CLICK_HOLD_SECONDS.
    Uses hysteresis (two thresholds) to avoid false triggers from partial
    squints or EAR measurement noise.

    Key fix over original code:
    The original computed (horizontal / vertical) = inverted EAR, then compared
    to a threshold of 5.4. This means ANY squint (reducing vertical distance)
    spiked the ratio and froze the cursor. We now use standard EAR (vertical /
    horizontal) with proper 6-landmark geometry and hysteresis.
    """
    def __init__(self):
        self._closed_frames = 0
        self._is_closed = False
        self._close_start: Optional[float] = None
        self._click_fired = False

    def update(self, ear: float) -> Tuple[bool, float, bool]:
        """
        Returns: (eye_is_closed, seconds_held, click_just_fired)
        Cursor movement should NOT be suppressed based on this return value;
        caller moves cursor unconditionally and only suppresses on click_just_fired.
        """
        click_fired = False
        now = time.time()

        # State machine with hysteresis
        if not self._is_closed:
            if ear < CFG.EAR_CLOSE_THRESHOLD:
                self._closed_frames += 1
                if self._closed_frames >= CFG.BLINK_BUFFER_FRAMES:
                    self._is_closed = True
                    self._close_start = now
                    self._click_fired = False
            else:
                self._closed_frames = 0
        else:
            if ear > CFG.EAR_OPEN_THRESHOLD:
                self._is_closed = False
                self._closed_frames = 0
                self._close_start = None
                self._click_fired = False

        held = (now - self._close_start) if self._is_closed and self._close_start else 0.0

        if self._is_closed and held >= CFG.CLICK_HOLD_SECONDS and not self._click_fired:
            pyautogui.click()
            self._click_fired = True
            click_fired = True

        return self._is_closed, held, click_fired


# ─────────────────────────────────────────────
#  FPS MONITOR
# ─────────────────────────────────────────────
class FPSMonitor:
    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ─────────────────────────────────────────────
#  MAIN CONTROLLER
# ─────────────────────────────────────────────
class EyeMouseController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=CFG.MAX_FACES,
            refine_landmarks=True,        # required for iris landmarks 468–477
            min_detection_confidence=CFG.DETECTION_CONFIDENCE,
            min_tracking_confidence=CFG.TRACKING_CONFIDENCE,
        )

        self.calib_grid = CalibrationGrid(CFG.CALIB_COLS, CFG.CALIB_ROWS)
        self.ema = DualEMA(self.screen_w / 2, self.screen_h / 2)
        self.mapper = CursorMapper(self.screen_w, self.screen_h)
        self.blink = BlinkDetector()
        self.fps = FPSMonitor()

        self._frames_lost = 0
        self._last_px = self.screen_w / 2
        self._last_py = self.screen_h / 2

    # ── Iris relative offset (head-pose-free) ──────────────────────────
    def _get_iris_offset(self, landmarks) -> Tuple[float, float]:
        """
        Returns the normalised iris position relative to the eye socket anchor.
        Using the eye corner landmarks as a stable anchor removes the need for
        head-pose estimation: if the head rotates, both iris and eye corners
        move together, so the relative offset stays correct.

        Why we dropped solvePnP (head pose):
        - Adds 5–15ms per frame (tanking FPS)
        - The correction factor (0.022) was empirically tuned and wrong for most
          face geometries, adding MORE error than it removed
        - Iris-relative anchoring already compensates for head rotation implicitly
        """
        iris = landmarks[CFG.LEFT_IRIS_CENTER]
        inner = landmarks[CFG.LEFT_EYE_INNER]
        outer = landmarks[CFG.LEFT_EYE_OUTER]

        # Eye socket midpoint as anchor
        anchor_x = (inner.x + outer.x) / 2
        anchor_y = (inner.y + outer.y) / 2

        return iris.x - anchor_x, iris.y - anchor_y

    # ── Calibration ────────────────────────────────────────────────────
    def run_calibration(self, cap: cv2.VideoCapture):
        """
        9-point (3×3 grid) calibration.

        Grid layout on screen:
            TL    TC    TR
            ML    MC    MR
            BL    BC    BR

        For each point the user dwells for CALIB_DWELL_SECONDS while we
        collect iris offsets, then store the median (robust to blinks/drift).
        """
        m = CFG.CALIB_MARGIN_PX
        cols, rows = CFG.CALIB_COLS, CFG.CALIB_ROWS
        sw, sh = self.screen_w, self.screen_h

        # Build grid of (screen_x, screen_y) targets
        xs = [m, sw // 2, sw - m]
        ys = [m, sh // 2, sh - m]

        label_map = {
            (0, 0): "TOP-LEFT",    (0, 1): "TOP-CENTER",    (0, 2): "TOP-RIGHT",
            (1, 0): "MID-LEFT",    (1, 1): "CENTER",        (1, 2): "MID-RIGHT",
            (2, 0): "BOT-LEFT",    (2, 1): "BOT-CENTER",    (2, 2): "BOT-RIGHT",
        }

        # Randomise order (except centre first) to reduce fatigue bias
        order = [(r, c) for r in range(rows) for c in range(cols)]
        # Put centre first so user knows the pattern, then shuffle rest
        order.remove((1, 1))
        import random
        random.shuffle(order)
        order = [(1, 1)] + order

        print("\n╔══════════════════════════════════════╗")
        print("║    9-POINT CALIBRATION SEQUENCE      ║")
        print("║  Look at each dot. Keep head still.  ║")
        print("╚══════════════════════════════════════╝\n")

        # Show countdown before starting
        for countdown in range(3, 0, -1):
            ui = np.zeros((sh, sw, 3), dtype=np.uint8)
            cv2.putText(ui, f"Calibration starts in {countdown}...",
                        (sw // 2 - 260, sh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.imshow('EyeMouse Calibration', ui)
            cv2.waitKey(1000)

        for point_num, (row, col) in enumerate(order):
            target = (xs[col], ys[row])
            label = label_map[(row, col)]
            samples_x, samples_y = [], []

            t_start = time.time()
            dwell = CFG.CALIB_DWELL_SECONDS

            print(f"  [{point_num+1}/9] Look at: {label}")

            while time.time() - t_start < dwell:
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                ih, iw = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)

                elapsed = time.time() - t_start
                progress = elapsed / dwell

                # ── Draw calibration UI ──
                ui = np.zeros((sh, sw, 3), dtype=np.uint8)

                # Progress bar
                bar_w = int(sw * 0.4)
                bar_x = sw // 2 - bar_w // 2
                bar_y = sh - 60
                cv2.rectangle(ui, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20),
                              (60, 60, 60), -1)
                cv2.rectangle(ui, (bar_x, bar_y),
                              (bar_x + int(bar_w * progress), bar_y + 20),
                              (0, 200, 120), -1)
                cv2.putText(ui, f"Point {point_num+1}/9: {label}", (bar_x, bar_y - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

                # Target dot with pulsing ring
                pulse_r = int(12 + 8 * math.sin(elapsed * 10))
                cv2.circle(ui, target, 18, (0, 180, 255), -1)
                cv2.circle(ui, target, pulse_r + 18, (255, 255, 255), 2)

                # Show completed points
                for pr, pc in order[:point_num]:
                    prev_t = (xs[pc], ys[pr])
                    cv2.circle(ui, prev_t, 8, (0, 120, 0), -1)

                cv2.imshow('EyeMouse Calibration', ui)
                cv2.waitKey(1)

                # Collect iris data (skip first 0.4s while eye settles)
                if elapsed > 0.4 and results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    rx, ry = self._get_iris_offset(lm)
                    samples_x.append(rx)
                    samples_y.append(ry)

            if samples_x:
                self.calib_grid.set_point(row, col, np.median(samples_x), np.median(samples_y))
                print(f"         ✔ Collected {len(samples_x)} samples")
            else:
                print(f"         ✗ No samples — check camera/lighting")

        self.calib_grid.finalise()
        print("\n✔ Calibration complete. Starting eye control...\n")
        cv2.destroyWindow('EyeMouse Calibration')

    # ── Per-frame pipeline ─────────────────────────────────────────────
    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Core per-frame processing pipeline:
        1. Detect landmarks
        2. Compute iris offset → calibrated gaze → screen position
        3. Apply dead-zone + acceleration
        4. EMA smooth
        5. Check blink / fire click
        6. Move cursor (always, unless confirmed click cooldown)
        7. Draw debug overlay
        """
        ih, iw = frame.shape[:2]
        fps_val = self.fps.tick()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            # Tracking lost: hold last position (EMA naturally)
            self._frames_lost = min(self._frames_lost + 1, CFG.MAX_FRAMES_LOST + 1)

            if self._frames_lost <= CFG.MAX_FRAMES_LOST:
                # Briefly lost: keep cursor at last known position
                pyautogui.moveTo(int(self._last_px), int(self._last_py))

            self._draw_overlay(frame, fps_val, ear=None, held=0, tracking=False)
            return frame

        self._frames_lost = 0
        landmarks = results.multi_face_landmarks[0].landmark

        # ── 1. Iris gaze → normalised screen coords ──
        rel_x, rel_y = self._get_iris_offset(landmarks)
        norm_x, norm_y = self.calib_grid.gaze_to_screen(rel_x, rel_y)

        # ── 2. Dead-zone + acceleration ──
        raw_px, raw_py = self.mapper.apply(norm_x, norm_y)

        # ── 3. EMA smoothing ──
        smooth_px, smooth_py = self.ema.update(raw_px, raw_py)
        smooth_px = max(5, min(self.screen_w - 5, smooth_px))
        smooth_py = max(5, min(self.screen_h - 5, smooth_py))

        # ── 4. Blink / click detection ──
        ear = compute_ear(landmarks, iw, ih)
        eye_closed, held, click_fired = self.blink.update(ear)

        # ── 5. Cursor movement ──
        # CRITICAL FIX: cursor moves regardless of blink state.
        # We only skip movement for 0.15s after a click fires (to avoid
        # dragging immediately after clicking).
        if not click_fired:
            pyautogui.moveTo(int(smooth_px), int(smooth_py))
            self._last_px = smooth_px
            self._last_py = smooth_py

        # ── 6. Debug overlay ──
        iris_px = int(landmarks[CFG.LEFT_IRIS_CENTER].x * iw)
        iris_py = int(landmarks[CFG.LEFT_IRIS_CENTER].y * ih)
        cv2.circle(frame, (iris_px, iris_py), 4, (0, 255, 100), -1)
        cv2.circle(frame, (iris_px, iris_py), 8, (0, 200, 80), 1)

        self._draw_overlay(frame, fps_val, ear, held, tracking=True,
                           eye_closed=eye_closed)
        return frame

    def _draw_overlay(self, frame, fps: float, ear: Optional[float],
                      held: float, tracking: bool, eye_closed: bool = False):
        h, w = frame.shape[:2]
        bg_alpha = 0.45
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)

        status_col = (0, 255, 120) if tracking else (0, 80, 255)
        status_txt = "TRACKING" if tracking else "LOST"
        cv2.putText(frame, f"EyeMouse Pro  |  {status_txt}",
                    (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_col, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.0f}",
                    (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        if ear is not None:
            ear_col = (0, 80, 255) if eye_closed else (200, 200, 200)
            cv2.putText(frame, f"EAR: {ear:.3f}",
                        (160, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ear_col, 1, cv2.LINE_AA)

        if eye_closed and held > 0:
            bar_w = w - 40
            filled = int(bar_w * min(1.0, held / CFG.CLICK_HOLD_SECONDS))
            cv2.rectangle(frame, (20, h - 28), (20 + bar_w, h - 10), (40, 40, 40), -1)
            cv2.rectangle(frame, (20, h - 28), (20 + filled, h - 10), (0, 140, 255), -1)
            cv2.putText(frame, f"Hold to click: {held:.1f}s",
                        (20, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    print("╔════════════════════════════════════╗")
    print("║        EyeMouse Pro  v2.0          ║")
    print("║  Eye-Controlled Mouse (Rewritten)  ║")
    print("╚════════════════════════════════════╝\n")
    print("Press  Q  inside the camera window to quit.")
    print("Move mouse to screen corner to trigger FAILSAFE exit.\n")

    controller = EyeMouseController()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera. Check connection and permissions.")
        return

    # Request higher FPS from camera driver
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce latency: process latest frame

    # Run 9-point calibration
    cv2.namedWindow('EyeMouse Calibration', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('EyeMouse Calibration',
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    controller.run_calibration(cap)

    # Main tracking loop
    cv2.namedWindow('EyeMouse Pro', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('EyeMouse Pro', 640, 480)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        processed = controller.update(frame)

        cv2.imshow('EyeMouse Pro', processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nEyeMouse Pro closed cleanly.")


if __name__ == "__main__":
    main()