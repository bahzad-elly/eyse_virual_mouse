import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np

# --- PYAUTOGUI CONFIGURATION ---
pyautogui.FAILSAFE = True  
pyautogui.PAUSE = 0.001

class AdvancedGazeController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Initialize MediaPipe Face Mesh with optimized parameters
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,  # Higher strictness for clean data
            min_tracking_confidence=0.8
        )
        
        # Calibration bounds (normalized space relative to eye anchor)
        self.calib_min_x = -0.05
        self.calib_max_x = 0.05
        self.calib_min_y = -0.05
        self.calib_max_y = 0.05
        
        # Control & Advanced Sensitivity Configuration
        self.sensitivity_scale = 2.2  # Increased for easier edge reach without head movement
        self.smooth_x, self.smooth_y = self.screen_w // 2, self.screen_h // 2
        
        # Blink Detection Settings
        self.BLINK_THRESHOLD = 5.4  # Finetuned to avoid accidental click triggers
        self.blink_start_time = None
        self.click_triggered = False

        # --- ADVANCED RADAR STATE TRACKING (KALMAN FILTER) ---
        # State: [x, y, dx, dy] (Position and Velocity)
        self.kalman = cv2.KalmanFilter(4, 2, 0)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], 
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], 
                                                 [0, 1, 0, 1], 
                                                 [0, 0, 1, 0], 
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Handles jitter
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5  # Rejects false skips
        
        # Initialize Kalman position to center screen
        self.kalman.statePost = np.array([[self.screen_w/2], [self.screen_h/2], [0], [0]], np.float32)
        
        # Track tracking health
        self.last_valid_time = time.time()

    def get_head_pose_offsets(self, landmarks, img_w, img_h):
        """ Estimates head orientation matrix to counteract posture changes. """
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye corner
            (225.0, 170.0, -135.0),      # Right eye corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

        image_points = np.array([
            (landmarks[1].x * img_w, landmarks[1].y * img_h),     
            (landmarks[152].x * img_w, landmarks[152].y * img_h), 
            (landmarks[33].x * img_w, landmarks[33].y * img_h),   
            (landmarks[263].x * img_w, landmarks[263].y * img_h), 
            (landmarks[61].x * img_w, landmarks[61].y * img_h),   
            (landmarks[291].x * img_w, landmarks[291].y * img_h)  
        ], dtype=np.float64)

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            return rotation_vector[0][0], rotation_vector[1][0]
        return 0.0, 0.0

    def run_calibration(self, cap):
        """ Interactive calibration sequence to lock down personal gaze endpoints. """
        points = [
            ("CENTER", (self.screen_w // 2, self.screen_h // 2)),
            ("TOP-LEFT", (80, 80)),
            ("TOP-RIGHT", (self.screen_w - 80, 80)),
            ("BOTTOM-LEFT", (80, self.screen_h - 80)),
            ("BOTTOM-RIGHT", (self.screen_w - 80, self.screen_h - 80))
        ]
        
        collected_x, collected_y = [], []
        print("\n=== SYSTEM CALIBRATION ACTIVE ===")
        print("Keep your head perfectly still. Move ONLY your eyes to look at the targets.")
        
        for name, pos in points:
            start_time = time.time()
            temp_x, temp_y = [], []
            
            while time.time() - start_time < 2.2:
                success, frame = cap.read()
                if not success: continue
                
                frame = cv2.flip(frame, 1)
                img_h, img_w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                calib_ui = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                
                # Render Target Visuals
                cv2.circle(calib_ui, pos, 20, (0, 165, 255), -1)
                pulse = abs(int(8 + 14 * math.sin(time.time() * 15)))
                cv2.circle(calib_ui, pos, pulse, (255, 255, 255), 2)
                
                cv2.putText(calib_ui, f"Focus here: {name}", (self.screen_w // 2 - 180, self.screen_h // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.imshow('System Output Display', calib_ui)
                cv2.waitKey(1)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    anchor = landmarks[362]  # Inner left eye corner
                    iris = landmarks[468]    # Center iris point
                    
                    temp_x.append(iris.x - anchor.x)
                    temp_y.append(iris.y - anchor.y)
            
            if temp_x and temp_y:
                collected_x.append(np.median(temp_x))
                collected_y.append(np.median(temp_y))
                
        # Lock in safety bounds
        self.calib_min_x = min(collected_x)
        self.calib_max_x = max(collected_x)
        self.calib_min_y = min(collected_y)
        self.calib_max_y = max(collected_y)
        
        print("✔ Calibration Configured Successfully!")

    def update_pipeline(self, frame):
        """ Core logic engine containing Kalman filtering matrices and positioning engines. """
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        # Always run Kalman prediction step to keep state rolling smoothly
        prediction = self.kalman.predict()
        pred_x, pred_y = prediction[0][0], prediction[1][0]

        if not results.multi_face_landmarks:
            # If tracking drops out completely, use the prediction to avoid mouse freezes
            if time.time() - self.last_valid_time < 0.5: # 500ms grace period
                pyautogui.moveTo(int(pred_x), int(pred_y))
            return frame
            
        self.last_valid_time = time.time()
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 1. EYE TRANSFORMATION MATRICES
        left_iris = landmarks[468]
        eye_anchor = landmarks[362]
        
        rel_x = left_iris.x - eye_anchor.x
        rel_y = left_iris.y - eye_anchor.y
        
        # Counteract head sway using SolvePnP Pitch & Yaw
        pitch, yaw = self.get_head_pose_offsets(landmarks, img_w, img_h)
        rel_x -= (yaw * 0.025)
        rel_y += (pitch * 0.025)

        # Secure non-zero denominators
        delta_x = max(0.012, self.calib_max_x - self.calib_min_x)
        delta_y = max(0.012, self.calib_max_y - self.calib_min_y)

        # Scale raw signals out linearly
        norm_x = (rel_x - self.calib_min_x) / delta_x
        norm_y = (rel_y - self.calib_min_y) / delta_y
        
        # Apply progressive center-outward scaling configuration multipliers
        norm_x = 0.5 + (norm_x - 0.5) * self.sensitivity_scale
        norm_y = 0.5 + (norm_y - 0.5) * self.sensitivity_scale
        
        # Raw Target Coordinates
        target_x = norm_x * self.screen_w
        target_y = norm_y * self.screen_h
        
        # 2. KALMAN FILTER MEASUREMENT UPDATE (Instant Jitter Suppression)
        measurement = np.array([[np.float32(target_x)], [np.float32(target_y)]], np.float32)
        self.kalman.correct(measurement)
        
        # Extract stabilized coordinates directly from the updated tracking engine state
        self.smooth_x = self.kalman.statePost[0][0]
        self.smooth_y = self.kalman.statePost[1][0]
        
        # Clean Border Safe Clipping
        self.smooth_x = max(8, min(self.screen_w - 8, self.smooth_x))
        self.smooth_y = max(8, min(self.screen_h - 8, self.smooth_y))

        # 3. ADVANCED CLICK HANDLING SYSTEM
        left_eye_top = landmarks[386]
        left_eye_bottom = landmarks[374]
        left_eye_inner = landmarks[362]
        left_eye_outer = landmarks[263]
        
        v_dist = math.hypot(left_eye_top.x - left_eye_bottom.x, left_eye_top.y - left_eye_bottom.y)
        h_dist = math.hypot(left_eye_inner.x - left_eye_outer.x, left_eye_inner.y - left_eye_outer.y)
        ear_ratio = (h_dist / v_dist) if v_dist != 0 else 0
        
        if ear_ratio > self.BLINK_THRESHOLD:
            if self.blink_start_time is None:
                self.blink_start_time = time.time()
            
            elapsed = time.time() - self.blink_start_time
            countdown = max(0.0, 1.0 - elapsed)
            
            if countdown > 0:
                cv2.putText(frame, f"Triggering Click: {countdown:.1f}s", (40, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2, cv2.LINE_AA)
            
            if elapsed >= 1.0 and not self.click_triggered:
                pyautogui.click()
                print("💥 Click Dispatched!")
                self.click_triggered = True
        else:
            self.blink_start_time = None
            self.click_triggered = False
            # Dispatch filtered positional coordinates to operating system cursor
            pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y))

        # Diagnostics Window Elements
        cv2.circle(frame, (int(left_iris.x * img_w), int(left_iris.y * img_h)), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"Tracking State: OK | EAR: {ear_ratio:.2f}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        
        return frame

def main():
    controller = AdvancedGazeController()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Hardware Error: Video capture device could not be opened.")
        return

    # Build borderless window properties
    cv2.namedWindow('System Output Display', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('System Output Display', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Initialize calibration matrix setup
    controller.run_calibration(cap)
    
    # Restore interface display format
    cv2.setWindowProperty('System Output Display', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow('System Output Display', 640, 480)
    
    print("\n>>> System live! Press 'q' inside the camera tracking window to exit safely.")
    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue
            
        frame = cv2.flip(frame, 1)
        processed_frame = controller.update_pipeline(frame)
        
        cv2.imshow('System Output Display', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Clean shutdown sequence complete.")

if __name__ == "__main__":
    main()