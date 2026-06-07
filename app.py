import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np

# --- PYAUTOGUI CONFIGURATION ---
pyautogui.FAILSAFE = True  
pyautogui.PAUSE = 0.001

class UltimateGazeController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Optimized MediaPipe Configuration
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,  # Lowered slightly to prevent edge dropouts
            min_tracking_confidence=0.6
        )
        
        # Initial Calibration Bounds (Will dynamically auto-adjust if user overshoots)
        self.calib_min_x = -0.04
        self.calib_max_x = 0.04
        self.calib_min_y = -0.04
        self.calib_max_y = 0.04
        
        # Control & Advanced Sensitivity Configuration
        self.base_sensitivity = 2.5  
        self.smooth_x, self.smooth_y = self.screen_w // 2, self.screen_h // 2
        
        # Blink Detection Settings
        self.BLINK_THRESHOLD = 5.4  
        self.blink_start_time = None
        self.click_triggered = False

        # Advanced Kalman Filtering States
        self.kalman = cv2.KalmanFilter(4, 2, 0)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05  
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.3  
        self.kalman.statePost = np.array([[self.screen_w/2], [self.screen_h/2], [0], [0]], np.float32)

    def get_head_pose_offsets(self, landmarks, img_w, img_h):
        """ Computes head rotation values using iterative Perspective-n-Point solvers. """
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
        return (rotation_vector[0][0], rotation_vector[1][0]) if success else (0.0, 0.0)

    def run_calibration(self, cap):
        """ Executing an interactive fullscreen calibration matrix routine. """
        points = [
            ("CENTER", (self.screen_w // 2, self.screen_h // 2)),
            ("TOP-LEFT", (100, 100)),
            ("TOP-RIGHT", (self.screen_w - 100, 100)),
            ("BOTTOM-LEFT", (100, self.screen_h - 100)),
            ("BOTTOM-RIGHT", (self.screen_w - 100, self.screen_h - 100))
        ]
        
        collected_x, collected_y = [], []
        print("\n=== INITIALIZING INTELLIGENT CALIBRATION ===")
        print("Look directly at each target point using ONLY your eyes. Keep head steady.")
        
        for name, pos in points:
            start_time = time.time()
            temp_x, temp_y = [], []
            
            while time.time() - start_time < 2.0:
                success, frame = cap.read()
                if not success: continue
                
                frame = cv2.flip(frame, 1)
                img_h, img_w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                calib_ui = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                
                # Visual targeting system UI
                cv2.circle(calib_ui, pos, 22, (0, 120, 255), -1)
                pulse = abs(int(6 + 12 * math.sin(time.time() * 14)))
                cv2.circle(calib_ui, pos, pulse, (255, 255, 255), 2)
                
                cv2.putText(calib_ui, f"Focus View: {name}", (self.screen_w // 2 - 160, self.screen_h // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.imshow('System Production Monitor', calib_ui)
                cv2.waitKey(1)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    anchor = landmarks[362]  
                    iris = landmarks[468]    
                    temp_x.append(iris.x - anchor.x)
                    temp_y.append(iris.y - anchor.y)
            
            if temp_x and temp_y:
                collected_x.append(np.median(temp_x))
                collected_y.append(np.median(temp_y))
                
        # Commit boundaries safely
        self.calib_min_x = min(collected_x)
        self.calib_max_x = max(collected_x)
        self.calib_min_y = min(collected_y)
        self.calib_max_y = max(collected_y)
        print("✔ Base Calibration Matrix Acquired.")

    def update_pipeline(self, frame):
        """ Core logic tracking pipeline with auto-recovery and bounds adaptation. """
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        # Advance the Kalman filter prediction state ahead
        prediction = self.kalman.predict()

        if not results.multi_face_landmarks:
            # Dropback Auto-Recovery Mode: prevent mouse freezes during frame drops
            pyautogui.moveTo(int(prediction[0][0]), int(prediction[1][0]))
            return frame
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 1. READ EYE POSITION 
        left_iris = landmarks[468]
        eye_anchor = landmarks[362]
        
        rel_x = left_iris.x - eye_anchor.x
        rel_y = left_iris.y - eye_anchor.y
        
        # Compensate head rotation angles dynamically
        pitch, yaw = self.get_head_pose_offsets(landmarks, img_w, img_h)
        rel_x -= (yaw * 0.022)
        rel_y += (pitch * 0.022)

        # --- ADVANCED IMPROVEMENT 1: AUTO-RELAXATION OF CALIBRATION BOUNDS ---
        # If your eye moves further than what was recorded during calibration, 
        # dynamically expand the bounds immediately instead of letting the mouse get stuck!
        if rel_x < self.calib_min_x: self.calib_min_x = rel_x
        if rel_x > self.calib_max_x: self.calib_max_x = rel_x
        if rel_y < self.calib_min_y: self.calib_min_y = rel_y
        if rel_y > self.calib_max_y: self.calib_max_y = rel_y

        # Define denominators safely
        delta_x = max(0.012, self.calib_max_x - self.calib_min_x)
        delta_y = max(0.012, self.calib_max_y - self.calib_min_y)

        # Calculate a true normalized linear percentage value (0.0 to 1.0)
        norm_x = (rel_x - self.calib_min_x) / delta_x
        norm_y = (rel_y - self.calib_min_y) / delta_y
        
        # --- ADVANCED IMPROVEMENT 2: NON-LINEAR CUBIC CURVE SCALING ---
        # Instead of straight multiplying, we use an exponential power curve. 
        # This gives high precision in the center, and massive acceleration at the edges.
        diff_x = norm_x - 0.5
        diff_y = norm_y - 0.5
        
        # Cubic curve mapping formula ($f(x) = 0.5 + sign(dx) * |dx|^1.5 * scale$)
        scaled_x = 0.5 + np.sign(diff_x) * (abs(diff_x) ** 1.3) * self.base_sensitivity
        scaled_y = 0.5 + np.sign(diff_y) * (abs(diff_y) ** 1.3) * self.base_sensitivity
        
        # Project values to absolute pixel dimensions
        target_x = scaled_x * self.screen_w
        target_y = scaled_y * self.screen_h
        
        # 2. STATE CORRECTION & STABILIZATION (KALMAN)
        measurement = np.array([[np.float32(target_x)], [np.float32(target_y)]], np.float32)
        self.kalman.correct(measurement)
        
        self.smooth_x = self.kalman.statePost[0][0]
        self.smooth_y = self.kalman.statePost[1][0]
        
        # Dynamic Safe Margins
        self.smooth_x = max(10, min(self.screen_w - 10, self.smooth_x))
        self.smooth_y = max(10, min(self.screen_h - 10, self.smooth_y))

        # 3. TIME-BASED CLICK SYSTEM
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
                cv2.putText(frame, f"Holding Action: {countdown:.1f}s", (40, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2, cv2.LINE_AA)
            
            if elapsed >= 1.0 and not self.click_triggered:
                pyautogui.click()
                print("💥 Mouse Click Dispatched Safely!")
                self.click_triggered = True
        else:
            self.blink_start_time = None
            self.click_triggered = False
            # Move the cursor only when our tracking state is valid and eye is wide open
            pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y))

        # Graphic overlay metrics
        cv2.circle(frame, (int(left_iris.x * img_w), int(left_iris.y * img_h)), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"System Active | EAR: {ear_ratio:.2f}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        
        return frame

def main():
    controller = UltimateGazeController()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Hardware Fault: Primary camera capture stream inaccessible.")
        return

    # Initialize Fullscreen View canvas
    cv2.namedWindow('System Production Monitor', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('System Production Monitor', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Perform runtime matrix setup calibration
    controller.run_calibration(cap)
    
    # Resize tracking monitor window frame seamlessly
    cv2.setWindowProperty('System Production Monitor', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow('System Production Monitor', 640, 480)
    
    print("\n>>> Control Engine Running Smoothly. Focus eye on corners to watch adaptation work.")
    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue
            
        frame = cv2.flip(frame, 1)
        processed_frame = controller.update_pipeline(frame)
        
        cv2.imshow('System Production Monitor', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed properly.")

if __name__ == "__main__":
    main()