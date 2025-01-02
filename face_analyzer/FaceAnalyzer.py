import cv2
import dlib
import numpy as np
import logging
from collections import deque
import time
from scipy.spatial import distance

class FaceAnalyzer:
    def __init__(self):
        # Initialize face detector and facial landmarks predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        # Constants for blink detection
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 3
        self.BLINK_COOLDOWN = 10  # frames to wait before detecting next blink
        
        # Constants for frown detection
        self.MOUTH_AR_THRESH = 0.2
        
        # Performance monitoring
        self.frame_times = deque(maxlen=30)
        self.blink_counter = 0
        self.frame_counter = 0
        self.frown_counter = 0
        
        # State tracking
        self.blink_cooldown_counter = 0
        self.is_eye_closed = False
        self.is_frowning = False
        self.distance_alert_cooldown = 0
        self.distance_alert_counter = 0
        
        # Initialize metrics with default values to prevent empty sequence errors
        self.metrics = {
            'cpu_usage': [0.0],
            'memory_usage': [0.0],
            'fps': [0.0],
            'latency': [0.0],
            'distance': 0.0,
            'screen_distance_alert': False
        }
        
        # Start time
        self.start_time = time.time()
        
        # Warning settings
        self.warning_timestamp = 0
        self.WARNING_DURATION = 2  # seconds

    def process_frame(self, frame):
        """Process a single frame and return analyzed metrics"""
        if frame is None:
            return frame, self.metrics
            
        frame_start = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        
        # Process each face (assuming single face for now)
        for face in faces:
            # Get facial landmarks
            landmarks = self.predictor(gray, face)
            points = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            # Extract eye coordinates
            left_eye = points[42:48]
            right_eye = points[36:42]
            
            # Calculate eye aspect ratios
            left_ear = self.calculate_eye_aspect_ratio(left_eye)
            right_ear = self.calculate_eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Blink detection with cooldown
            if ear < self.EYE_AR_THRESH and not self.is_eye_closed and self.blink_cooldown_counter == 0:
                self.is_eye_closed = True
                self.blink_counter += 1
                self.blink_cooldown_counter = self.BLINK_COOLDOWN
            elif ear >= self.EYE_AR_THRESH:
                self.is_eye_closed = False
            
            if self.blink_cooldown_counter > 0:
                self.blink_cooldown_counter -= 1
            
            # Draw facial landmarks
            for point in points:
                cv2.circle(frame, tuple(point), 1, (0, 255, 0), -1)
            
            # Draw eye contours
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
            
            # Calculate distance (adjusted calibration)
            face_width = face.right() - face.left()
            self.metrics['distance'] = self.estimate_distance(face_width)
            self.metrics['screen_distance_alert'] = self.metrics['distance'] < 40  # Alert if closer than 40cm
            
            # Draw face rectangle
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Update performance metrics
        self.metrics['latency'].append((time.time() - frame_start) * 1000)  # Convert to ms
        
        return frame, self.metrics

    def calculate_eye_aspect_ratio(self, eye_points):
        """Calculate the eye aspect ratio for blink detection"""
        # Compute vertical distances
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])
        # Compute horizontal distance
        C = distance.euclidean(eye_points[0], eye_points[3])
        # Calculate eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    def estimate_distance(self, face_width_pixels):
        """Estimate distance from camera based on face width"""
        # Adjusted constants for better accuracy
        KNOWN_DISTANCE = 50.0  # cm (calibration distance)
        KNOWN_WIDTH = 16.0     # cm (average face width)
        KNOWN_WIDTH_PIXELS = 300  # pixels at KNOWN_DISTANCE
        
        # Calculate distance using triangle similarity
        distance = (KNOWN_WIDTH * KNOWN_DISTANCE * KNOWN_WIDTH_PIXELS) / (face_width_pixels * KNOWN_WIDTH)
        
        # Apply tighter bounds and offset correction
        distance = max(10, min(150, distance))  # Adjusted bounds
        return distance

    def log_final_statistics(self):
        """Log final performance statistics"""
        if not any(self.metrics['cpu_usage']):
            self.metrics['cpu_usage'] = [0.0]
        if not any(self.metrics['memory_usage']):
            self.metrics['memory_usage'] = [0.0]
        if not any(self.metrics['fps']):
            self.metrics['fps'] = [0.0]
        if not any(self.metrics['latency']):
            self.metrics['latency'] = [0.0]

        logging.info(f"""
Final Statistics:
----------------
Total Runtime: {time.time() - self.start_time:.1f} seconds
Peak CPU Usage: {max(self.metrics['cpu_usage']):.1f}%
Peak Memory Usage: {max(self.metrics['memory_usage']):.1f}%
Average FPS: {np.mean(self.metrics['fps']):.1f}
Average Latency: {np.mean(self.metrics['latency']):.1f}ms
Total Blinks Detected: {self.blink_counter}
Total Frowns Detected: {self.frown_counter}
""")