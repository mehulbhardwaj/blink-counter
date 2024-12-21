import cv2
import dlib
import numpy as np
import time
from scipy.spatial import distance
import psutil
import threading
from collections import deque
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_analysis.log', mode='a', delay=False),  # Set delay to False for real-time logging
        logging.StreamHandler()  # Optional: also log to console
    ]
)

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
        self.MOUTH_AR_THRESH = 0.2  # Threshold for mouth aspect ratio
        
        # Performance monitoring
        self.frame_times = deque(maxlen=30)
        self.blink_counter = 0
        self.frame_counter = 0
        self.frown_counter = 0
        
        # State tracking
        self.blink_cooldown_counter = 0
        self.is_eye_closed = False
        self.is_frowning = False
        
        # Initialize metrics
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'fps': [],
            'latency': []
        }
        
        # Start time
        self.start_time = None

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

    def calculate_mouth_aspect_ratio(self, mouth_points):
        """Calculate the mouth aspect ratio for frown detection"""
        # Compute vertical distances between outer mouth points
        A = distance.euclidean(mouth_points[3], mouth_points[9])  # Upper to lower middle point
        
        # Compute horizontal distance between mouth corners
        C = distance.euclidean(mouth_points[0], mouth_points[6])  # Corner to corner
        
        # Calculate mouth aspect ratio
        mar = A / C
        return mar

    def estimate_distance(self, face_width_pixels):
        """Estimate distance from camera based on face width"""
        # Constants for distance estimation (calibrated for typical webcam)
        KNOWN_DISTANCE = 60.0  # cm
        KNOWN_WIDTH = 16.0    # cm (average face width)
        KNOWN_WIDTH_PIXELS = 250  # pixels at KNOWN_DISTANCE
        
        # Calculate distance using triangle similarity
        distance = (KNOWN_WIDTH * KNOWN_DISTANCE * KNOWN_WIDTH_PIXELS) / (face_width_pixels * KNOWN_WIDTH)
        
        # Apply smoothing and bounds
        return max(20, min(200, distance))

    def process_frame(self, frame):
        """Process a single frame and return analyzed metrics"""
        frame_start = time.time()
        
        # Convert frame to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Keep original resolution
        
        # Detect faces
        faces = self.detector(gray, 0)
        
        frame_metrics = {
            'blinks': 0,
            'distance': 0,
            'frowning': False
        }
        
        # Update cooldown counters
        if self.blink_cooldown_counter > 0:
            self.blink_cooldown_counter -= 1
        
        # Process the largest face if multiple faces are detected
        if len(faces) > 0:
            face = max(faces, key=lambda rect: (rect.right() - rect.left()) * (rect.bottom() - rect.top()))
            
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
                frame_metrics['blinks'] = 1
            elif ear >= self.EYE_AR_THRESH:
                self.is_eye_closed = False
            
            # Calculate face width and estimate distance
            face_width = face.right() - face.left()
            frame_metrics['distance'] = self.estimate_distance(face_width)
            
            # Extract mouth points for frown detection (outer mouth points)
            mouth_points = points[48:60]
            
            # Calculate mouth aspect ratio
            mar = self.calculate_mouth_aspect_ratio(mouth_points)
            
            # Frown detection without cooldown
            if mar > self.MOUTH_AR_THRESH:
                if not self.is_frowning:
                    self.is_frowning = True
                    self.frown_counter += 1
                    frame_metrics['frowning'] = True
            else:
                self.is_frowning = False
            
            # Draw facial landmarks
            for point in points:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
            
            # Draw debug information
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Calculate frame processing time
        frame_time = time.time() - frame_start
        self.frame_times.append(frame_time)
        
        return frame, frame_metrics
    
    
    def update_performance_metrics(self):
        """Update CPU, memory, and FPS metrics"""
        # Log metrics every 10 seconds instead of every 5 seconds
        if time.time() - self.start_time >= 5:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.Process().memory_percent()
            
            if self.frame_times:
                current_fps = 1.0 / np.mean(self.frame_times)
                current_latency = np.mean(self.frame_times) * 1000  # Convert to ms
            else:
                current_fps = 0
                current_latency = 0
            
            self.metrics['cpu_usage'].append(cpu_percent)
            self.metrics['memory_usage'].append(memory_percent)
            self.metrics['fps'].append(current_fps)
            self.metrics['latency'].append(current_latency)
            
            logging.info(f"CPU: {cpu_percent:.1f}% | Memory: {memory_percent:.1f}% | "
                        f"FPS: {current_fps:.1f} | Latency: {current_latency:.1f}ms")
            self.start_time = time.time()  # Reset start time for next logging interval

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    analyzer = FaceAnalyzer()
    analyzer.start_time = time.time()
    
    # Start metrics monitoring thread
    def metrics_monitor():
        while time.time() - analyzer.start_time < 120:  # Run for 2 minutes
            analyzer.update_performance_metrics()
            time.sleep(1)
    
    metrics_thread = threading.Thread(target=metrics_monitor)
    metrics_thread.start()
    
    frame_count = 0  # Initialize frame counter

    try:
        while time.time() - analyzer.start_time < 120:  # Run for 2 minutes
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1  # Increment frame counter
            
            # Process every 5th frame instead of every 3rd
            if frame_count % 5 == 0:
                # Process frame
                frame, metrics = analyzer.process_frame(frame)
                
                # Update frame counter
                analyzer.frame_counter += 1
                
                # Display metrics on frame
                cv2.putText(frame, f"Blinks: {analyzer.blink_counter}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Distance: {metrics['distance']:.1f}cm", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frowns: {analyzer.frown_counter}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Face Analysis', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate and log final statistics
        avg_cpu = np.mean(analyzer.metrics['cpu_usage'])
        peak_cpu = max(analyzer.metrics['cpu_usage'])
        avg_memory = np.mean(analyzer.metrics['memory_usage'])
        peak_memory = max(analyzer.metrics['memory_usage'])
        avg_fps = np.mean(analyzer.metrics['fps'])
        avg_latency = np.mean(analyzer.metrics['latency'])
        
        logging.info("\n=== Final Statistics ===")
        logging.info(f"Average CPU Usage: {avg_cpu:.1f}%")
        logging.info(f"Peak CPU Usage: {peak_cpu:.1f}%")
        logging.info(f"Average Memory Usage: {avg_memory:.1f}%")
        logging.info(f"Peak Memory Usage: {peak_memory:.1f}%")
        logging.info(f"Average FPS: {avg_fps:.1f}")
        logging.info(f"Average Latency: {avg_latency:.1f}ms")
        logging.info(f"Total Frames Processed: {analyzer.frame_counter}")
        logging.info(f"Total Blinks Detected: {analyzer.blink_counter}")
        logging.info(f"Total Frowns Detected: {analyzer.frown_counter}")

if __name__ == "__main__":
    main()