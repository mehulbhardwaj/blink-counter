import cv2
import time
import signal
import sys
import logging
from face_analyzer import (
    FaceAnalyzer,
    find_working_camera,
    configure_logging,
    signal_handler,
    setup_signal_handler
)
from face_analyzer.config import FRAME_WIDTH, FRAME_HEIGHT, FPS
from threading import Thread
from face_analyzer.monitoring import metrics_monitor
import psutil
from collections import deque
import numpy as np


def main():
    configure_logging()

    # Initialize video capture with working camera
    camera_index = find_working_camera()
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Verify camera is working properly
    if not cap.isOpened():
        logging.error("Error: Camera not accessible")
        return

    # Attach signal handler for graceful exit
    setup_signal_handler(cap)

    analyzer = FaceAnalyzer()
    analyzer.start_time = time.time()

    # Start metrics monitoring thread
    metrics_thread = Thread(target=metrics_monitor, args=(analyzer,))
    metrics_thread.daemon = True  # Make thread daemon so it exits with main program
    metrics_thread.start()

    cpu_readings = deque(maxlen=30)  # Store last 30 readings (1 second at 30 FPS)

    try:
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to capture frame. Exiting...")
                break

            frame_count += 1
            current_time = time.time()
            
            # Calculate FPS
            if current_time - fps_start_time >= 1.0:
                fps = frame_count / (current_time - fps_start_time)
                frame_count = 0
                fps_start_time = current_time

            # Process frame and get metrics
            processed_frame, metrics = analyzer.process_frame(frame)

            # Calculate positions and dimensions for overlays
            frame_width = processed_frame.shape[1]
            frame_height = processed_frame.shape[0]
            metrics_padding = 20
            line_height = 30

            # Add semi-transparent background for top-left metrics
            metrics_overlay = processed_frame.copy()
            cv2.rectangle(metrics_overlay,
                         (metrics_padding, metrics_padding),
                         (300, 150),
                         (0, 0, 0),
                         -1)
            cv2.addWeighted(metrics_overlay, 0.3, processed_frame, 0.7, 0, processed_frame)

            # Display metrics overlay on top-left with white text
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                       (metrics_padding + 10, metrics_padding + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Blinks: {analyzer.blink_counter}", 
                       (metrics_padding + 10, metrics_padding + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if 'distance' in metrics:
                cv2.putText(processed_frame, f"Distance: {metrics['distance']:.1f}cm", 
                           (metrics_padding + 10, metrics_padding + 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Frowns: {analyzer.frown_counter}", 
                       (metrics_padding + 10, metrics_padding + 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display system metrics on bottom-right
            runtime = time.time() - analyzer.start_time
            current_cpu = psutil.cpu_percent()
            cpu_readings.append(current_cpu)
            avg_cpu = np.mean(cpu_readings) if cpu_readings else current_cpu
            memory_usage = psutil.Process().memory_percent()
            
            # Add semi-transparent background for bottom-right metrics and quit button
            metrics_overlay = processed_frame.copy()
            cv2.rectangle(metrics_overlay,
                         (frame_width - 300, frame_height - 190),  # Increased height for button spacing
                         (frame_width - metrics_padding, frame_height - metrics_padding),
                         (0, 0, 0),
                         -1)
            cv2.addWeighted(metrics_overlay, 0.3, processed_frame, 0.7, 0, processed_frame)
            
            # Add system metrics text
            text_x = frame_width - 290  # Left alignment position
            cv2.putText(processed_frame, 
                       f"Runtime: {int(runtime//3600):02d}:{int((runtime%3600)//60):02d}:{int(runtime%60):02d}",
                       (text_x, frame_height - 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame,
                       f"CPU: {avg_cpu:.1f}%",
                       (text_x, frame_height - 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame,
                       f"Memory: {memory_usage:.1f}%",
                       (text_x, frame_height - 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add quit button aligned with text
            button_width = 100
            button_height = 30
            button_x = text_x  # Align with text
            button_y = frame_height - 80  # Position below the last text line
            
            # Create deep red quit button with shadow
            button_overlay = processed_frame.copy()
            # Shadow
            cv2.rectangle(processed_frame,
                         (button_x + 2, button_y + 2),
                         (button_x + button_width + 2, button_y + button_height + 2),
                         (0, 0, 0),
                         -1)
            # Button
            cv2.rectangle(button_overlay,
                         (button_x, button_y),
                         (button_x + button_width, button_y + button_height),
                         (139, 0, 0),  # Deep red color
                         -1)
            cv2.addWeighted(button_overlay, 0.9, processed_frame, 0.1, 0, processed_frame)
            
            # Add white text to button
            cv2.putText(processed_frame, 
                       "QUIT",
                       (button_x + 25, button_y + 22),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (255, 255, 255),
                       2)

            # Update the mouse callback coordinates for the new button position
            quit_button = (button_x, button_y, button_width, button_height)

            # Show warning for close distance
            if metrics.get('screen_distance_alert'):
                cv2.putText(processed_frame, "WARNING: Too Close to Screen!", 
                           (frame_width//4, frame_height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Display the frame
            cv2.imshow('Face Analysis', processed_frame)

            # Handle mouse clicks for quit button
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    if (quit_button[0] <= x <= quit_button[0] + quit_button[2] and 
                        quit_button[1] <= y <= quit_button[1] + quit_button[3]):
                        analyzer.log_final_statistics()
                        cap.release()
                        cv2.destroyAllWindows()
                        sys.exit(0)

            cv2.setMouseCallback('Face Analysis', mouse_callback)

            # Check for keyboard interrupt (q key)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        analyzer.log_final_statistics()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()