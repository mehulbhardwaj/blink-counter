import cv2
import logging

def find_working_camera():
    """
    Find the best working camera by testing indices from 0 to 5.
    Prefers external cameras (higher indices) over built-in cameras.
    Returns the highest index of a working camera or 0 if none found.
    """
    working_cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                working_cameras.append(i)
                logging.info(f"Found working camera at index {i}")
    
    if working_cameras:
        # Prefer the highest index (usually external cameras)
        selected_camera = working_cameras[-1]
        logging.info(f"Selected camera at index {selected_camera}")
        return selected_camera
    
    logging.warning("No working camera found, defaulting to camera 0")
    return 0

def configure_logging():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('face_analysis.log', mode='a', delay=False),
            logging.StreamHandler()
        ]
    )