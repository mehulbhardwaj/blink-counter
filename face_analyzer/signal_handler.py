import logging
import signal
import sys

def signal_handler(cap, signum, frame):
    """Handle cleanup when receiving interrupt signal"""
    logging.info("Interrupt received, cleaning up...")
    if cap is not None:
        cap.release()
    sys.exit(0)

def setup_signal_handler(cap):
    """Set up the signal handler for graceful shutdown"""
    signal.signal(signal.SIGINT, lambda signum, frame: signal_handler(cap, signum, frame))