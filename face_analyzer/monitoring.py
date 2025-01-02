import logging
import time

def configure_logging():
    """Configure logging settings for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def metrics_monitor(analyzer, duration=120):
    """Monitor and update performance metrics in a separate thread."""
    start_time = time.time()
    while time.time() - start_time < duration:
        analyzer.update_performance_metrics()
        time.sleep(1) 