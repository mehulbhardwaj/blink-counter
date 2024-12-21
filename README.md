# Face Analysis

This project is a real-time face analysis system that uses a webcam to detect and analyze facial expressions. It provides insights such as blink detection, frown detection, and distance estimation based on face width. The system also tracks performance metrics like CPU usage, memory usage, FPS, and latency.

## Features
- **Blink Detection**: Identifies blinks based on the eye aspect ratio (EAR).
- **Frown Detection**: Detects frowns based on the mouth aspect ratio (MAR).
- **Distance Estimation**: Estimates the distance of a person from the camera based on face width.
- **Performance Monitoring**: Logs CPU usage, memory usage, FPS, and processing latency.

## Getting Started

### Prerequisites
Ensure you have the following installed on your system:
- Python 3.7+
- A webcam or video capture device

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/imsaksham-c/FaceAnalysis.git
   cd face-analysis-system
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the `shape_predictor_68_face_landmarks.dat` file from [dlib's model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project directory.

### Running the Program
Execute the `main.py` script to start the face analysis system:
```bash
python main.py
```

### Usage
- The system will open a window displaying the webcam feed with metrics overlaid.
- Press **Q** to quit the program at any time.

## Project Structure
- **main.py**: The main script for running the face analysis system.
- **requirements.txt**: Lists all the Python dependencies required.
- **face_analysis.log**: A log file that records performance metrics and final statistics.
- **shape_predictor_68_face_landmarks.dat**: A pre-trained dlib model for facial landmarks detection (not included in the repository; download separately).

## Output
The system displays:
- Blink count
- Distance estimation in cm
- Frown count
- Real-time performance metrics such as FPS and latency

Logs are stored in `face_analysis.log` and include average and peak CPU/memory usage, FPS, and latency.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or additional features.

## Acknowledgments
- [Dlib](http://dlib.net/) for providing powerful face detection and landmark models.
- OpenCV for video processing and rendering.
- Scipy and NumPy for numerical computations. 

## Troubleshooting
If you encounter issues:
- Ensure the `shape_predictor_68_face_landmarks.dat` file is in the correct directory.
- Verify that your webcam is working correctly.
- Check the logs in `face_analysis.log` for errors.

---