# Face Analysis

This project is a real-time face analysis system that uses a webcam to detect and analyze facial expressions. It provides insights such as blink detection, frown detection, and distance estimation based on face width. The system also tracks performance metrics like CPU usage, memory usage, FPS, and latency.

## Features
- **Enhanced Blink Detection**: Improved accuracy of blink detection using rolling average and cooldown mechanism
- **Precise Distance Estimation**: Calibrated distance calculation with better accuracy and real-time warnings when too close to screen
- **Performance Monitoring**: 
  - Real-time CPU usage (smoothed 1-second average)
  - Memory usage
  - FPS tracking
  - Processing latency
  - Total runtime
- **Improved UI/UX**:
  - Semi-transparent overlay for metrics display
  - Organized metrics layout with clear grouping
  - Integrated quit button with visual feedback
  - Face landmark visualization
  - Eye contour tracking
- **Automatic Camera Selection**:
  - The system automatically scans indices 0-5 for working cameras
  - External cameras (higher indices) are preferred over built-in cameras
  - Camera selection is logged in `face_analysis.log`
  - No manual configuration needed for camera selection
  - If you want to override the automatic selection, you can modify the `find_working_camera()` function in `utils.py`


## Code Structure Improvements
- Reorganized into modular components
- Separated face analysis logic into dedicated class
- Enhanced error handling and safety checks
- Improved code readability and documentation
- Added configuration constants for easy tuning

## Getting Started

### Prerequisites
Ensure you have the following installed on your system:
- Python 3.7+
- A webcam or video capture device

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mehulbhardwaj/blink-counter.git
   cd face-analysis-system
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the `shape_predictor_68_face_landmarks.dat` file from [dlib's model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project directory.

For Mac ARM users, you may need to install the dependencies using conda:
1. Install conda: https://docs.anaconda.com/miniconda/install/#quick-command-line-install 
2. Add conda to your path: https://docs.anaconda.com/miniconda/install/#add-conda-to-your-path-macos-and-linux
3. Get Python 3.9 - 3.12 (for compatibility with Pytorch): https://pytorch.org/get-started/locally/
4. Create a conda environment:
   ```bash
   conda create -n face_analysis python=3.11
   ```
5. Activate the environment:
   ```bash
   conda activate face_analysis
   ```
6. Install the dependencies:
   ```bash
   conda install -c conda-forge opencv numpy scipy psutil dlib opencv 
   ```

### Running the Program
Execute the `main.py` script to start the face analysis system:
```bash
python main.py
```
For Mac ARM users, you will need to provide camera access to your terminal/code editor using Settings -> Privacy -> Camera.

If you are using a different camera, you can change the camera index in the `main.py` file:
   'cap = cv2.VideoCapture(1)' to 'cap = cv2.VideoCapture(0)'

### Usage
- The system will open a window displaying the webcam feed with metrics overlaid
- Real-time metrics are displayed in semi-transparent overlays:
  - Top-left: Performance metrics (FPS, Blinks, Distance, Frowns)
  - Bottom-right: System metrics (Runtime, CPU, Memory) and Quit button
- Press **Q** or click the Quit button to exit the program

## Project Structure
- **main.py**: The main script for running the face analysis system
- **face_analyzer/**: Module containing face analysis logic
- **requirements.txt**: Lists all the Python dependencies required
- **face_analysis.log**: A log file that records performance metrics and final statistics
- **shape_predictor_68_face_landmarks.dat**: Pre-trained dlib model for facial landmarks detection (not included)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or additional features.

## Troubleshooting
If you encounter issues:
- Ensure the `shape_predictor_68_face_landmarks.dat` file is in the correct directory
- Verify that your webcam is working correctly
- Check the logs in `face_analysis.log` for errors

## Acknowledgments

This project is based on the work originally created by [Saksham Chaurasia](https://github.com/imsaksham-c/FaceAnalysis). The original repository can be found [here](https://github.com/imsaksham-c/FaceAnalysis).

Changes have been made to extend and adapt the project with improved accuracy, better UI/UX, and additional features.

- [Dlib](http://dlib.net/) for providing powerful face detection and landmark models
- OpenCV for video processing and rendering
- Scipy and NumPy for numerical computations
  