# Handwritten Digit Real-Time Recognition

## Project Overview
This project uses OpenCV and TensorFlow to create a real-time handwritten digit recognition system. It captures video from a webcam, processes frames to detect handwritten digits, and recognizes them using a CNN model.

## Features
- Real-time video processing using OpenCV
- Handwritten digit detection (0-9) with contour detection and thresholding
- CNN-based digit recognition

## Requirements
- Python 3.x
- OpenCV 4.x
- TensorFlow 2.x
- NumPy
- Keras

## Installation
1. Clone the repository:  
   `git clone https://github.com/your-username/Handwritten-Digit-Real-Time-Recognition.git`
2. Install required libraries:  
   `pip install -r requirements.txt`
3. Run the project:  
   `python main.py`

## Usage
1. Run the project.
2. Write digits on paper.
3. Hold the paper in front of the webcam.
4. The system will detect and recognize digits in real-time.

## Model Training
The CNN model is trained on the MNIST dataset. To retrain:
1. Download the MNIST dataset.
2. Modify `train_model.py` as needed.
3. Run the script to train the model.

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! Fork, make changes, and submit a pull request.

## Acknowledgments
- MNIST dataset
- OpenCV documentation
- TensorFlow documentation
