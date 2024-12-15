# YOLOv5 Object Detection with Webcam and Trailer Overlay

This project uses the YOLOv5 model for object detection on a webcam feed and overlays a trailer video on the detected objects. The trailer video is paused when certain objects are detected, and the opacity of the trailer is reduced when a person is detected. Additionally, an icon is displayed on the top right of the screen when the trailer is paused.

## Features

- Object detection using YOLOv5
- Overlay trailer video on detected objects
- Pause trailer video when certain objects are detected
- Reduce trailer opacity when a person is detected
- Display an icon when the trailer is paused

## Requirements

- Python 3.x
- OpenCV
- PyTorch
- YOLOv5 model

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:char-projects/spatial-computing.git spatialComputing
   cd ./spatialComputing/eyeContact
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have a `trailer.mp4` file in the project directory. You can change the source to any other video file if needed.
2. Run the script:
   ```bash
   python3 eyeContact/test.py
   ```

## Configuration

- To change the camera source, modify the device index in the code:
  ```python
  cap = cv2.VideoCapture(0)  # Change to the correct device index for your camera
  ```

- To change the video file, update the path in the code:
  ```python
  video_path = os.path.abspath('trailer.mp4')  # Replace with your video file path
  ```

- To change the icon image, update the path in the code:
  ```python
  icon_path = os.path.abspath('icon.png')  # Replace with your icon image path
  ```

- To modify the object classes that pause the trailer or reduce opacity, update the lists in the code:
  ```python
  stop_objects = ['car', 'bicycle']  # Add the object classes you want to stop the video
  reduce_opacity_objects = ['person']  # Add the object classes you want to reduce the opacity
  ```
