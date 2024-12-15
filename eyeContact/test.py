import cv2
import torch
import time
import os  # Import the os module

# Load YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Pre-trained YOLOv5s

# Open EpocCam feed (use the correct device index for EpocCam)
cap = cv2.VideoCapture(0)  # Change to the correct device index for EpocCam

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Get the absolute path of the video file
video_path = os.path.abspath('trailer.mp4')

# Open the video file
video_cap = cv2.VideoCapture(video_path)
if not video_cap.isOpened():
    print(f"Error: Could not open video file {video_path}.")
    exit()

# Load the icon image
icon_path = os.path.abspath('icon.png')  # Replace with the path to your icon image
icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)

print("Starting video feed... Press 'q' to quit.")

# List of object classes that will stop the video diffusion
stop_objects = ['car', 'bicycle']  # Add the object classes you want to stop the video
reduce_opacity_objects = ['person']  # Add the object classes you want to reduce the opacity

previous_detections = {}
pause_trailer = False
last_detection_time = 0

while True:
    ret, frame = cap.read()  # Read webcam frame by frame
    if not ret:
        print("Error: Unable to capture video.")
        break

    # Object detection
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Extract bounding box data as a DataFrame

    current_time = time.time()
    if any(detections['name'].isin(stop_objects)):
        pause_trailer = True
        last_detection_time = current_time
    else:
        if current_time - last_detection_time > 5:
            pause_trailer = False

    reduce_opacity = any(detections['name'].isin(reduce_opacity_objects))

    for index, row in detections.iterrows():
        object_id = row['class']  # Use class as object ID
        x_center = (row['xmin'] + row['xmax']) / 2
        y_center = (row['ymin'] + row['ymax']) / 2

        if object_id in previous_detections:
            prev_x, prev_y, prev_time = previous_detections[object_id]
            distance = ((x_center - prev_x) ** 2 + (y_center - prev_y) ** 2) ** 0.5
            time_diff = current_time - prev_time

            # Check if the time difference is not zero to avoid division by zero
            if time_diff != 0:
                speed = distance / time_diff

        previous_detections[object_id] = (x_center, y_center, current_time)

    # Render the results on the frame
    frame = results.render()[0]

    if not pause_trailer:
        # Read video frame by frame
        ret_video, video_frame = video_cap.read()
        if not ret_video:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video if it ends
            ret_video, video_frame = video_cap.read()

        # Resize video frame to a smaller size
        video_frame = cv2.resize(video_frame, (320, 240))  # Resize to 320x240

        # Get dimensions of the webcam frame
        frame_height, frame_width = frame.shape[:2]

        # Calculate the position to center the video frame on the webcam frame
        x_offset = (frame_width - video_frame.shape[1]) // 2
        y_offset = (frame_height - video_frame.shape[0]) // 2

        # Create a copy of the frame to modify
        frame_copy = frame.copy()

        # Reduce opacity if a person is detected
        if reduce_opacity:
            alpha = 0.3  # Set the opacity level (0.0 to 1.0)
        else:
            alpha = 1.0

        # Overlay the video frame onto the webcam frame with reduced opacity
        frame_copy[y_offset:y_offset + video_frame.shape[0], x_offset:x_offset + video_frame.shape[1]] = \
            cv2.addWeighted(frame_copy[y_offset:y_offset + video_frame.shape[0], x_offset:x_offset + video_frame.shape[1]], 1 - alpha, video_frame, alpha, 0)

        # Resize the frame to a larger size
        frame_copy = cv2.resize(frame_copy, (1280, 720))  # Resize to 1280x720

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame_copy)
    else:
        # Resize the frame to a larger size
        frame = cv2.resize(frame, (1280, 720))  # Resize to 1280x720

        # Get dimensions of the webcam frame
        frame_height, frame_width = frame.shape[:2]

        # Resize the icon to a smaller size if needed
        icon = cv2.resize(icon, (50, 50))  # Resize to 50x50

        # Get dimensions of the icon
        icon_height, icon_width = icon.shape[:2]

        # Calculate the position to place the icon on the top right of the webcam frame
        x_offset = frame_width - icon_width - 10  # 10 pixels from the right edge
        y_offset = 10  # 10 pixels from the top edge

        # Overlay the icon onto the webcam frame
        for c in range(0, 3):
            frame[y_offset:y_offset + icon_height, x_offset:x_offset + icon_width, c] = \
                icon[:, :, c] * (icon[:, :, 3] / 255.0) + frame[y_offset:y_offset + icon_height, x_offset:x_offset + icon_width, c] * (1.0 - icon[:, :, 3] / 255.0)

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and video capture, and close windows
cap.release()
video_cap.release()
cv2.destroyAllWindows()
