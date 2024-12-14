import cv2
import torch
import time
import os  # Import the os module

# Load YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Pre-trained YOLOv5s

# Open the main video file instead of the webcam
video_path_street = os.path.abspath('street.mp4')  # Replace with the correct path to your video file
cap = cv2.VideoCapture(video_path_street)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path_street}.")
    exit()

# Open the trailer video file
video_path_trailer = os.path.abspath('trailer.mp4')
video_cap = cv2.VideoCapture(video_path_trailer)
if not video_cap.isOpened():
    print(f"Error: Could not open video file {video_path_trailer}.")
    exit()

# Load the icon image
icon_path = os.path.abspath('icon.png')  # Replace with the path to your icon image
icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)

# Check if the icon image is loaded correctly
if icon is None:
    print(f"Error: Could not load icon image from {icon_path}")
    exit()

print("Starting video feed... Press 'q' to quit.")

# List of object classes that will stop the video diffusion
stop_objects = ['bicycle']  # Add the object classes you want to stop the video
reduce_opacity_objects = ['car', 'person']  # Add the object classes you want to reduce the opacity

previous_detections = {}
pause_trailer = False
last_detection_time = 0

# Create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while True:
    ret, frame = cap.read()  # Read main video frame by frame
    if not ret:
        print("Error: Unable to capture video.")
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Remove shadows (shadows are detected as gray, so we threshold to remove them)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

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

    # Count the number of people detected
    num_people = len(detections[detections['name'] == 'person'])

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

    # Render the results on the frame (draw bounding boxes)
    results.render()

    # Create a writable copy of the frame
    frame_copy = results.ims[0].copy()  # Use the rendered frame with bounding boxes

    if not pause_trailer:
        # Read trailer video frame by frame
        ret_video, video_frame = video_cap.read()
        if not ret_video:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video if it ends
            ret_video, video_frame = video_cap.read()

        # Resize trailer video frame to a smaller size
        video_frame = cv2.resize(video_frame, (320, 240))  # Resize to 320x240

        # Get dimensions of the main video frame
        frame_height, frame_width = frame.shape[:2]

        # Calculate the position to center the trailer video frame on the main video frame
        x_offset = (frame_width - video_frame.shape[1]) // 2
        y_offset = (frame_height - video_frame.shape[0]) // 2

        # Reduce opacity if a person is detected
        if reduce_opacity:
            alpha = 0.3  # Set the opacity level (0.0 to 1.0)
        else:
            alpha = 1.0

        # Overlay the trailer video frame onto the main video frame with reduced opacity
        frame_copy[y_offset:y_offset + video_frame.shape[0], x_offset:x_offset + video_frame.shape[1]] = \
            cv2.addWeighted(frame_copy[y_offset:y_offset + video_frame.shape[0], x_offset:x_offset + video_frame.shape[1]], 1 - alpha, video_frame, alpha, 0)

    if pause_trailer:
        # Get dimensions of the main video frame
        frame_height, frame_width = frame.shape[:2]

        # Resize the icon to a smaller size if needed
        icon_resized = cv2.resize(icon, (50, 50))  # Resize to 50x50

        # Get dimensions of the icon
        icon_height, icon_width = icon_resized.shape[:2]

        # Calculate the position to place the icon on the top right of the main video frame
        x_offset = frame_width - icon_width - 10  # 10 pixels from the right edge
        y_offset = 10  # 10 pixels from the top edge

        # Overlay the icon onto the main video frame
        for c in range(0, 3):
            frame_copy[y_offset:y_offset + icon_height, x_offset:x_offset + icon_width, c] = \
                icon_resized[:, :, c] * (icon_resized[:, :, 3] / 255.0) + frame_copy[y_offset:y_offset + icon_height, x_offset:x_offset + icon_width, c] * (1.0 - icon_resized[:, :, 3] / 255.0)

    # Display the number of people detected at the bottom right corner
    text = f"People: {num_people}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = frame.shape[0] - 10
    cv2.putText(frame_copy, text, (text_x, text_y), font, font_scale, font_color, thickness)

    # Display the resulting frame
    cv2.imshow('Street Video Feed', frame_copy)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video captures and close windows
cap.release()
video_cap.release()
cv2.destroyAllWindows()