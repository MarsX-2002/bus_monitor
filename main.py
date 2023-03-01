import cv2
import numpy as np

# Load MobileNet SSD model
model = cv2.dnn.readNetFromTensorflow('path/to/mobilenet_ssd.pb', 'path/to/mobilenet_ssd.pbtxt')

# Define classes to be detected
classes = ["background", "person", "bus_card"]

# Load input video from file or camera
cap = cv2.VideoCapture('path/to/input/video.mp4')
# OR
cap = cv2.VideoCapture(0)  # Use camera

# Set up output video writer
output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('path/to/output/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (output_width, output_height))

# Loop over frames in input video
while cap.isOpened():
    # Read frame from input video
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess input frame
    frame_blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
    model.setInput(frame_blob)

    # Detect objects using MobileNet SSD
    output = model.forward()

    # Loop over detected objects
    num_paid = 0
    num_unpaid = 0
    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > 0.5:  # Only show detections with high confidence
            class_id = int(detection[1])
            class_label = classes[class_id]
            x1, y1, x2, y2 = (detection[3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)

            # Check if passenger paid or not
            if class_label == 'bus_card':
                num_paid += 1
                color = (0, 255, 0)  # Green
            else:
                num_unpaid += 1
                color = (0, 0, 255)  # Red
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
            cv2.putText(frame, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display and write output frame
    cv2.imshow('Output', frame)
    out.write(frame)

    # Check for key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release video resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Print number of paid and unpaid passengers
print(f'Number of paid passengers: {num_paid}')
print(f'Number of unpaid passengers: {num_unpaid}')
