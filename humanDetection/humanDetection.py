import cv2
import numpy as np

# Load the pre-trained YOLOv3 model
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Load the class labels
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Create a VideoCapture object
cap = cv2.VideoCapture(1)  # 0 represents the default camera, change it if necessary

while True:
    # Read the next frame
    ret, frame = cap.read()

    # Create a blob from the input frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    # Set the input blob for the neural network
    net.setInput(blob)

    # Forward pass through the network
    output_layers = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers)

    # Process the outputs
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0:  # 0 represents the class ID for humans
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                # Calculate the top-left corner coordinates of the bounding box
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes and class labels on the frame
    for i in range(len(boxes)):
        if i in indices and confidences[i] >= 0.9:
            x, y, width, height = boxes[i]
            label = f'{classes[class_ids[i]]}: {confidences[i]:.2f}'
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    # Display the resulting frame
    cv2.imshow('Human Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
