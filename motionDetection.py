import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture(1)  # 0 represents the default camera, change it if necessary

# Read the first frame
ret, frame1 = cap.read()

# Convert the frame to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Define a motion detection threshold
threshold = 10000

while True:
    # Read the next frame
    ret, frame2 = cap.read()

    # Convert the frame to grayscale
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the current and previous frame
    frame_diff = cv2.absdiff(gray1, gray2)

    # Apply a threshold to the frame difference
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Apply a series of morphological operations to remove noise
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=2)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    # Iterate over the contours and find if motion is above the threshold
    for contour in contours:
        if cv2.contourArea(contour) > threshold:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion_detected = True

    # Display the resulting frame
    cv2.imshow("Motion Detection", frame2)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update the previous frame
    gray1 = gray2

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
