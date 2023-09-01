import cv2
import time

# Load the pre-trained face detection model
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize variables for FPS calculation
prev_time = time.time()
fps = 0
fps_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Flip the frame
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces on the frame
    face_coordinates = trained_face_data.detectMultiScale(gray)

    for (x, y, w, h) in face_coordinates:
        # Draw a rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # Calculate and display FPS
    curr_time = time.time()
    fps_count += 1

    if curr_time - prev_time >= 1:
        fps = fps_count / (curr_time - prev_time)
        prev_time = curr_time
        fps_count = 0

    # Display FPS on each frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
