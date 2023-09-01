import cv2

# Load the pre-trained face detection model
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Reading an image
image = cv2.imread('img/friends.jpeg')

scale_percent = 40  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize image
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Convert the image to grayscale for face detection
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Detect faces on the image
# coordinates : [[x=100 y=40 w=250 h=250]]
face_coordinates = trained_face_data.detectMultiScale(gray)

for (x, y, w, h) in face_coordinates:
    # Draw a rectangle
    cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), 1)

# Displaying the image
cv2.imshow('Face Detection', resized)

# waits for user to press 'q' key and then close the window
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# closing all open windows
cv2.destroyAllWindows()
