import cv2
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
# Create a VideoCapture object with camera index 0 (default camera)
cap = cv2.VideoCapture(0)
# for my phone
# capture = cv2.VideoCapture("https://....../video")

# Set the width of the frame (ID 3)
cap.set(3, IMAGE_WIDTH)

# Set the height of the frame (ID 4)
cap.set(4, IMAGE_HEIGHT)

# Set the brightness (ID 10)
cap.set(10, 100)

while True:
    # Read a frame from the video capture
    success, img = cap.read()

    # Display the frame in a window named "Live Feed"
    cv2.imshow("Live Feed", img)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
