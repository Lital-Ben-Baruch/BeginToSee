import cv2
import numpy as np

# Constants for image dimensions
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Preprocessing parameters
KERNEL_SIZE = (5, 5)
SIGMA = 1
THRESH_CANNY1 = 50
THRESH_CANNY2 = 50
KERNEL = np.ones((5, 5), np.uint8)


def preprocess_image(img):
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, KERNEL_SIZE, SIGMA)
    # Apply Canny edge detection
    img_edges = cv2.Canny(img_blur, THRESH_CANNY1, THRESH_CANNY2)
    # Perform dilation and erosion
    img_dilation = cv2.dilate(img_edges, KERNEL, iterations=1)
    img_erosion = cv2.erode(img_dilation, KERNEL, iterations=1)
    return img_erosion


def main():
    # Create a VideoCapture object with camera index 0 (default camera)
    cap = cv2.VideoCapture(0)

    # Set the width of the frame (ID 3)
    cap.set(3, IMAGE_WIDTH)

    # Set the height of the frame (ID 4)
    cap.set(4, IMAGE_HEIGHT)

    # Set the brightness (ID 10)
    cap.set(10, 100)

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        # Resize frame if necessary
        height, width, channels = frame.shape
        if height != IMAGE_HEIGHT or width != IMAGE_WIDTH:
            frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # Preprocess the frame
        processed_frame = preprocess_image(frame)

        # Display the processed frame in a window named "Live Feed"
        cv2.imshow("Live Feed", processed_frame)

        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
