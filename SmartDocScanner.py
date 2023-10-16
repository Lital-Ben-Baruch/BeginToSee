import cv2
import numpy as np
from image_utils import preprocess_image, put_letters_on_corner_points, generate_centered_text_image,\
    stack_images_to_show
from contour_utils import find_and_draw_largest_contour, get_warped_image

# Constants for image dimensions
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def main():
    # Create a VideoCapture object with camera index 0 (default camera)
    capture = cv2.VideoCapture(0)
    # for my phone
    # capture = cv2.VideoCapture("https://....../video")

    # Set the width of the frame (ID 3)
    capture.set(3, IMAGE_WIDTH)
    # Set the height of the frame (ID 4)
    capture.set(4, IMAGE_HEIGHT)
    # Set the brightness (ID 10)
    capture.set(10, 100)

    # Create a blank image
    background_color = (0, 0, 0)  # Black
    blank_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    blank_image[:] = background_color
    # Place the text on the image

    while True:
        # Read a frame from the video capture
        ret, frame = capture.read()
        # Resize frame if necessary
        height, width, channels = frame.shape
        if height != IMAGE_HEIGHT or width != IMAGE_WIDTH:
            frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

        image_with_contours = frame.copy()
        # Preprocess the frame
        processed_frame = preprocess_image(frame)
        max_area_contour = find_and_draw_largest_contour(processed_frame, image_with_contours)

        if np.any(max_area_contour):
            # The array is not empty
            # print("max_area_contour", max_area_contour)
            warped_image, rearrange_approx = get_warped_image(frame, max_area_contour)

            # Adding to image_with_contours letters for the corner points
            put_letters_on_corner_points(image_with_contours, rearrange_approx)

        else:
            # The array is empty
            # print("Array is empty")
            text = 'No paper detected yet.'
            warped_image = generate_centered_text_image(text)

            text_1 = 'No paper contour detected yet.'
            image_with_contours = generate_centered_text_image(text_1)

        image_list = [frame, image_with_contours, processed_frame, warped_image]
        titles = ["Original Feed", "Contours Detection", "Edge Detection", "Warped Paper Image"]

        combined_img = stack_images_to_show(image_list, titles)
        cv2.imshow("All Images", combined_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close all OpenCV windows
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
