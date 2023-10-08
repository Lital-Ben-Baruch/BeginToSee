import cv2
import numpy as np
from image_utils import preprocess_image, fix_image_dimensions_to_show, put_letters_on_corner_points
from contour_utils import find_and_draw_largest_contour, get_warped_image

# Constants for image dimensions
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Preprocessing parameters
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 1
CANNY_THRESH1 = 50
CANNY_THRESH2 = 50
DILATION_KERNEL = np.ones((5, 5), np.uint8)
BOUNDING_BOX_AREA_THRESHOLD = 5000


def main():
    # Create a VideoCapture object with camera index 0 (default camera)
    capture = cv2.VideoCapture(0)

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
        max_area_contour = find_and_draw_largest_contour(processed_frame, image_with_contours,
                                                         BOUNDING_BOX_AREA_THRESHOLD)

        if np.any(max_area_contour):
            # The array is not empty
            # print("max_area_contour", max_area_contour)
            warped_image, rearrange_approx = get_warped_image(frame, max_area_contour)

            # Adding to image_with_contours letters for the corner points
            put_letters_on_corner_points(image_with_contours, rearrange_approx)

        else:
            # The array is empty
            print("Array is empty")
            text = 'No paper contour detected yet.'
            # Get the size of the text
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            # Calculate the center position
            x = (IMAGE_WIDTH - text_width) // 2
            y = (IMAGE_HEIGHT + text_height) // 2
            blank_image_warp = blank_image.copy()
            cv2.putText(blank_image_warp, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        lineType=cv2.LINE_AA)
            warped_image = blank_image_warp
            blank_image_contour = blank_image.copy()
            text_1 = 'No paper contour detected yet.'
            cv2.putText(blank_image_contour, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        lineType=cv2.LINE_AA)
            image_with_contours = blank_image_contour

        image_list = [frame, image_with_contours, processed_frame, warped_image]
        titles = ["Original", "Contours", "Processed", "Warped"]
        for idx, image in enumerate(image_list):
            image_list[idx] = fix_image_dimensions_to_show(image, IMAGE_WIDTH, IMAGE_HEIGHT)
            cv2.putText(image_list[idx], titles[idx], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                        2)  # Adjust position, font, size, color, thickness as needed

        top_row = np.hstack([image_list[0], image_list[1]])
        bottom_row = np.hstack([image_list[2], image_list[3]])
        combined_img = np.vstack([top_row, bottom_row])

        cv2.imshow("All Images", combined_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close all OpenCV windows
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
