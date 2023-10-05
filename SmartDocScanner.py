import cv2
import numpy as np

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


def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
    # Apply Canny edge detection
    edges_image = cv2.Canny(blurred_image, CANNY_THRESH1, CANNY_THRESH2)
    # Perform dilation and erosion
    dilated_image = cv2.dilate(edges_image, DILATION_KERNEL, iterations=1)
    eroded_image = cv2.erode(dilated_image, DILATION_KERNEL, iterations=1)
    return eroded_image


def find_and_draw_largest_contour(image, image_with_contours):
    contour_color = (255, 0, 0)
    contour_thickness = 20
    is_closed_curve = True
    approximation_resolution = 0.02
    max_area_contour = np.array([])
    max_area = 0

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > BOUNDING_BOX_AREA_THRESHOLD:
            arc_length = cv2.arcLength(contour, is_closed_curve)
            approx = cv2.approxPolyDP(contour, approximation_resolution * arc_length, is_closed_curve)
            if area > max_area and len(approx) == 4:  # 4 corners for a paper
                max_area_contour = approx
                max_area = area

    cv2.drawContours(image_with_contours, max_area_contour, -1, contour_color, contour_thickness)
    return max_area_contour


def rearrange_corners(approx):
    points = approx.reshape((4, 2))
    new_approx = np.zeros((4, 1, 2), np.int32)
    corner_sum = points.sum(1)

    new_approx[0] = points[np.argmin(corner_sum)]
    new_approx[3] = points[np.argmax(corner_sum)]
    corner_diff = np.diff(points, axis=1)
    new_approx[1] = points[np.argmin(corner_diff)]
    new_approx[2] = points[np.argmax(corner_diff)]

    return new_approx


def get_warped_image(image, approx):
    approx = rearrange_corners(approx)
    paper_width, paper_height = 480, 640
    src_points = np.float32(approx)
    dst_points = np.float32([[0, 0], [paper_width, 0], [0, paper_height], [paper_width, paper_height]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, matrix, (paper_width, paper_height))
    return warped_image


def check_image_dimensions(image):
    if len(image.shape) == 2:
        print("The image is grayscale (1D).")
        image = cv2.merge((image, image, image))
    elif len(image.shape) == 3:
        print("The image is color (3D) with {} channels.".format(image.shape[2]))
    else:
        print("The image has an unexpected number of dimensions.")


def main():
    # Create a VideoCapture object with camera index 0 (default camera)
    capture = cv2.VideoCapture(0)

    # Set the width of the frame (ID 3)
    capture.set(3, IMAGE_WIDTH)
    # Set the height of the frame (ID 4)
    capture.set(4, IMAGE_HEIGHT)
    # Set the brightness (ID 10)
    capture.set(10, 100)

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
        print(max_area_contour)

        if np.any(max_area_contour):
            # The array is not empty
            print("Array is not empty")
            warped_image = get_warped_image(frame, max_area_contour)
        else:
            # The array is empty
            print("Array is empty")
            warped_image = image_with_contours

        cv2.imshow('frame', frame)
        cv2.imshow('processed_frame', processed_frame)
        cv2.imshow('image_with_contours', image_with_contours)
        cv2.imshow('warped_image', warped_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close all OpenCV windows
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
