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
    """
        Preprocesses the input image for paper contour detection.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The preprocessed image.
        """
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
    """
        Finds and draws the largest paper contour on the input image.

        Args:
            image (numpy.ndarray): The input image.
            image_with_contours (numpy.ndarray): The image on which contours will be drawn.

        Returns:
            numpy.ndarray: The largest paper contour.
        """
    contour_color = (255, 0, 0)
    contour_thickness = 5
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


def contour_centroid(approx):
    M = cv2.moments(approx)
    points = approx.reshape((4, 2))

    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
    else:
        # Handle the case where the moments are undefined
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))
    return center_x, center_y


def rearrange_corners(approx):
    """
    Rearranges the corners of a paper contour to ensure a consistent order.

    Args:
          approx (numpy.ndarray): The paper contour, represented as a NumPy array of shape (4, 1, 2).

    Returns:
        numpy.ndarray: The rearranged paper contour, with corners sorted in a consistent order.
    """
    # Reshape the input paper contour to a 4x2 array for processing
    points = approx.reshape((4, 2))

    # Calculate the centroid of the paper contour
    center_x, center_y = contour_centroid(approx)

    # Calculate the angles of each corner relative to the centroid
    angles = np.arctan2(points[:, 1] - center_y, points[:, 0] - center_x)

    # Sort the corner points based on their angles in a clockwise order
    sorted_indices = np.argsort(angles)
    sorted_corners = points[sorted_indices]

    # Calculate distances 'a' and 'b' between specific corner points - adjusting paper layout
    # Rearrange the sorted corners to match the desired order
    # dst_points =[[0, 0], [w, 0], [w, h], [0, h]]--> A  B  C  D
    # (0,0)-->(w,0) = b
    # (0,0)-->(0,h) = a
    b = np.sqrt((sorted_corners[1][0] - sorted_corners[0][0]) ** 2 + (sorted_corners[1][1] - sorted_corners[0][1]) ** 2)
    a = np.sqrt((sorted_corners[3][0] - sorted_corners[0][0]) ** 2 + (sorted_corners[3][1] - sorted_corners[0][1]) ** 2)

    if a * 1.2 < b:  # correct the layout of the paper
        approx[[0]] = sorted_corners[3]
        approx[[1]] = sorted_corners[0]
        approx[[2]] = sorted_corners[1]
        approx[[3]] = sorted_corners[2]
    if a * 1.2 > b:
        sorted_corners = sorted_corners.reshape((4, 1, 2))
        approx = sorted_corners

    return approx


def get_warped_image(image, approx):
    """
        Warps the input image using the provided paper contour.

        Args:
            image (numpy.ndarray): The input image.
            approx (numpy.ndarray): The paper contour.

        Returns:
            numpy.ndarray: The warped image.
        """
    paper_width, paper_height = 480, 640
    approx = rearrange_corners(approx)

    src_points = np.float32(approx)
    # dst_points =[[0, 0], [w, 0], [w, h], [0, h]]--> A  B  C  D
    dst_points = np.float32([[0, 0], [paper_width, 0], [paper_width, paper_height], [0, paper_height]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, matrix, (paper_width, paper_height))
    return warped_image


def fix_image_dimensions_to_show(image, image_width, image_height):  # TODO
    """
    Checks the dimensions of the input image and provides information.
    Args:
        image (numpy.ndarray): The input image.
    """
    if image.shape[0] != image_height and image.shape[1] != image_width :
        # Calculate the scaling factor
        width_ratio = 640 / image.shape[1]
        height_ratio = 480 / image.shape[0]
        ratio = min(width_ratio, height_ratio)

        # Resize the image
        resized_paper = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))

        # Calculate padding values
        delta_w = 640 - resized_paper.shape[1]
        delta_h = 480 - resized_paper.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # Pad with zeros (black color)
        color = [0, 0, 0]
        image = cv2.copyMakeBorder(resized_paper, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    if len(image.shape) == 2:
        print("The image is grayscale (1D).")
        image = cv2.merge((image, image, image))
    elif len(image.shape) == 3:
        print("The image is color (3D) with {} channels.".format(image.shape[2]))
    else:
        print("The image has an unexpected number of dimensions.")

    return image


def put_letters_on_corner_points(image, corner_points):
    """
    Adds letters to the corner points of an image contour.

    Args:
        image (numpy.ndarray): The input image.
        corner_points (numpy.ndarray): The corner points of the contour to annotate with letters.

    Returns:
        None. The function modifies the input image in place.
    """
    # Define the letters you want to add
    letters = ['A', 'B', 'C', 'D']

    # Iterate through the corner_points and add the corresponding letter to the image
    for i, point in enumerate(corner_points):
        x, y = point[0]
        letter = letters[i]
        cv2.putText(image, letter, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (204, 102, 0), 2, cv2.LINE_AA)


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

        if np.any(max_area_contour):
            # The array is not empty
            print("max_area_contour", max_area_contour)
            warped_image = get_warped_image(frame, max_area_contour)

            # Adding to image_with_contours letters for the corner points
            put_letters_on_corner_points(image_with_contours, max_area_contour)

        else:
            # The array is empty
            print("Array is empty")
            warped_image = image_with_contours

        cv2.imshow('frame', frame)

        cv2.imshow('image_with_contours', image_with_contours)

        processed_frame = fix_image_dimensions_to_show(processed_frame, IMAGE_WIDTH, IMAGE_HEIGHT)
        warped_image = fix_image_dimensions_to_show(warped_image, IMAGE_WIDTH, IMAGE_HEIGHT)
        cv2.imshow('processed_frame', processed_frame)
        cv2.imshow('warped_image', warped_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close all OpenCV windows
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
