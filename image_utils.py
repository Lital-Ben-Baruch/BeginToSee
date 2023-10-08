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



def fix_image_dimensions_to_show(image):
    """
    Checks the dimensions of the input image and provides information.
    Args:
        image (numpy.ndarray): The input image.
    Returns:
        numpy.ndarray: The processed image.
    """
    # Check if the image is grayscale
    if len(image.shape) == 2:
        print("The image is grayscale (1D).")
        image = cv2.merge((image, image, image))
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            print("The image is color (3D) with {} channels.".format(image.shape[2]))
        elif image.shape[2] == 4:
            print("The image is in RGBA format.")
            # Consider converting to RGB if necessary.
            image = image[:, :, :3]
        else:
            print("The image has an unexpected number of channels.")
            return image
    else:
        print("The image has an unexpected number of dimensions.")
        return image

    # Resize and pad if necessary
    if image.shape[0] != IMAGE_HEIGHT or image.shape[1] != IMAGE_WIDTH:
        width_ratio = IMAGE_WIDTH / image.shape[1]
        height_ratio = IMAGE_HEIGHT / image.shape[0]
        ratio = min(width_ratio, height_ratio)

        resized_paper = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))

        delta_w = IMAGE_WIDTH - resized_paper.shape[1]
        delta_h = IMAGE_HEIGHT - resized_paper.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        image = cv2.copyMakeBorder(resized_paper, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

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

