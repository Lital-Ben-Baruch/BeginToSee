import cv2
import numpy as np


def preprocess_image(image, gaussian_kernel_size=(5, 5), gaussian_sigma=1, canny_thresh1=50, canny_thresh2=50,
                     dilation_kernel=np.ones((5, 5), np.uint8)):
    """
    Preprocesses the input image for paper contour detection.

    Args:
        image (numpy.ndarray): The input image.
        gaussian_kernel_size (tuple): Size of the Gaussian kernel for blurring.
        gaussian_sigma (int): Standard deviation of the Gaussian kernel.
        canny_thresh1 (int): First threshold for the Canny edge detector.
        canny_thresh2 (int): Second threshold for the Canny edge detector.
        dilation_kernel (numpy.ndarray): The kernel for dilation and erosion.

    Returns:
        numpy.ndarray: The preprocessed image.
    """
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, gaussian_kernel_size, gaussian_sigma)
    # Apply Canny edge detection
    edges_image = cv2.Canny(blurred_image, canny_thresh1, canny_thresh2)
    # Perform dilation and erosion
    dilated_image = cv2.dilate(edges_image, dilation_kernel, iterations=1)
    eroded_image = cv2.erode(dilated_image, dilation_kernel, iterations=1)
    return eroded_image


def fix_image_dimensions_to_show(image, image_width, image_height):
    """
    Checks the dimensions of the input image and provides information.

    Args:
        image (numpy.ndarray): The input image.
        image_width (int): The desired width of the output image.
        image_height (int): The desired height of the output image.

    Returns:
        numpy.ndarray: The processed image.
    """
    # Check if the image is grayscale
    if len(image.shape) == 2:
        # Convert grayscale to color (3 channels)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # The image is in RGBA format; consider converting to RGB if necessary.
        image = image[:, :, :3]
    elif len(image.shape) != 3 or image.shape[2] != 3:
        print("The image has an unexpected number of channels or dimensions.")
        return image

    # Resize and pad if necessary
    if image.shape[0] != image_height or image.shape[1] != image_width:
        width_ratio = image_width / image.shape[1]
        height_ratio = image_height / image.shape[0]
        ratio = min(width_ratio, height_ratio)

        resized_paper = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))

        delta_w = image_width - resized_paper.shape[1]
        delta_h = image_height - resized_paper.shape[0]
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
