import cv2
import numpy as np

BOUNDING_BOX_AREA_THRESHOLD = 5000


def find_and_draw_largest_contour(image, image_with_contours, boundin_box_are_threshold):
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

        if area > boundin_box_are_threshold:
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
    rearrange_approx = rearrange_corners(approx)

    src_points = np.float32(rearrange_approx)
    # dst_points =[[0, 0], [w, 0], [w, h], [0, h]]--> A  B  C  D
    dst_points = np.float32([[0, 0], [paper_width, 0], [paper_width, paper_height], [0, paper_height]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, matrix, (paper_width, paper_height))
    return warped_image, rearrange_approx
