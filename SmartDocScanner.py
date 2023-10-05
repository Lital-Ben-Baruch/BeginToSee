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

BB_AREA = 5000


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


# def find_contour_centroid(contour, frame_to_draw=None):
#     cx, cy = 0, 0
#     M = cv2.moments(contour)
#     if M["m00"] != 0:  # avoid division by zero
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#     if frame_to_draw is not None:
#         cv2.drawContours(frame_to_draw, contour, -1, (0, 0, 0), 3)
#     return cx, cy


def find_and_draw_contours(image, imgContours):  # TODO change the variables names
    contour_color = (255, 0, 0)
    contour_thick = 20
    curve_is_closed = True
    resolution = 0.02
    max_area_approx = np.array([])
    max_area = 0
    # contour_center_x, contour_center_y = 0, 0

    contours, Hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > BB_AREA:

            arc_length = cv2.arcLength(contour, curve_is_closed)
            approx = cv2.approxPolyDP(contour, resolution * arc_length, curve_is_closed)
            if area > max_area and len(approx) == 4:  # len(approx) == 4 should be  4 for a paper.
                #
                max_area_approx = approx
                max_area = area
                # contour_center_x, contour_center_y = find_contour_centroid(contour)
                # # draw contour centroid
                # cv2.circle(image, (contour_center_x, contour_center_y), 8, contour_color,
                #            cv2.FILLED)
    cv2.drawContours(imgContours, max_area_approx, -1, contour_color, contour_thick)
    # contour_center_x, contour_center_y = find_contour_centroid(max_area_approx)
    # cv2.circle(image, (contour_center_x, contour_center_y), 20, contour_color, cv2.FILLED)
    return max_area_approx


def rearrange_points(approx):
    points = approx.reshape((4, 2))
    new_approx = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)
    print("add", add)
    new_approx[0] = points[np.argmin(add)]
    new_approx[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_approx[1] = points[np.argmin(diff)]
    new_approx[2] = points[np.argmax(diff)]
    return new_approx


def get_wrap(img, approx):
    approx = rearrange_points(approx)
    paperWidth, paperHeight = 480, 640
    # We should have:
    # Extracting the 4 corner points of the paper from the 'approx' variable.
    pts1 = np.float32(approx)

    # We should have:
    # Defining specific locations in the image for each of the corner points.
    # However, the 'approx' variable might not provide the points in a specific order,
    # so we may need to rearrange the points.
    pts2 = np.float32([[0, 0], [paperWidth, 0], [0, paperHeight], [paperWidth, paperHeight]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (paperWidth, paperHeight))

    # # crop 20 pixels from each size
    # img_cropped = imgOutput[20:imgOutput.shape[0]-20, 20:imgOutput.shape[1]-20]
    # img_cropped = cv2.resize(img_cropped, (paperWidth, paperHeight))
    # return img_cropped
    return imgOutput
def img_3D_convert(img):
    # Check if the image is grayscale (1D) or color (3D)
    if len(img.shape) == 2:
        print("The image is grayscale (1D).")
        img_3_channels = cv2.merge((img, img, img))
    elif len(img.shape) == 3:
        print("The image is color (3D) with {} channels.".format(img.shape[2]))
        img_3_channels = img
    else:
        print("The image has an unexpected number of dimensions.")
    return img_3_channels

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

        imgContours = frame.copy()
        # Preprocess the frame
        processed_frame = preprocess_image(frame)
        max_area_approx = find_and_draw_contours(processed_frame, imgContours)
        print(max_area_approx)
        # Assuming 'approx' is your NumPy array
        if np.any(max_area_approx):
            # The array is not empty
            print("Array is not empty")
            imgOutput = get_wrap(frame, max_area_approx)
        else:
            # The array is empty
            print("Array is empty")
            imgOutput = imgContours

        cv2.imshow('frame', frame)
        cv2.imshow('processed_frame', processed_frame)
        cv2.imshow('imgContours', imgContours)
        cv2.imshow('Original Image and Detection', imgOutput)

        # # Display the processed frame in a window named "Live Feed"
        # imgOutput_show = img_3D_convert(imgOutput)
        # top = np.hstack([frame, imgOutput_show])
        # cv2.imshow("Live Feed", top)
        # bottom = np.hstack([frame, imgOutput])
        # combined_img = np.vstack([top, bottom])
        # cv2.imshow('Original Image and Detection', combined_img)

        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#
# def find_contour_centroid(contour):
#     cx, cy = 0, 0  # Initialize the center coordinates.
#     # Finding the Centroid:
#     # Calculates moments, which are a set of scalar values that provide information about the image's shape.
#     # The centroid (center) of the contour is calculated using these moments.
#     M = cv2.moments(contour)
#     if M["m00"] != 0:  # avoid division by zero
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#     return cx, cy
#
#
# def process_contours(mask, frame_to_draw=None):
#     cnt_color = (255, 0, 0)  # Blue color for drawing the detected contours.
#     cnt_thick = 3  # Thickness of the contour lines.
#     largest_area = 0  # To keep track of the largest contour's area.
#     cx, cy = 0, 0  # Initialize the center coordinates.
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
#     # Processing Each Contour:
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         # The contour is considered relevant if its area is greater than BB_AREA (to ignore noise) and if its area is
#         # larger than the previous largest contour's area.
#         if area > BB_AREA and area > largest_area:
#             largest_area = area
#             cx, cy = find_contour_centroid(contour)
#             # Visualization -Draw contours if a 'frame_to_draw' is provided
#             if frame_to_draw is not None:
#                 cv2.drawContours(frame_to_draw, contour, -1, cnt_color, cnt_thick)
#
#     # Returns the x and y coordinates of the largest contour's center.
#     return cx, cy
