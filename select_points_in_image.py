import cv2
import numpy as np

# Global variables
points = []
num_points = 4


def select_point(event, x, y, flags, param):
    global points, num_points, img

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        points.append((x, y))
        cv2.putText(img, str((x, y)), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow('Image', img)


img = cv2.imread("Resources/cards.jpg")  # Replace with your image path
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', select_point)

while True:
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(points) == num_points:
        break

# Display the image with selected points
for pt in points:
    cv2.circle(img, pt, 5, (0, 0, 255), -1)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Printing the selected points
for pt in points:
    print(pt)
