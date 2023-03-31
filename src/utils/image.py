import cv2
import numpy as np


def filter_colors(image_path, colors, tolerance=10):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Create a blank image with the same dimensions as the input image
    result = np.zeros_like(img)

    # Loop through the color set
    for color in colors:
        # Define the lower and upper boundaries for each color
        lower = np.array([c - tolerance for c in color])
        upper = np.array([c + tolerance for c in color])

        # Create a mask for the current color in the image
        mask = cv2.inRange(img, lower, upper)

        # Combine the mask with the result image
        result = cv2.add(result, cv2.bitwise_and(img, img, mask=mask))

    return result


def plot_arc(image, start_point, end_point, arc_center, arc_radius, color=(0, 255, 0), thickness=6):
    arc_start_angle = np.rad2deg(np.arctan2(
        start_point[1] - arc_center[1], start_point[0] - arc_center[0]))
    arc_end_angle = np.rad2deg(np.arctan2(
        end_point[1] - arc_center[1], end_point[0] - arc_center[0]))
    cv2.ellipse(image, (int(arc_center[0]), int(
        arc_center[1])), (int(arc_radius), int(arc_radius)), 0, arc_start_angle, arc_end_angle, color, thickness)
