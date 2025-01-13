
import cv2

import numpy as np
'''
# Read image of planets
planets = cv2.imread('beans.jpg')

# Convert image to grayscale
gray_img = cv2.cvtColor(planets, cv2.COLOR_BGR2GRAY)

# Apply median blur to the grayscale image
img = cv2.medianBlur(gray_img, 5)

# Convert blurred grayscale image back to BGR
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Detect circles in the image using HoughCircles
# Parameters:
# - img: the input image
# - cv2.HOUGH_GRADIENT: the detection method
# - 1: the inverse ratio of the accumulator resolution to the image resolution
# - 120: minimum distance between the centers of detected circles
# - param1: higher threshold for the Canny edge detector
# - param2: threshold for center detection
# - minRadius: minimum circle radius
# - maxRadius: maximum circle radius
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 120, param1=800, param2=30, minRadius=0, maxRadius=0)

# Round circle parameters and convert to unsigned 16-bit integers
circles = np.uint16(np.around(circles))

# Loop through each detected circle
for i in circles[0, :]:
    # Draw the outer circle
    # Parameters:
    # - planets: the image to draw on
    # - (i[0], i[1]): the center coordinates of the circle
    # - i[2]: the radius of the circle
    # - (0, 255, 0): the color of the circle (green)
    # - 6: the thickness of the circle
    cv2.circle(planets, (i[0], i[1]), i[2], (0, 255, 0), 6)

    # Draw the center of the circle
    # Parameters:
    # - planets: the image to draw on
    # - (i[0], i[1]): the center coordinates of the circle
    # - 2: the radius of the center point
    # - (0, 0, 255): the color of the center point (red)
    # - 3: the thickness of the center point
    cv2.circle(planets, (i[0], i[1]), 2, (0, 0, 255), 3)

# Display image with detected circles
cv2.imshow("HoughCircles", planets)

# Wait for key press indefinitely
cv2.waitKey()

# Close all OpenCV windows
cv2.destroyAllWindows()'''

# Read image.
img = cv2.imread('beanlid.jpg', cv2.IMREAD_COLOR)

# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(gray, (1, 1))



# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray_blurred,
                                    cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                    param2=30, minRadius=20, maxRadius=2000)


# Draw circles that are detected.
if detected_circles is not None:
    circles = []

    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))

    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        circles.append((a, b, r))
    avg_a = 0
    avg_b = 0
    avg_r = 0

    for circle in circles:
        if circle == circles[0]:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (circle[0], circle[1]), circle[2], 255, -1)

            # Apply the mask
            masked_image = cv2.bitwise_and(img, img, mask=mask)
        if circle == circles[-1]:
            avg_a = avg_a/len(circles)
            avg_b = avg_b/ len(circles)
            avg_r = avg_r/ len(circles)

            cv2.circle(masked_image, (int(avg_a), int(avg_b)), int(avg_r), (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(masked_image, (int(avg_a), int(avg_b)), 1, (0, 0, 255), 3)
            cv2.imshow("Detected Circle", masked_image)
            cv2.waitKey(0)
    else:
            a, b, r = circle[0], circle[1], circle[2]
            avg_a += a
            avg_b += b
            avg_r += r
