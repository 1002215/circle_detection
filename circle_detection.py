# Emma Chetan PWP Detect Circumference and Center Point of a Can
import cv2
import numpy as np

# Read image.
img = cv2.imread('beans.jpg', cv2.IMREAD_COLOR)


height, width, _ = img.shape

# Resize the image to make it work with the src_pts and dts_pts later on using perspective transform.
resize_factor = 0.8

new_width = int(width * resize_factor)

new_height = int(height * resize_factor)

img = cv2.resize(img, (new_width, new_height))

height, width, _ = img.shape

# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# https://www.geeksforgeeks.org/perspective-transformation-python-opencv/
src_pts = np.float32([
        [width * 0.15, height * 0.1],  
        [width * 0.85, height * 0.1],  
        [width * 0.87, height * 0.8],  
        [width * 0.13, height * 0.8]   
    ])
dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

# Matrix of perspective transform.
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

warped_frame = cv2.warpPerspective(gray, M, (img.shape[1], img.shape[0]))


# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(warped_frame, (3, 3))

canny = cv2.Canny(gray_blurred, 50, 150)

#https://www.geeksforgeeks.org/circle-detection-using-opencv-python/ 
# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(canny,
                                    cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                    param2=30, minRadius=1, maxRadius=4000)

# Draw the circle with the biggest circumference. Also draw its center.
if detected_circles is not None:

    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))

    # Draw the largest possible circle, in this case the first detected circle
    for pt in detected_circles[0]:
        a, b, r = pt[0], pt[1], pt[2]

        # Draw the circumference of the circle.
        cv2.circle(warped_frame, (a, b), r, (0, 255, 0), 2)

        # Draw a small circle to show the center.
        cv2.circle(warped_frame, (a, b), 1, (0, 0, 255), 3)

        # Inverse of the perspective transform matrix from earlier.
        M_inv = np.linalg.inv(M)

        # Center of the circle.
        center = np.array([[[a, b]]], dtype=np.float32)

        # Apply perspective transform to the center of the circle using the inverse matrix.
        transformed_center = cv2.perspectiveTransform(center, M_inv)

        # Draw the transformed circle back in the original image.
        center_x, center_y = int(transformed_center[0][0][0]), int(transformed_center[0][0][1])

        # Transform the radius as well.
        radius_transformed = int(r * np.linalg.norm(M_inv[0, :2]))

        # Draw the circle outline back in the original image.
        cv2.circle(img, (center_x, center_y), radius_transformed, (0, 255, 0), 2)
        cv2.circle(img, (center_x, center_y), 1, (0, 0, 255), 3)  # Center of the circle


        cv2.imshow("Emma Chetan Circumference and Center", img)

        cv2.waitKey(0)

        break

# End the program after pressing '0'
cv2.destroyAllWindows()
