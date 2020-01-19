import imutils
import cv2
import numpy as np
from time import sleep
# load the input image and show its dimensions, keeping in mind that
# images are represented as a multi-dimensional NumPy array with
# shape no. rows (height) x no. columns (width) x no. channels (depth)

image = cv2.imread("jp.png")
h, w, d = image.shape
print(f"Width={w}, height={h}, depth={d}")

# display the image to our screen -- we will need to click the window
# open by OpenCV and press a key on our keyboard to continue execution
cv2.imshow("Tratata", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# access the RGB pixel located at x=50, y=100, keepind in mind that
# OpenCV stores images in BGR order rather than RGB
B, G, R = image[100, 50]
print(f"R={R}, G={G}, B={B}")

# extract a 100x100 pixel square ROI (Region of Interest) from the
# input image starting at x=320,y=60 at ending at x=420,y=160
ROI = image[60:160, 320:420]
cv2.imshow("ROI", ROI)
cv2.waitKey(0)
cv2.destroyAllWindows()

# resize the image to 200x200px, ignoring aspect ratio
resized = cv2.resize(image, (200, 200))
cv2.imshow("Resized", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# fixed resizing and distort aspect ratio so let's resize the width
# to be 300px but compute the new height based on the aspect ratio
r = 300.0 / w
dimesions = (300, int(h * r))
resized = cv2.resize(image, dimesions)
cv2.imshow("Resized with aspect ratio", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# manually computing the aspect ratio can be a pain so let's use the
# imutils library instead
resized = imutils.resize(image, width=300)
cv2.imshow("Resized by imutils", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# let's rotate an image 45 degrees clockwise using OpenCV by first
# computing the image center, then constructing the rotation matrix,
# and then finally applying the affine warp
center = w // 2, h // 2
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# rotation can also be easily accomplished via imutils with less code
rotated = imutils.rotate(image, 45)
cv2.imshow("Rotated by imutils", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# OpenCV doesn't "care" if our rotated image is clipped after rotation
# so we can instead use another imutils convenience function to help
# us out
rotated = imutils.rotate_bound(image, 45)
cv2.imshow("Rotated by imutils", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

for angle in np.arange(0, 360, 4):
	rotated = imutils.rotate(image, angle)
	cv2.imshow("Rotated (Problematic)", rotated)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# apply a Gaussian blur with a 11x11 kernel to the image to smooth it,
# useful when reducing high frequency noise
blur = cv2.GaussianBlur(image, (11,11), 0)
cv2.imshow("Gaussian Blur", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# draw a 2px thick red rectangle surrounding the face
output = image.copy()
rectangle = cv2.rectangle(output, (320, 60), (420, 160), (0, 0, 255), 2)
cv2.imshow("Rectangle", rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

# draw a blue 20px (filled in) circle on the image centered at
# x=300,y=150
output = image.copy()
cv2.circle(output, (300, 150), 20, (255, 0, 0), -1)
cv2.imshow("Circle", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# draw a 5px thick red line from x=60,y=20 to x=400,y=200
output = image.copy()
cv2.line(output, (60, 20), (400, 200), (0, 255, 0), 3)
cv2.imshow("Line", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# draw green text on the image
output = image.copy()
cv2.putText(output, "OpenCV + Jurassic Park!!!", (300, 150), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.imshow("Text", output)
cv2.waitKey(0)
cv2.destroyAllWindows()






