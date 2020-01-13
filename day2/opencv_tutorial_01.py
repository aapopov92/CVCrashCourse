import imutils
import cv2

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