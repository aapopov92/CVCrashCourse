# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# import the necessary packages
from imutils import face_utils
from imutils.video import VideoStream
from random import randint
import time
import numpy as np
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

vs = VideoStream(src=0).start()
time.sleep(2.0)
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	black = np.zeros(frame.shape, np.uint8)
	# detect faces in the frame
	rects = detector(frame, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(frame, rect)
		shape = face_utils.shape_to_np(shape)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
		
		#color = (randint(10, 255), randint(10, 255), randint(10, 255))
		color = (0, 255, 0)
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(black, (x, y), 1, color, 2)
		result = np.concatenate((frame, black), axis=1)

	cv2.imshow("Frame", result)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

vs.stop()
cv2.destroyAllWindows()
