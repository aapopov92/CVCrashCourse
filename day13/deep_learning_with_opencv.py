# import packages
import numpy as np
import argparse
import time
import cv2


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
ap.add_argument('-p', '--prototxt', required=True, help='path to Caffe prototxt file')
ap.add_argument('-m', '--model', required=True, help='path to Caffe pre-trained model')
ap.add_argument('-l', '--labels', required=True, help='path to ImageNet labels')
args = vars(ap.parse_args())

# load image from disk
image = cv2.imread(args["image"])

# load the classlabels from disk
with open(args["labels"]) as file:
	rows = file.read().strip().split('\n')
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# our CNN requires fixed spatial dimensions for our input image(s)
# so we need to ensure it is resized to 224x224 pixels while
# performing mean subtraction (104, 117, 123) to normalize the input;
# after executing this command our "blob" now has the shape:
# (1, 3, 224, 224)
blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

# load model from disk
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] classification took {:.5} seconds".format(end - start))

# sort the indexes of the probabilities in descending order (higher
# probabilitiy first) and grab the top-5 predictions
idxs = np.argsort(preds[0])[::-1][:5]

for i, idx in enumerate(idxs):
	if i == 0:
		text = f'Label: {classes[idx]}, {preds[0][idx] * 100}'
		cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)
	# display the predicted label + associated probability to the
	# console	
	print(f"[INFO] {i + 1}. label: {classes[idx]}, probability: {preds[0][idx]}")

# display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)