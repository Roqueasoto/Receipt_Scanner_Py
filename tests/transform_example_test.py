# This is an example function to help test the transform module. Example
# is taken from the www.pyimagesearch.com 4 Point OpenCV getPerspective
# Transform example.

# First import the necessary packages.
from record.scan.transform import four_point_transform
import numpy as np
import argparse
import cv2

# Construct the argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-c", "--coords", help="comma separated list of source points")
args = vars(ap.parse_args())

# Load the image and grab the source coordinates (list of x, y points).
image = cv2.imread(args["image"])
pts = np.array(eval(args["coords"]), dtype="float32")

# Apply four_point_transform to get birds-eye view of image.
new_img = four_point_transform(image, pts)

# Show original and new images and then wait for input.
cv2.imshow("Original", image)
cv2.imshow("New", new_img)
cv2.waitKey(0)
