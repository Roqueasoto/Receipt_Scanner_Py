# This is an example function to help test the transform module. Example
# is taken from the www.pyimagesearch.com 4 Point OpenCV getPerspective
# Transform example.

# First import necessary modules.
import sys
import numpy as np
import argparse
import cv2

# Now append main directory to path and import transform function.
sys.path.append('C:\\Users\\roque\\Documents\\Projects\\Receipt_Scanner_Py')
from record.scan.transform import four_point_transform

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

# Below is a list of command line commands to execute this example
# script with.
#
# python transform_example.py --image images/example_01.png --coords
# "[(73, 239), (356, 117), (475, 265), (187, 443)]"
#
# python transform_example.py --image images/example_02.png --coords
# "[(101, 185), (393, 151), (479, 323), (187, 441)]"
#
# python transform_example.py --image images/example_03.png --coords
# "[(63, 242), (291, 110), (361, 252), (78, 386)]"
