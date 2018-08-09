# Import the necessary packages - Note, for skimage, scikit-image and
# Cython need to be installed to the python environment. For imutils,
# you must install the package by the same name.
from skimage.filters import threshold_local
import numpy
import cv2
import imutils
import sys

sys.path.append('C:\\Users\\roque\\Documents\\Projects\\Receipt_Scanner_Py')
from record.scan.transform import four_point_transform


# Set Constants
# New height for image to reshape to while edge finding.
NEW_IMG_HEIGHT = 500.0
# X and Y dimensions of the Gaussian filter kernel. Must be odd.
GAUSS_KERN = 5
# This sigma x value sets Gaussian filter to calculate standard dev.
# from the dimensions of the kernel as defined above.
SIGMA_X = 0
# Threshold limits for Canny edge detection
THRESHOLD_BOT = 75
THRESHOLD_TOP = 200
# Tuple for the color of the contour draw as (int, int, int)
CONT_COLOR = (0, 255, 0)
# Thickness of draw contour.
CONT_THICK = 2
# Threshold block size for threshold_local. Must be odd.
BLOCK_SIZE = 51
# Offset to subtract from weighted mean for threshold_local.
OFFSET = 10
# Method to determine adaptive threshold for threshold_local.
METHOD = "gaussian"
# Final image height of both the original and scanned images.
FIN_IMG_HEIGHT = 650


def doc_scan(image_path):
    """
    Converts an image of a document into a top-down scan of it.

    This function uses the imutils, numpy, and opencv python library to
    take a path to a given image and automatically detect the corners of
    the document within the picture. The picture is then reshaped and
    transformed using the four_point_transform function of transform.py
    to an image of just the document. This is adapted from the
    www.pyimagesearch.com python scanner tutorial.

    The document within the picture must be a rectangle with 4 corners,
    otherwise contour selection will return an error.

    :param image_path:
        Path to the image to be scanned.
    :return: final_image -
        Top-down, full document version of the original image as an
        opencv image object.
    """

    # STEP 1 : Edge Detection
    # Load the image, get ratio of old to new height, clone & resize it.
    image = cv2.imread(image_path)
    ratio = image.shape[0] / NEW_IMG_HEIGHT
    original = image.copy()
    image = imutils.resize(image, height=int(NEW_IMG_HEIGHT))

    # convert to gray-scale, blur, and find edges.
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (GAUSS_KERN, GAUSS_KERN), SIGMA_X)
    img_edges = cv2.Canny(gray_img, THRESHOLD_BOT, THRESHOLD_TOP)

    # STEP 2 : Detecting Contours
    # Find the contours in the edge image, keep the largest contours, &
    # initialize the screen contour.
    contours = cv2.findContours(img_edges.copy(), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Loop over contours
    for cont in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(cont, True)
        approx_cont = cv2.approxPolyDP(cont, 0.02 * perimeter, True)

        # If the approximated contour has 4 pts, assume it's the screen.
        if len(approx_cont) == 4:
            screen_cont = approx_cont
            break
    else:
        raise Exception("Document not found, please try again")

    # STEP 3 : Apply Perspective Transform & Threshold
    # Apply four_point_transform to get top-down view of original image.
    top_down = four_point_transform(
        original, screen_cont.reshape(4, 2) * ratio
    )

    # Convert top-down image to gray-scale, and threshold it to give the
    # effect of a 'black and white' copy.
    top_down = cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY)
    threshold = threshold_local(
        top_down, BLOCK_SIZE, offset=OFFSET, method=METHOD
    )
    top_down = (top_down > threshold).astype("uint8") * 255

    final_image = imutils.resize(top_down, height=FIN_IMG_HEIGHT)

    return final_image

