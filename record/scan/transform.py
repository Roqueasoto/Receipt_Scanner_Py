# Python Script to perform a 4 point perspective transform on a photo
# document (rectangle shape), converting it to birds-eye view.

import numpy as np
import cv2


def _order_points(pts):
    # Initializing a list of coordinates to store the corners of the
    # rectangle in the picture, with the 0th entry corresponding to the
    # top-left corner, the 1st entry to the top-right corner, the 2nd
    # entry to the bottom-left corner and the 3rd entry to the bottom-
    # right corner.
    corners = np.zeros((4, 2), dtype="float32")

    # The top-left corner and bottom-right corner can be identified w/
    # the smallest and largest sum of x & y values, respectively.
    xy_sum = pts.sum(axis=1)
    corners[0] = pts[np.argmin(xy_sum)]
    corners[3] = pts[np.argmax(xy_sum)]

    # The top-right corner and bottom-left corner can be identified w/
    # the smallest and largest difference between x and y, respectively.
    xy_diff = np.diff(pts, axis=1)
    corners[1] = pts[np.argmin(xy_diff)]
    corners[2] = pts[np.argmax(xy_diff)]

    # Return the ordered corner points.
    return corners


def four_point_transform(image, pts):
    # Order the points consistently and unpack them.
    rect = _order_points(pts)
    (tr, tl, bl, br) = rect

    # Compute the width of the new image, as the maximum of the distance
    # between the top corners or the bottom corners.
    width_top = np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))
    width_bot = np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))
    max_width = max(int(width_bot), int(width_top))

    # Compute the height of the new image, as the max of the left corner
    # distances and the right corner distances.
    height_l = np.sqrt(((tl[0]-bl[0])**2) + ((tl[1]-bl[1])**2))
    height_r = np.sqrt(((tr[0]-br[0])**2) + ((tr[1]-br[1])**2))
    max_height = max(int(height_l), int(height_r))

    # Using the new image dimensions we define new point locations w/
    # the top-left, top-right, bottom-left, bottom-right order.
    new_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [0, max_height - 1],
        [max_width - 1, max_height - 1]],
        dtype="float32"
    )

    # Compute the perspective transform matrix and apply it
    p_mat = cv2.getPerspectiveTransform(rect, new_pts)
    warped = cv2.warpPerspective(image, p_mat, (max_width, max_height))

    # Return the warped image
    return warped

