'''
Created on 05.10.2016
Modified on 23.12.2020

@author: Daniel
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


# do not import more modules!


def drawCircle(img, x, y):
    '''
    Draw a circle at circle of radius 5px at (x, y) stroke 2 px
    This helps you to visually check your methods.
    :param img: a 2d nd-array
    :param y:
    :param x:
    :return: img with circle at desired position
    '''
    center_coordinates = (x, y)
    radius = 5
    color = (255, 0, 0)
    thickness = 2
    cv2.circle(img, center_coordinates, radius, color, thickness)
    return img


def binarizeAndSmooth(img) -> np.ndarray:
    '''
    First Binarize using threshold of 115, then smooth with gauss kernel (5, 5)
    :param img: greyscale image in range [0, 255]
    :return: preprocessed image
    '''
    _, binarized_image = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY)
    smoothed_image = cv2.GaussianBlur(binarized_image, (5, 5), 0)
    return smoothed_image



def drawLargestContour(img) -> np.ndarray:
    '''
    find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
    :param img: preprocessed image (mostly b&w)
    :return: contour image
    '''
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    contour_imgage = np.zeros_like(img)
    cv2.drawContours(image=contour_imgage, contours=[largest_contour], contourIdx=0, color=(255, 0, 0), thickness=2)
    return contour_imgage


def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    '''
    Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
    (For help check Palmprint_Algnment_Helper.pdf section 2b)
    :param contour_img:
    :param x: position of the image column to run along
    :return: y-values in np.ndarray in shape (6,)
    '''
    intersections = []
    preceding_pixel = 0
    # first pixel is a border pixel
    is_border_pixel = True
    # run along the whole height of the image
    for index, current_pixel in np.ndenumerate(contour_img[:, x]):
        # consider only the white pixels/contour
        if current_pixel == 255 and preceding_pixel == 0:
            # leave out border pixels
            if is_border_pixel == False:
                intersections.append(index[0])
            is_border_pixel = False
        preceding_pixel = current_pixel

        if len(intersections) == 6:
            break

    return np.asarray(intersections)


def findKPoints(img, y1, x1, y2, x2) -> tuple:
    '''
    given two points and the contour image, find the intersection point k
    :param img: binarized contour image (255 == contour)
    :param y1: y-coordinate of point
    :param x1: x-coordinate of point
    :param y2: y-coordinate of point
    :param x2: x-coordinate of point
    :return: intersection point k as a tuple (ky, kx)
    '''
    # slope of a linear function
    m = (y2 - y1) / (x2 - x1)
    # linear function equation y1 = mx1 + c -> reformulated to get the y intercept
    c = y1 - m * x1

    for kx in range(0, img.shape[1]):
        ky = int(m * kx + c)
        # when it's a contour, return ky, kx
        if img[ky, kx] == 255:
            return ky, kx


def getCoordinateTransform(k1, k2, k3) -> np.ndarray:
    '''
    Get a transform matrix to map points from old to new coordinate system defined by k1-k3
    Hint: Use cv2 for this.
    :param k1: point in (y, x) order
    :param k2: point in (y, x) order
    :param k3: point in (y, x) order
    :return: 2x3 matrix rotation around origin by angle
    '''
    k1_y = k1[0]
    k1_x = k1[1]

    k2_y = k2[0]
    k2_x = k2[1]

    k3_y = k3[0]
    k3_x = k3[1]

    # slope of k1 and k3
    epsilon = 1e-7  # small value to prevent division by zero
    m1 = (k3_y - k1_y) / (k3_x - k1_x + epsilon)
    # inverse of the lines slope -> perpendicular to slope m1
    m2 = -1 / m1

    c1 = k1_y - m1 * k1_x
    c2 = k2_y - m2 * k2_x

    # intersection = new center, rotation center
    x_new = (c2 - c1) / (m1 - m2)
    y_new = m1 * x_new + c1

    # rotation angle for transformation
    # m = tan(alpha) -> alpha = arctan(m)
    angle = np.rad2deg(np.arctan(m2))

    return cv2.getRotationMatrix2D((y_new, x_new), angle, scale=1)


def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''

    # threshold and blur
    blurred_img = binarizeAndSmooth(img)
    #  find and draw largest contour in image
    contour_img = drawLargestContour(blurred_img)

    # choose two suitable columns and find 6 intersections with the finger's contour
    x1_column, x2_column = None, None
    x1_intersections, x2_intersections = None, None
    x2_offset = 5

    # iterate over each column of the image
    for column in range(img.shape[1]):
        # get intersections
        intersections = getFingerContourIntersections(contour_img, column)
        if intersections is not None and len(intersections) == 6:   # Ensure 6 intersection points
            if x1_column is None:
                x1_column = column
                x1_intersections = intersections
            elif x2_column is None:
                # to assure that x2_column is not the same as x1_column/lays right to x1_column
                x2_column = max(column - x2_offset, x1_column + 1)
                x2_intersections = getFingerContourIntersections(contour_img, x2_column)
                if len(x2_intersections) == 6:   # Ensure 6 intersection points
                    break

    # Check if intersections were found
    if x1_intersections is None or x2_intersections is None:
        return None

    # Calculate the middle points and find K-points
    k_points = np.empty([3, 2], dtype=int)
    # for each k_points
    for i in range(len(k_points)):
        # calculate the middle points between the intersections
        y1 = (x1_intersections[2 * i] + x1_intersections[2 * i + 1]) // 2
        y2 = (x2_intersections[2 * i] + x2_intersections[2 * i + 1]) // 2
        k_points[i] = findKPoints(contour_img, y1, x1_column, y2, x2_column)

    # compute rotation matrix
    rotation_matrix = getCoordinateTransform(*k_points)
    # rotate the image around new origin
    return cv2.warpAffine(img, rotation_matrix, img.shape[::-1])
