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
    m1 = (k3_y - k1_y) / (k3_x - k1_x)
    # inverse of the lines slope -> perpendicular to slope m1
    m2 = -1 / m1

    c1 = k1_y - m1 * k1_x
    c2 = k2_y - m2 * k2_x

    # new coordinate system, rotation center
    x_new = (c2 - c1) / (m1 - m2)
    y_new = m1 * x_new + c1

    # rotation angle for transformation
    angle = np.rad2deg(np.arctan(m2))

    return cv2.getRotationMatrix2D((y_new, x_new), angle, scale=1)


def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''
    smoothed_image = binarizeAndSmooth(img)

    # find and draw the largest contour in image
    contour_image = drawLargestContour(smoothed_image)

    #TODO: everything below doesnt work

    # choose two suitable columns and find 6 intersections with the finger's contour
    x1 = img.shape[1] // 3
    x2 = 2 * img.shape[1] // 3
    intersections1 = getFingerContourIntersections(contour_image, x1)
    intersections2 = getFingerContourIntersections(contour_image, x2)

    # compute middle points from these contour intersections
    midpoint1 = np.mean(intersections1)
    midpoint2 = np.mean(intersections2)

    # K-Punkte extrapolieren
    y1, y2 = img.shape[0], 0
    x1 = int((midpoint1 - y1) * (x2 - x1) / (y2 - y1) + x1)
    x3 = int((midpoint2 - y1) * (x2 - x1) / (y2 - y1) + x1)
    k1 = findKPoints(contour_image, y1, x1, y2, x2)
    k2 = findKPoints(contour_image, y1, x2, y2, x2)
    k3 = findKPoints(contour_image, y1, x3, y2, x2)

    # Koordinatentransformation erhalten
    transform_matrix = getCoordinateTransform(k1, k2, k3)

    # Bild um den neuen Ursprung rotieren
    aligned_img = cv2.warpAffine(img, transform_matrix, img.shape[::-1])

    return aligned_img
