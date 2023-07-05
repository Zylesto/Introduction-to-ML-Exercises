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
    cv2.circle(img, (x, y), 5, 255, 2)
    return img


def binarizeAndSmooth(img) -> np.ndarray:
    '''
    First Binarize using threshold of 115, then smooth with gauss kernel (5, 5)
    :param img: greyscale image in range [0, 255]
    :return: preprocessed image
    '''
    _, thresholded = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY)
    smoothed = cv2.GaussianBlur(thresholded, (5, 5), 0)
    return smoothed


def drawLargestContour(img) -> np.ndarray:
    '''
    find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
    :param img: preprocessed image (mostly b&w)
    :return: contour image
    '''
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    contour_img = np.zeros_like(img)
    cv2.drawContours(contour_img, [largest_contour], 0, 255, 2)
    return contour_img


def getFingerContourIntersections(contour_img, x) -> np.ndarray:
    '''
    Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
    (For help check Palmprint_Algnment_Helper.pdf section 2b)
    :param contour_img:
    :param x: position of the image column to run along
    :return: y-values in np.ndarray in shape (6,)
    '''
    intersections = []
    contour_pixels = np.where(contour_img[:, x] == 255)[0]
    # Entferne Werte, die nahe den Bildrändern liegen
    contour_pixels = contour_pixels[(contour_pixels > 4) & (contour_pixels < contour_img.shape[0] - 5)]
    step = len(contour_pixels) // 6
    for i in range(6):
        intersections.append(contour_pixels[i * step])
    return np.array(intersections)


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
    # Convert the image to grayscale
    grayscale_img = img

    # Binarize the image
    _, binary_img = cv2.threshold(grayscale_img, 127, 255, cv2.THRESH_BINARY)

    binary_img = binary_img.astype(np.uint8)

    # Find the contours in the binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define the line equation: y = mx + c
    if x2 == x1:
        return 0, 0

    m = (y2 - y1) / (x2 - x1)  # Slope
    c = y1 - m * x1  # Intercept

    # Iterate over the contours and find the intersection points
    for contour in contours:
        # Get the contour points
        contour_pts = contour.reshape(-1, 2)

        # Find the y-coordinate of the intersection with the line
        intersection_y = np.round(m * contour_pts[:, 0] + c).astype(int)

        # Find the indices where the intersection occurs
        intersection_indices = np.where(intersection_y == contour_pts[:, 1])

        if len(intersection_indices[0]) > 0:
            # Get the x-coordinate of the intersection
            intersection_x = contour_pts[intersection_indices[0], 0][0]

            return intersection_y[intersection_indices[0]][0], intersection_x

    return -1, -1  # If no intersection found

def getCoordinateTransform(k1, k2, k3) -> np.ndarray:
    '''
    Get a transform matrix to map points from old to new coordinate system defined by k1-3
    Hint: Use cv2 for this.
    :param k1: point in (y, x) order
    :param k2: point in (y, x) order
    :param k3: point in (y, x) order
    :return: 2x3 matrix rotation around origin by angle
    '''


def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''
    # Binarisieren und glätten
    preprocessed_img = binarizeAndSmooth(img)

    # Größten Umriss finden und zeichnen
    contour_img = drawLargestContour(preprocessed_img)

    # Zwei geeignete Spalten auswählen und 6 Schnittpunkte mit dem Fingerrand finden
    x1 = img.shape[1] // 3
    x2 = 2 * img.shape[1] // 3
    intersections1 = getFingerContourIntersections(contour_img, x1)
    intersections2 = getFingerContourIntersections(contour_img, x2)

    # Mittelpunkte der Schnittpunkte berechnen
    midpoint1 = np.mean(intersections1)
    midpoint2 = np.mean(intersections2)

    # K-Punkte extrapolieren
    y1, y2 = img.shape[0], 0
    x1 = int((midpoint1 - y1) * (x2 - x1) / (y2 - y1) + x1)
    x3 = int((midpoint2 - y1) * (x2 - x1) / (y2 - y1) + x1)
    k1 = findKPoints(contour_img, y1, x1, y2, x2)
    k2 = findKPoints(contour_img, y1, x2, y2, x2)
    k3 = findKPoints(contour_img, y1, x3, y2, x2)

    # Koordinatentransformation erhalten
    transform_matrix = getCoordinateTransform(k1, k2, k3)

    # Bild um den neuen Ursprung rotieren
    aligned_img = cv2.warpAffine(img, transform_matrix, img.shape[::-1])

    return aligned_img
