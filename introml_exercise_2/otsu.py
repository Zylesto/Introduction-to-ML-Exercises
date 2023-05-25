import numpy as np
#
# NO OTHER IMPORTS ALLOWED
#


def create_greyscale_histogram(img):
    '''
    returns a histogram of the given image
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    '''
    hist = np.zeros(256, dtype=int)
    rows, cols = img.shape

    for i in range(rows):
        for j in range(cols):
            pixel_value = img[i, j]
            hist[pixel_value] += 1

    return hist


def binarize_threshold(img, t):
    '''
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    '''
    binarized_img = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > t:
                binarized_img[i, j] = 255

    return binarized_img


def p_helper(hist, theta: int):
    '''
    Compute p0 and p1 using the histogram and the current theta,
    do not take care of border cases in here
    :param hist:
    :param theta: current theta
    :return: p0, p1
    '''
    p0 = 0
    p1 = 0

    for i in range(theta + 1):
        p0 += hist[i]

    for i in range(theta + 1, len(hist)):
        p1 += hist[i]

    return p0, p1


def mu_helper(hist, theta, p0, p1):
    '''
    Compute mu0 and m1
    :param hist: histogram
    :param theta: current theta
    :param p0:
    :param p1:
    :return: mu0, mu1
    '''
    mu0 = 0
    mu1 = 0

    for i in range(theta + 1):
        mu0 += i * hist[i] / p0

    for i in range(theta + 1, len(hist)):
        mu1 += i * hist[i] / p1

    return mu0, mu1


def calculate_otsu_threshold(hist):
    '''
    calculates theta according to otsus method

    :param hist: 1D array
    :return: threshold (int)
    '''

    total_pixels = np.sum(hist) #needed to prevent overflows
    max_variance = 0
    best_threshold = 0

    for t in range(256):
        p0, p1 = p_helper(hist, t)
        if p0 == 0 or p1 == 0:
            continue

        mu0, mu1 = mu_helper(hist, t, p0, p1)
        p0_norm = p0 / total_pixels
        p1_norm = p1 / total_pixels
        variance = p0_norm * p1_norm * (mu0 - mu1) ** 2

        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    return best_threshold



def otsu(img):
    '''
    calculates a binarized image using the otsu method.
    Hint: reuse the other methods
    :param image: grayscale image values in range [0, 255]
    :return: np.ndarray binarized image with values {0, 255}
    '''
    hist = create_greyscale_histogram(img)
    threshold = calculate_otsu_threshold(hist)
    binarized_img = binarize_threshold(img, threshold)
    return binarized_img
