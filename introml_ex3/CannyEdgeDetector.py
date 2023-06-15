import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from PIL import Image

#
# NO MORE MODULES ALLOWED
#

def gaussFilter(img_in, ksize, sigma):
    """
    Filter the image with a Gaussian kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and Gaussian filtered image (both np.ndarray)
    """
    kernel = np.zeros((ksize, ksize))
    center = ksize // 2

    for i in range(ksize):
        for j in range(ksize):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    kernel /= np.sum(kernel)

    # Perform convolution with the Gaussian kernel using scipy's convolve function
    filtered = convolve(img_in, kernel).astype(int)

    # Return the kernel and filtered image
    return kernel, filtered


def sobel(img_in):
    """
    Applies the Sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - Sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    #TODO: *-1 because of the flipped kernel?
    #scipy applies convolution invertedd so multiply it with *-1, it changes gradientchange
    gx = convolve(img_in, sobel_x * -1).astype(int)
    gy = convolve(img_in, sobel_y * -1).astype(int)

    return gx, gy


def gradientAndDirection(gx, gy):
    """
    Calculates the gradient magnitude and direction images
    :param gx: Sobel filtered image in x direction (np.ndarray)
    :param gy: Sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    g = np.sqrt(gx**2 + gy**2).astype(int)
    theta = np.arctan2(gy, gx)

    return g, theta


def convertAngle(angle):
    """
    Computes the nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    angle = np.rad2deg(angle) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        return 0
    elif 22.5 <= angle < 67.5:
        return 45
    elif 67.5 <= angle < 112.5:
        return 90
    else:
        return 135


def maxSuppress(g, theta):
    """
    Calculates maximum suppression
    :param g: gradient magnitude (np.ndarray)
    :param theta: gradient direction image (np.ndarray)
    :return: max_sup - maximum suppression result (np.ndarray)
    """
    max_sup = np.zeros_like(g)
    height, width = g.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            angle = convertAngle(theta[i, j]) #gradient angle
            if angle == 0: #horizontal edge
                if g[i, j] >= g[i, j - 1] and g[i, j] >= g[i, j + 1]: #if gradient is left and right higher, then g[i,j] is highest anstieg
                    max_sup[i, j] = g[i, j]
            elif angle == 45: #edge from left to right (schräg)
                if g[i, j] >= g[i - 1, j + 1] and g[i, j] >= g[i + 1, j - 1]:
                    max_sup[i, j] = g[i, j]
            elif angle == 90: #vertical edge
                if g[i, j] >= g[i - 1, j] and g[i, j] >= g[i + 1, j]:
                    max_sup[i, j] = g[i, j]
            elif angle == 135: #from left to right /schräg)
                if g[i, j] >= g[i - 1, j - 1] and g[i, j] >= g[i + 1, j + 1]:
                    max_sup[i, j] = g[i, j]

    return max_sup


def hysteris(max_sup, t_low, t_high):
    """
    Calculates hysteresis thresholding.
    Attention! This is a simplified version of the lecture's hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2D image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteresis thresholded image (np.ndarray)
    """
    thresholded = np.zeros_like(max_sup)
    strong_i, strong_j = np.where(max_sup >= t_high)
    weak_i, weak_j = np.where((max_sup >= t_low) & (max_sup < t_high)) #indices of strong and weak edges

    thresholded[strong_i, strong_j] = 255 #mark them (here strong)
    thresholded[weak_i, weak_j] = 75

    height, width = thresholded.shape
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if thresholded[i, j] == 75: #weak edge
                if np.max(thresholded[i - 1:i + 2, j - 1:j + 2]) == 255:
                    thresholded[i, j] = 255 #if weak edge is in range of strong edge, it is strong edge
                else:
                    thresholded[i, j] = 0

    return thresholded


def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 10, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result

if __name__ == "__main__":
    canny(np.array(Image.open("contrast.jpg").convert("L")).astype(int))
