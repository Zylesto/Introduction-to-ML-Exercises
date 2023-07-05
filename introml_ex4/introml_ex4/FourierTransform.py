'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''

import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!



def polarToKart(shape, r, theta):
    '''
    convert polar coordinates with origin in image center to kartesian
    :param shape: shape of the image
    :param r: radius from image center
    :param theta: angle
    :return: y, x
    '''
    center_y = shape[0] // 2
    center_x = shape[1] // 2
    x = r * np.cos(theta) + center_x
    y = r * np.sin(theta) + center_y
    return y, x


def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    use the fft to generate a magnitude spectrum and shift it to the image center.
    Hint: This can be done with numpy :)
    :param img:
    :return:
    '''
    f = np.fft.fftshift(np.fft.fft2(img))

    # Berechne das Magnitudenspektrum
    magnitude_spectrum = np.abs(f)

    return magnitude_spectrum


def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum:
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring
    :return: feature vector of k features
    '''
    rows, cols = magnitude_spectrum.shape
    feature_vector = np.zeros(k)

    for i in range(k):
        # Radius des aktuellen Rings
        radius = (i + 1) * max(rows, cols) / (2 * k)

        # Winkelabstand zwischen den Abtastpunkten auf dem Ring
        angle_step = 2 * np.pi / sampling_steps

        # Abtastpunkte auf dem Ring
        for j in range(sampling_steps):
            angle = j * angle_step
            y, x = polarToKart(magnitude_spectrum.shape, radius, angle)
            y = int(y)
            x = int(x)
            feature_vector[i] += magnitude_spectrum[y, x]

    return feature_vector


def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    """
    Follow the approach to extract Fan features
    Assume all rays have same length regardless of angle.
    Their length should be set by the smallest feasible ray.
    :param magnitude_spectrum:
    :param k: number of fans-like features to extract
    :param sampling_steps: number of rays to sample from in one fan-like area
    :return: feature vector of length k
    """
    rows, cols = magnitude_spectrum.shape
    feature_vector = np.zeros(k)

    for i in range(k):
        # Winkelabstand zwischen den Abtaststrahlen
        angle_step = 2 * np.pi / sampling_steps

        # Abtaststrahlen im aktuellen Ventilator-ähnlichen Bereich
        for j in range(sampling_steps):
            angle = i * angle_step
            y, x = polarToKart(magnitude_spectrum.shape, min(rows, cols) / 2, angle)
            y = int(y)
            x = int(x)
            feature_vector[i] += magnitude_spectrum[y, x]

    return feature_vector


def calcuateFourierParameters(img, k, sampling_steps) -> (np.ndarray, np.ndarray):
    '''
    Extract Features in Fourier space following the paper.
    :param img: input image
    :param k: number of features to extract from each method
    :param sampling_steps: number of samples to accumulate for each feature
    :return: R, T feature vectors of length k
    '''
    magnitude_spectrum = calculateMagnitudeSpectrum(img)

    R = extractRingFeatures(magnitude_spectrum, k, sampling_steps)
    T = extractFanFeatures(magnitude_spectrum, k, sampling_steps)

    return R, T
