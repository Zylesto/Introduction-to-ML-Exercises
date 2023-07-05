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
    img_freq_dom = np.fft.fftshift(np.fft.fft2(img))
    return 20 * np.log10(np.abs(img_freq_dom) + 1)


def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum:
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring
    :return: feature vector of k features
    '''

    feature_vector = np.zeros(k)
    spectrum_shape = magnitude_spectrum.shape
    radius_inc = k
    theta = np.linspace(0, np.pi, sampling_steps)

    for i in range(len(feature_vector)):
        r_steps = np.arange(radius_inc * i, radius_inc * (i + 1) + 1)
        for r in r_steps:
            spectrum_indices = np.asarray(polarToKart(spectrum_shape, r, theta)).astype(int)
            feature_vector[i] += np.sum(magnitude_spectrum[spectrum_indices[0], spectrum_indices[1]])

    return feature_vector


def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    """
    Follow the approach to extract Fan features
    Assume all rays have the same length regardless of angle.
    Their length should be set by the smallest feasible ray.
    :param magnitude_spectrum: Magnitude spectrum of a signal or image
    :param k: Number of fan-like features to extract
    :param sampling_steps: Number of rays to sample from in one fan-like area
    :return: Feature vector of length k
    """

    feature_vector = np.zeros(k)
    spectrum_shape = magnitude_spectrum.shape
    r = np.arange(0, (np.min(spectrum_shape) / 2) - 1)

    for i in range(len(feature_vector)):
        for theta in np.linspace(i * np.pi / k, (i + 1) * np.pi / k, sampling_steps, endpoint=False):
            spectrum_indices = polarToKart(spectrum_shape, r, theta)
            spectrum_indices = np.asarray(spectrum_indices, dtype=int)
            feature_vector[i] += np.sum(magnitude_spectrum[spectrum_indices[0], spectrum_indices[1]])

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
