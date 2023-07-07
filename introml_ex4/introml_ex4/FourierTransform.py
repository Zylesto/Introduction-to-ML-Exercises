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
    ft = np.fft.fft2(img)
    shifted_ft = np.fft.fftshift(ft)
    magnitude = np.abs(shifted_ft)
    # + 1 to avoid log of zero and convert it to decibel
    return 20 * np.log10(magnitude + 1)


def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    Follow the approach to extract ring features
    :param magnitude_spectrum:
    :param k: number of rings to extract = #features
    :param sampling_steps: times to sample one ring
    :return: feature vector of k features
    '''

    features = np.zeros(k)
    theta_steps = np.linspace(0, np.pi, sampling_steps)
    for i in range(len(features)):
        r_steps = np.arange(k * i, k * (i + 1) + 1)
        for r in r_steps:
            y, x = polarToKart(magnitude_spectrum.shape, r, theta_steps)
            y = y.astype(int)
            x = x.astype(int)
            features[i] += np.sum(magnitude_spectrum[y, x])
    return features



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
    features = np.zeros(k)
    r = np.arange(0, (np.min(magnitude_spectrum.shape) / 2) - 1)
    for i in range(len(features)):
        theta_steps = np.linspace(i * np.pi / k, (i+1) * np.pi / k, sampling_steps, endpoint=False)
        for theta in theta_steps:
            y, x = polarToKart(magnitude_spectrum.shape, r, theta)
            y = y.astype(int)
            x = x.astype(int)
            features[i] += np.sum(magnitude_spectrum[y, x])
    return features

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
