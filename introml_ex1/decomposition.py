import numpy as np
import matplotlib.pyplot as plt


def createTriangleSignal(samples: int, frequency: int, k_max: int):
    step_size = 1 / samples
    t = np.arange(0, 1, step_size)

    signal = np.zeros(samples)
    for k in range(0, k_max+1):
        signal += (-1)**k * np.sin(2 * np.pi * (2 * k + 1) * frequency * t) / (2 * k + 1)**2
    signal = (8 / (np.pi**2)) * signal

    plt.plot(t, signal)
    plt.show()

    return signal


def createSquareSignal(samples: int, frequency: int, k_max: int):
    step_size = 1 / samples
    t = np.arange(0, 1, step_size)

    signal = np.zeros(samples)

    for k in range(1, k_max+1):
        signal += np.sin(2 * np.pi * (2 * k - 1) * frequency * t) / (2 * k - 1)
    signal = (4 / np.pi) * signal

    plt.plot(t, signal)
    plt.show()

    return signal


def createSawtoothSignal(samples: int, frequency: int, k_max: int, amplitude: int):
    step_size = 1 / samples
    t = np.arange(0, 1, step_size)

    signal = np.zeros(samples)
    for k in range(1, k_max+1):
        signal += np.sin(2 * np.pi * k * frequency * t) / k
    signal = (amplitude / 2) - (amplitude / np.pi) * signal
    return signal