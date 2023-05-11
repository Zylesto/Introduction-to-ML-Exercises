import numpy as np

def createTriangleSignal(samples: int, frequency: int, k_max: int):
        t = np.linspace(0, 1, samples, endpoint=False)
        signal = 0
        #By using a step size of 2 in the range function (range(1, kMax+1, 2)), we ensure that k takes on only odd values during each iteration of the loop.
        for k in range(1, k_max + 1):
            signal += (-1) ** k * np.sin(2 * np.pi * (2*k+1) * frequency * t) / (2 * k + 1) ** 2

        return signal *(4/np.pi ** 2 )



def createSquareSignal(samples: int, frequency: int, k_max: int):
    t = np.linspace(0, 1, samples, endpoint=False)
    signal = 0
    for k in range(1, k_max+1):
        signal += np.sin(2 * np.pi * (2 * k - 1) * frequency * t) / (2 * k -1)
    return signal * (4/np.pi)


def createSawtoothSignal(samples: int, frequency: int, k_max: int, amplitude: int):
    t = np.linspace(0, 1, samples, endpoint=False)
    signal = 0
    for k in range(1, k_max + 1):
        signal += np.sin(2 * np.pi * k * frequency * t) / k
    return (amplitude/2) -((amplitude / np.pi)*signal )
