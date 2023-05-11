import numpy as np

def createChirpSignal(samplingrate: int, duration: int, freqfrom: int, freqto: int, linear: bool):
    # returns the chirp signal as list or 1D-array
    time = np.linspace(0, duration, samplingrate)
    if linear == True:
        c = (freqto - freqfrom) / duration                          # linear chirp rate c
        phase = 2 * np.pi * (freqfrom + c * time / 2) * time
    else:
        k = (freqto / freqfrom) ** (1/duration)                     # exponential chirp rate k
        phase = 2 * np.pi / np.log(k) * (k**time - 1)

    return np.sin(phase)
