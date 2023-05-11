import numpy as np

def createChirpSignal(samplingrate: int, duration: int, freqfrom: int, freqto: int, linear: bool):
    time = np.linspace(0, duration, samplingrate)
    if linear == True:
        phase = 2 * np.pi * (freqfrom * time + ((freqto - freqfrom) / (2 * duration)) * time ** 2)
    else:
        # phase = 2 * np.pi * ((freqfrom * duration / np.log(freqto / freqfrom)) *
        #                      (np.exp(time * np.log(freqto / freqfrom) / duration) - 1))
        k = (freqto/freqfrom) ** (1/duration)
        phase = 2 * np.pi * freqfrom * (k ** time - 1) / np.log(k)

    return np.sin(phase), time
