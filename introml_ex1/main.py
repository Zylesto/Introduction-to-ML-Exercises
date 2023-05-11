from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal
import chirp
import matplotlib.pyplot as plt

def main():
    samplingrate = 200  # (Hz)
    duration = 1  # (seconds)
    freqfrom = 1  # (Hz)
    freqto = 10  # (Hz)

    exp_chirp = chirp.createChirpSignal(samplingrate, duration, freqfrom, freqto, linear=False)
    linear_chirp = chirp.createChirpSignal(samplingrate, duration, freqfrom, freqto, linear=True)

if __name__ == "__main__":
    main()
