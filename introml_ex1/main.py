from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal
import chirp
import matplotlib.pyplot as plt

def main():
    samplingrate = 200  # (Hz)
    duration = 1  # (seconds)
    freqfrom = 1  # (Hz)
    freqto = 10  # (Hz)

    exp_chirp, time = chirp.createChirpSignal(samplingrate, duration, freqfrom, freqto, linear=False)
    linear_chirp, time = chirp.createChirpSignal(samplingrate, duration, freqfrom, freqto, linear=True)

    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].plot(time, linear_chirp)
    axs[0].set_title("Linear chirp")

    axs[1].plot(time, exp_chirp)
    axs[1].set_title("Exponential chirp")
    plt.show()


if __name__ == "__main__":
    main()
