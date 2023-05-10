from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal
import chirp

# TODO: Test the functions imported in lines 1 and 2 of this file.

samplingrate = 30000  # (Hz)
duration = 3.0  # (seconds)
freqfrom = 2000  # (Hz)
freqto = 30000  # (Hz)

exp_chirp = chirp.createChirpSignal(samplingrate, duration, freqfrom, freqto, linear=False)
linear_chirp = chirp.createChirpSignal(samplingrate, duration, freqfrom, freqto, linear=True)
