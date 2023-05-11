import numpy as np

def load_sample(filename, duration=4*44100, offset=44100//10):
    sample = np.load(filename)
    highest_value = np.argmax(np.abs(sample))
    start_pos = highest_value + offset
    sliced_sample = sample[start_pos:start_pos + duration]
    return sliced_sample

def compute_frequency(signal, min_freq=20):

    fft = np.fft.fft(signal)
    fft = np.where(fft < min_freq, 0, fft)

    magnitude = np.abs(fft)
    frequency = np.fft.fftfreq(len(signal))
    highest_peak = np.argmax(magnitude)

    return frequency[highest_peak] * len(signal) * (1 / (len(signal) / 4*44100)) # Abtastrate = len(signal) (Anzahl der Sample) / Dauer des Signals; dann Abtastperiode = 1 / Abtastrate
                                                                                # n = Anzahl der Sample

if __name__ == '__main__':
    # Implement the code to answer the questions here
    f = compute_frequency(load_sample("./sounds/Piano.ff.A2.npy"))
    print(f)

# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies