import numpy as np

def load_sample(filename, duration=4*44100, offset=44100//10):
    sample = np.load(filename)
    highest_value = np.argmax(np.abs(sample))
    start_pos = highest_value + offset
    sliced_sample = sample[start_pos:start_pos + duration]
    return sliced_sample

def compute_frequency(signal, min_freq=20):
    fft = np.fft.fft(signal)
    magnitude = np.abs(fft)
    freq = np.fft.fftfreq(len(signal), d=1/44100)    # d = inverse sampling frequency
    selection = freq > min_freq                      # mask to get values only where frequency > min_freq
    peak = np.argmax(magnitude[selection])
    return freq[selection][peak]

if __name__ == '__main__':
    freq_a2 = compute_frequency(load_sample("./sounds/Piano.ff.A2.npy"))
    freq_a3 = compute_frequency(load_sample("./sounds/Piano.ff.A3.npy"))
    freq_a4 = compute_frequency(load_sample("./sounds/Piano.ff.A4.npy"))
    freq_a5 = compute_frequency(load_sample("./sounds/Piano.ff.A5.npy"))
    freq_a6 = compute_frequency(load_sample("./sounds/Piano.ff.A6.npy"))
    freq_a7 = compute_frequency(load_sample("./sounds/Piano.ff.A7.npy"))
    freq_xx = compute_frequency(load_sample("./sounds/Piano.ff.XX.npy"))
    print(freq_xx)  # D6

# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies