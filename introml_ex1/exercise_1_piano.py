import numpy as np



def load_sample(filename, duration=4*44100, offset=44100//10):
    sample = np.load(filename)
    highest_value = np.argmax(np.abs(sample))
    start_pos = highest_value + offset
    sliced_sample = sample[start_pos:start_pos + duration]
    return sliced_sample


def compute_frequency(signal, min_freq=50):
    # so calculer   la magnitude de la transformée de Fourier du signal
    spectrum = np.abs(np.fft.fft(signal))

   # Calculer des fréquences associées au spectre
    freqs = np.fft.fftfreq(len(signal), d=1/44100)


    # Trouver le pic bon
    mask = (freqs > min_freq) & (spectrum == np.max(spectrum[freqs > min_freq]))

    max_freq = freqs[mask]

    if len(max_freq) > 0:
        return max_freq[0]
    else:
        return 0
if __name__ == '__main__':
    # Implement the code to answer the questions here
    f = compute_frequency(load_sample("./sounds/Piano.ff.A2.npy"))
    print(f)

# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies