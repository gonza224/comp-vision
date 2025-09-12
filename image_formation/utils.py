import numpy as np


def thin_lens_zi(f, z0):
    return 1 / (1/f - 1/z0)

def aperture_diameter(f, N):
    return f / N

    
def original_signal(t, signal_freq):
    return np.sin(2 * np.pi * signal_freq * t)

def add_Gaussian_noise(signal, mean, std):
    mag = np.max(signal) - np.min(signal)
    noise = np.random.normal(mean, std * mag, len(signal))
    return signal + noise

def mse(original, noisy):
    return np.mean((original - noisy) ** 2)

def rmse(original, noisy):
    return np.sqrt(mse(original, noisy))

def psnr(original, noisy):
    m = mse(original, noisy)
    peak = np.max(np.abs(original))
    return 10 * np.log10((peak ** 2) / m)