import matplotlib.pyplot as plt
import numpy as np

from utils import add_Gaussian_noise, mse, original_signal, psnr, rmse

signal_freq = 5.0 # in Hz
duration = 2 # in seconds
sampling_freq = 8 # in Hz
num_bits = 3 # 3-bit quantization (8 levels: 0 - 7)
min_signal = -1 # min signal value
max_signal = 1 # max signal value

# Noise parameters
mean = 0.0
std_dev = 0.1

def quantize(signal, bits, vmin, vmax):
    n_levels = 2 ** bits
    scaled = (signal - vmin) / (vmax - vmin) * (n_levels - 1)
    q_idx = np.round(scaled)
    q_idx = np.clip(q_idx, 0, n_levels - 1)
    q_val = vmin + q_idx * (vmax - vmin) / (n_levels - 1)
    return q_val

def main():
    # Generate sampled signal
    n = int(sampling_freq * duration)
    t_sampled = np.linspace(0, duration, n, endpoint=False)
    sampled = original_signal(t_sampled, signal_freq)

    # Add noise and quantize
    noisy = add_Gaussian_noise(sampled, mean, std_dev)
    quantized = quantize(noisy, num_bits, min_signal, max_signal)

    # Compute errors
    mse_val = mse(sampled, quantized)
    rmse_val = rmse(sampled, quantized)
    psnr_val = psnr(sampled, quantized)

    plt.figure(figsize=(10, 6))
    plt.plot(t_sampled, sampled, 'o-', label="Original samples")
    plt.plot(t_sampled, noisy, 'x-', label="Noisy samples", alpha=0.7)
    plt.step(t_sampled, quantized, where='post', linestyle='--', color='r',
             label=f"Quantized ({num_bits}-bit)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Noisy Sampled and Quantized Signal")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"MSE  = {mse_val:.2f}")
    print(f"RMSE = {rmse_val:.2f}")
    print(f"PSNR = {psnr_val:.2f} dB")

if __name__ == "__main__":
    main()
