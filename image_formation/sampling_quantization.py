import matplotlib.pyplot as plt
import numpy as np

from utils import original_signal

signal_freq = 5.0 # in Hz
duration = 2 # in seconds
sampling_freq = 8 # in Hz
num_bits = 3 # 3-bit quantization (8 levels: 0 - 7)
min_signal = -1 # min signal value
max_signal = 1 # max signal value

# Derived
n_levels = 2 ** num_bits

def main():
    # Original signal over a 2-second time window
    t_points = np.linspace(0, duration, 1000, endpoint=False)  # 1000 pts in [0, duration)
    cont_signal = original_signal(t_points, signal_freq)

    # Sample the original signal.
    n = int(sampling_freq * duration)
    t_sampled = np.linspace(0, duration, n, endpoint=False)
    sampled_signal = original_signal(t_sampled, signal_freq)

    # Quantize the sampled signal
    scaled = (sampled_signal - min_signal) / (max_signal - min_signal) * (n_levels - 1)
    q_s = np.round(scaled)
    q_s = np.clip(q_s, 0, n_levels - 1)
    q_v = min_signal + q_s * (max_signal - min_signal) / (n_levels - 1)

    plt.figure(figsize=(10, 6))
    # Continuous signal
    plt.plot(t_points, cont_signal, label="Continuous signal")
    # Sampled points
    plt.plot(t_sampled, sampled_signal, "o", label=f"Samples ({sampling_freq} Hz)", alpha=0.7)
    # Quantized signal as staircase plot
    plt.step(t_sampled, q_v, where="post",
             label=f"Quantized signal ({num_bits}-bit)", color="r", linestyle="--")

    plt.title("Sampling and Quantization of continuous signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# A reasonable sampling frequency should be at least twice
#  the signal frequency (Nyquist–Shannon theorem).
# Since the signal is 5 Hz, you should use at least 10 Hz
