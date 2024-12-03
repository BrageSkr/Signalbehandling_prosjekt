import numpy as np
from scipy.signal import butter, filtfilt


def freq_detection(x_n: np.ndarray, fs: int, N: int = 1024, num_harmonics: int = 3) -> tuple:
    timestamps = []
    freqs = []

    # Band-pass filter (keep the same as before)
    nyquist = fs / 2
    low = 25 / nyquist
    high = 4200 / nyquist
    b, a = butter(4, [low, high], btype='band')
    x_n = filtfilt(b, a, x_n)

    for window_start in range(0, len(x_n), N):
        window_end = min(window_start + N, len(x_n))
        x_slice = x_n[window_start:window_end]

        # Calculate FFT
        X_m = np.fft.rfft(x_slice, n=N)
        magnitude_spectrum = np.abs(X_m)

        # Calculate HPS
        hps_spectrum = np.copy(magnitude_spectrum)
        for harmonic in range(2, num_harmonics + 1):
            decimated = magnitude_spectrum[::harmonic]  # Downsample spectrum
            pad_length = len(hps_spectrum) - len(decimated)
            hps_spectrum *= np.pad(decimated, (0, pad_length))

        # Find peak in HPS
        peak_idx = np.argmax(hps_spectrum[1:]) + 1  # Skip DC
        freq = peak_idx * fs / N

        freqs.append(freq)
        timestamps.append(window_end / fs)

    return np.array(timestamps), np.array(freqs)