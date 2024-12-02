import numpy as np
from scipy.signal import butter, filtfilt


def freq_detection(x_n: np.ndarray, fs: int, N: int = 1024) -> tuple:
    timestamps = []
    freqs = []


    nyquist = fs / 2
    low = 25 / nyquist
    high = 4200 / nyquist
    b, a = butter(4, [low, high], btype='band')


    x_n = filtfilt(b, a, x_n)

    for window_start in range(0, len(x_n), N):
        window_end = min(window_start + N, len(x_n))
        x_slice = x_n[window_start:window_end]
        X_m = np.fft.rfft(x_slice, n=N)
        X_m[0] = 0
        peak_idx = np.argmax(np.abs(X_m))
        freqs.append(peak_idx / N * fs)
        timestamps.append(window_end / fs)

    return np.array(timestamps), np.array(freqs)