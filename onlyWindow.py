import numpy as np
from scipy import signal

def freq_detection(x_n: np.ndarray, fs: int, N: int = 1024) -> tuple:
    timestamps = []
    freqs = []
    for window_start in range(0, len(x_n), N):
        window_end = min(window_start + N, len(x_n))
        x_slice = x_n[window_start:window_end]

        window =signal.windows.hamming(len(x_slice))
        x_slice = x_slice * window

        X_m = np.fft.rfft(x_slice, n=N)
        X_m[0] = 0
        peak_idx = np.argmax(np.abs(X_m))
        freqs.append(peak_idx / N * fs)
        timestamps.append(window_end / fs)
    return np.array(timestamps), np.array(freqs)