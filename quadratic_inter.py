import numpy as np


def freq_detection(x_n: np.ndarray, fs: int, N: int = 1024) -> tuple:
    """
    Frequency detection using quadratic interpolation.
    """
    timestamps = []
    freqs = []
    for window_start in range(0, len(x_n), N):
        window_end = min(window_start + N, len(x_n))
        x_slice = x_n[window_start:window_end]
        X_m = np.fft.rfft(x_slice, n=N)
        X_m[0] = 0

        peak_idx = np.argmax(np.abs(X_m))
        if peak_idx > 0 and peak_idx < len(X_m) - 1:
            alpha = np.abs(X_m[peak_idx - 1])
            beta = np.abs(X_m[peak_idx])
            gamma = np.abs(X_m[peak_idx + 1])
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            m_peak = peak_idx + p
        else:
            m_peak = peak_idx

        freqs.append(m_peak / N * fs)
        timestamps.append(window_end / fs)
    return np.array(timestamps), np.array(freqs)