import numpy as np
from scipy.signal import butter, filtfilt

def freq_detection(x_n: np.ndarray, fs: int, mode: str = 'general', N: int = 1024) -> tuple:
    def bandpass_filter(signal, fs, lowcut=200, highcut=800):
        nyquist = fs / 2
        b, a = butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')
        return filtfilt(b, a, signal)

    def interpolate_peak(X_m, peak_idx):
        if peak_idx == 0 or peak_idx == len(X_m) - 1:
            return peak_idx
        alpha = np.log(np.abs(X_m[peak_idx - 1]))
        beta = np.log(np.abs(X_m[peak_idx]))
        gamma = np.log(np.abs(X_m[peak_idx + 1]))
        p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
        return peak_idx + p

    timestamps = []
    freqs = []

    if mode == 'vocal':
        x_n = bandpass_filter(x_n, fs)

    for window_start in range(0, len(x_n), N):
        window_end = min(window_start + N, len(x_n))
        x_slice = x_n[window_start:window_end]

        window = np.hanning(len(x_slice)) if mode == 'vocal' else np.hamming(len(x_slice))
        x_slice = x_slice * window

        X_m = np.fft.rfft(x_slice, n=N)
        X_m[0] = 0

        peak_idx = np.argmax(np.abs(X_m))
        m_peak = interpolate_peak(X_m, peak_idx)

        freqs.append(m_peak / N * fs)
        timestamps.append(window_end / fs)

    return np.array(timestamps), np.array(freqs)