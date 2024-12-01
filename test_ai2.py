import numpy as np
from scipy import signal


def freq_detection(x_n: np.ndarray, fs: int, N: int = 2048) -> tuple:
    """
    Robust frequency detection optimized for musical signals with noise resistance.

    Parameters:
    x_n - signal samples x[n] to be analyzed
    fs - sampling frequency
    N - DFT window size (default: 2048 for better frequency resolution)

    Returns:
    timestamps - ndarray of floats: Time points of frequency estimates
    freqs - ndarray of floats: Detected frequencies
    """
    # Pre-calculate frequency bins for the range we care about (230-4500 Hz)
    freq_bins = np.fft.rfftfreq(N, d=1 / fs)
    freq_mask = (freq_bins >= 230) & (freq_bins <= 4500)

    # Calculate number of windows
    num_windows = len(x_n) // (N // 2)  # Using 50% overlap
    timestamps = np.zeros(num_windows)
    freqs = np.zeros(num_windows)

    # Create Hanning window
    window = signal.windows.hann(N, sym=False)

    # Normalize signal
    x_n = x_n - np.mean(x_n)  # Remove DC offset
    x_n = x_n / (np.std(x_n) + 1e-10)  # Normalize amplitude

    for i in range(num_windows):
        # Extract window with 50% overlap
        start_idx = i * (N // 2)
        end_idx = start_idx + N

        if end_idx > len(x_n):
            break

        # Apply window and compute FFT
        x_slice = x_n[start_idx:end_idx] * window
        X_m = np.fft.rfft(x_slice, n=N)

        # Get magnitude spectrum and apply frequency mask
        magnitude_spectrum = np.abs(X_m)
        magnitude_spectrum[~freq_mask] = 0

        # Apply quadratic interpolation for better frequency estimation
        peak_idx = np.argmax(magnitude_spectrum)
        if 0 < peak_idx < len(magnitude_spectrum) - 1:
            alpha = magnitude_spectrum[peak_idx - 1]
            beta = magnitude_spectrum[peak_idx]
            gamma = magnitude_spectrum[peak_idx + 1]
            peak_offset = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            refined_idx = peak_idx + peak_offset
        else:
            refined_idx = peak_idx

        # Convert to frequency
        freqs[i] = refined_idx * fs / N
        timestamps[i] = (start_idx + N / 2) / fs

        # Apply confidence threshold
        peak_height = magnitude_spectrum[peak_idx]
        mean_magnitude = np.mean(magnitude_spectrum)
        if peak_height < 2 * mean_magnitude:  # SNR threshold
            # If peak is not significant, use previous valid frequency
            if i > 0:
                freqs[i] = freqs[i - 1]

    # Apply median filter to remove outliers
    freqs = signal.medfilt(freqs, 5)

    return timestamps, freqs