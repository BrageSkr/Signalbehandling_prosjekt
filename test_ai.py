import numpy as np


def freq_detection(x_n: np.ndarray, fs: int, N: int = 1024) -> tuple:
    """
    Optimized version of frequency detection focusing on 230-4500 Hz range.
    Uses frequency masking and vectorized operations for better performance.

    Parameters:
    x_n - signal samples x[n] to be analyzed
    fs - sampling frequency
    N - DFT window size in number of samples
        Defaults to 1024 samples

    Returns:
    timestamps - ndarray of floats
        Points in time at which frequency contents were estimated.
    freqs - ndarray of floats
        Most prominent frequency detected in 230-4500 Hz range.
    """
    # Pre-calculate frequency bins
    freq_bins = np.fft.rfftfreq(N, d=1 / fs)

    # Create frequency mask for 230-4500 Hz range
    freq_mask = (freq_bins >= 230) & (freq_bins <= 4500)

    # Calculate number of complete windows
    num_windows = len(x_n) // N

    # Pre-allocate arrays for better performance
    timestamps = np.zeros(num_windows + (1 if len(x_n) % N else 0))
    freqs = np.zeros_like(timestamps)

    # Process complete windows
    for i in range(num_windows):
        window_start = i * N
        x_slice = x_n[window_start:window_start + N]

        # Apply Hanning window for better frequency resolution
        x_slice = x_slice * np.hanning(len(x_slice))

        # Calculate FFT and apply frequency mask
        X_m = np.fft.rfft(x_slice, n=N)
        X_m_masked = np.abs(X_m.copy())
        X_m_masked[~freq_mask] = 0

        # Find peak frequency
        peak_idx = np.argmax(X_m_masked)
        freqs[i] = freq_bins[peak_idx]
        timestamps[i] = (window_start + N) / fs

    # Handle remaining samples if any
    if len(x_n) % N:
        remaining_samples = x_n[num_windows * N:]
        x_slice = remaining_samples * np.hanning(len(remaining_samples))
        X_m = np.fft.rfft(x_slice, n=N)
        X_m_masked = np.abs(X_m.copy())
        X_m_masked[~freq_mask] = 0
        peak_idx = np.argmax(X_m_masked)
        freqs[-1] = freq_bins[peak_idx]
        timestamps[-1] = len(x_n) / fs

    return timestamps, freqs