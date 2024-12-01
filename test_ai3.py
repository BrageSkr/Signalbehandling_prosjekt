import numpy as np
from scipy import signal


def freq_detection(x_n: np.ndarray, fs: int, N: int = 2048) -> tuple:
    """
    Frequency detection using parabolic interpolation for better accuracy.
    Returns results in same format as original freq_detection function.

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

    # Calculate number of windows with 75% overlap
    hop_size = N // 4  # 75% overlap
    num_windows = (len(x_n) - N) // hop_size + 1

    timestamps = np.zeros(num_windows)
    freqs = np.zeros(num_windows)

    # Create Hanning window
    window = signal.windows.hann(N, sym=False)

    # Normalize input signal
    x_n = x_n - np.mean(x_n)  # Remove DC offset
    x_n = x_n / (np.std(x_n) + 1e-10)  # Normalize amplitude

    def parabolic_interpolation(spectrum, peak_index):
        """
        Perform parabolic interpolation around the peak to get a more accurate frequency.
        Returns interpolated peak position.
        """
        if peak_index <= 0 or peak_index >= len(spectrum) - 1:
            return peak_index

        alpha = abs(spectrum[peak_index - 1])
        beta = abs(spectrum[peak_index])
        gamma = abs(spectrum[peak_index + 1])

        p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)

        # Only use interpolation if the result is reasonable
        if -1 <= p <= 1:
            return peak_index + p
        return peak_index

    for i in range(num_windows):
        # Extract window with overlap
        start_idx = i * hop_size
        end_idx = start_idx + N

        if end_idx > len(x_n):
            break

        # Apply window and compute FFT
        x_slice = x_n[start_idx:end_idx] * window
        X_m = np.fft.rfft(x_slice, n=N)

        # Get magnitude spectrum and apply frequency mask
        magnitude_spectrum = np.abs(X_m)
        masked_spectrum = magnitude_spectrum.copy()
        masked_spectrum[~freq_mask] = 0

        # Find peak in masked spectrum
        peak_idx = np.argmax(masked_spectrum)

        # Apply parabolic interpolation
        interpolated_peak_idx = parabolic_interpolation(masked_spectrum, peak_idx)

        # Convert to frequency
        freq = interpolated_peak_idx * fs / N

        # Add confidence check
        peak_height = magnitude_spectrum[peak_idx]
        mean_magnitude = np.mean(magnitude_spectrum)
        if peak_height < 2 * mean_magnitude:  # SNR threshold
            # If peak is not significant, use previous valid frequency
            if i > 0:
                freq = freqs[i - 1]

        freqs[i] = freq
        timestamps[i] = (start_idx + N / 2) / fs

    # Apply median filter to remove outliers
    kernel_size = 5
    freqs = signal.medfilt(freqs, kernel_size)

    # Remove any remaining zeros at the end
    valid_mask = timestamps > 0
    timestamps = timestamps[valid_mask]
    freqs = freqs[valid_mask]

    return timestamps, freqs