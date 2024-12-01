import numpy as np
from scipy import signal


def freq_detection(x_n: np.ndarray, fs: int, N: int = 4096) -> tuple:
    """
    Frequency detection optimized for musical instruments with strong harmonics.

    Parameters:
    x_n - signal samples x[n] to be analyzed
    fs - sampling frequency
    N - DFT window size (increased for better low-frequency resolution)

    Returns:
    timestamps - ndarray of floats: Time points of frequency estimates
    freqs - ndarray of floats: Detected frequencies
    """
    # Use longer windows for better low-frequency resolution
    hop_size = N // 4
    num_windows = (len(x_n) - N) // hop_size + 1

    timestamps = np.zeros(num_windows)
    freqs = np.zeros(num_windows)

    # Create Hanning window
    window = signal.windows.hann(N, sym=False)

    # Normalize input signal
    x_n = x_n - np.mean(x_n)
    x_n = x_n / (np.max(np.abs(x_n)) + 1e-10)

    def find_fundamental(spectrum, freq_bins):
        """
        Find fundamental frequency using harmonic product spectrum (HPS).
        """
        num_harmonics = 3
        product_spectrum = np.ones_like(spectrum)

        for harmonic in range(1, num_harmonics + 1):
            # Downsample spectrum for each harmonic
            downsampled = spectrum[::harmonic]
            # Pad to original length
            padded = np.pad(downsampled, (0, len(spectrum) - len(downsampled)))
            product_spectrum *= padded

        # Find peak in product spectrum
        peak_idx = np.argmax(product_spectrum)
        return freq_bins[peak_idx]

    def peak_picking(spectrum, freq_bins):
        """
        Advanced peak picking with harmonic consideration.
        """
        # Find all peaks
        peaks = signal.find_peaks(spectrum, height=np.max(spectrum) / 10,
                                  distance=int(200 * N / fs))[0]  # Minimum 200 Hz between peaks

        if len(peaks) == 0:
            return freq_bins[np.argmax(spectrum)]

        # Get peak heights
        peak_heights = spectrum[peaks]

        # Sort peaks by height
        sorted_indices = np.argsort(peak_heights)[::-1]
        sorted_peaks = peaks[sorted_indices]

        # Check for harmonic relationships
        fundamental_candidates = []
        for peak_idx in sorted_peaks:
            freq = freq_bins[peak_idx]
            if 200 <= freq <= 1000:  # Reasonable range for fundamental
                harmonic_ratios = freq_bins[sorted_peaks] / freq
                # Count near-integer ratios
                near_integer_count = np.sum(np.abs(harmonic_ratios - np.round(harmonic_ratios)) < 0.15)
                fundamental_candidates.append((freq, near_integer_count))

        if fundamental_candidates:
            # Choose candidate with most harmonics
            return max(fundamental_candidates, key=lambda x: x[1])[0]

        return freq_bins[sorted_peaks[0]]

    # Process each window
    for i in range(num_windows):
        start_idx = i * hop_size
        end_idx = start_idx + N

        if end_idx > len(x_n):
            break

        # Extract and window the signal
        x_slice = x_n[start_idx:end_idx] * window

        # Compute FFT
        X_m = np.fft.rfft(x_slice, n=N)
        magnitude_spectrum = np.abs(X_m)

        # Get frequency bins
        freq_bins = np.fft.rfftfreq(N, d=1 / fs)

        # Apply harmonic product spectrum method
        freq_hps = find_fundamental(magnitude_spectrum, freq_bins)

        # Apply peak picking method
        freq_peaks = peak_picking(magnitude_spectrum, freq_bins)

        # Choose the more likely frequency based on consistency
        if i > 0 and abs(freqs[i - 1] - freq_peaks) < abs(freqs[i - 1] - freq_hps):
            freq = freq_peaks
        else:
            freq = freq_hps

        # Store results
        freqs[i] = freq
        timestamps[i] = (start_idx + N / 2) / fs

    # Apply median filter to remove outliers
    kernel_size = min(5, len(freqs))
    if kernel_size % 2 == 0:
        kernel_size += 1
    freqs = signal.medfilt(freqs, kernel_size)

    return timestamps, freqs