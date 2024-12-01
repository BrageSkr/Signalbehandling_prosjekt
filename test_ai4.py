import numpy as np
from scipy import signal


def freq_detection(x_n: np.ndarray, fs: int, N: int = 2048) -> tuple:
    """
    Frequency detection with adaptive noise filtering and parabolic interpolation.

    Parameters:
    x_n - signal samples x[n] to be analyzed
    fs - sampling frequency
    N - DFT window size (default: 2048 for better frequency resolution)

    Returns:
    timestamps - ndarray of floats: Time points of frequency estimates
    freqs - ndarray of floats: Detected frequencies
    """
    # Pre-calculate frequency bins
    freq_bins = np.fft.rfftfreq(N, d=1 / fs)
    freq_mask = (freq_bins >= 230) & (freq_bins <= 4500)

    # Calculate windows with 75% overlap
    hop_size = N // 4
    num_windows = (len(x_n) - N) // hop_size + 1

    timestamps = np.zeros(num_windows)
    freqs = np.zeros(num_windows)
    noise_levels = np.zeros(num_windows)  # Track noise levels

    # Create Hanning window
    window = signal.windows.hann(N, sym=False)

    # Initial signal preprocessing
    x_n = x_n - np.mean(x_n)  # Remove DC
    x_n = x_n / (np.std(x_n) + 1e-10)  # Normalize

    # Design bandpass filter for initial noise reduction
    nyquist = fs / 2
    low_cut = 200  # Hz, slightly below our range of interest
    high_cut = 4700  # Hz, slightly above our range of interest
    b, a = signal.butter(4, [low_cut / nyquist, high_cut / nyquist], btype='band')

    # Apply initial bandpass filtering
    x_n = signal.filtfilt(b, a, x_n)

    def estimate_noise_level(spectrum):
        """Estimate noise level from spectrum."""
        # Sort spectrum magnitudes
        sorted_magnitudes = np.sort(np.abs(spectrum))
        # Use lower 20% as noise estimate
        noise_floor = np.mean(sorted_magnitudes[:len(sorted_magnitudes) // 5])
        return noise_floor

    def adaptive_filter(x_slice, noise_level):
        """Apply adaptive filtering based on noise level."""
        if noise_level > 0.1:  # High noise condition
            # Apply more aggressive filtering for noisy signals
            # Spectral subtraction
            X = np.fft.rfft(x_slice)
            mag = np.abs(X)
            phase = np.angle(X)
            # Subtract estimated noise floor with spectral floor
            mag = np.maximum(mag - noise_level * 2, mag * 0.1)
            # Reconstruct signal
            X_clean = mag * np.exp(1j * phase)
            return np.fft.irfft(X_clean)
        return x_slice

    def parabolic_interpolation(spectrum, peak_index):
        """Perform parabolic interpolation around the peak."""
        if peak_index <= 0 or peak_index >= len(spectrum) - 1:
            return peak_index

        alpha = abs(spectrum[peak_index - 1])
        beta = abs(spectrum[peak_index])
        gamma = abs(spectrum[peak_index + 1])

        p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)

        if -1 <= p <= 1:
            return peak_index + p
        return peak_index

    # Initialize Kalman filter parameters
    kalman_var = 100.0  # Initial variance
    kalman_q = 10.0  # Process noise
    kalman_r = 50.0  # Measurement noise
    prev_freq = None
    prev_var = kalman_var

    def kalman_update(measurement, prev_estimate, prev_var):
        """Apply Kalman filtering to frequency estimates."""
        # Prediction
        pred_var = prev_var + kalman_q

        # Update
        kalman_gain = pred_var / (pred_var + kalman_r)
        estimate = prev_estimate + kalman_gain * (measurement - prev_estimate)
        new_var = (1 - kalman_gain) * pred_var

        return estimate, new_var

    # Process windows
    for i in range(num_windows):
        start_idx = i * hop_size
        end_idx = start_idx + N

        if end_idx > len(x_n):
            break

        # Extract and window the signal
        x_slice = x_n[start_idx:end_idx] * window

        # Estimate noise level
        X_initial = np.fft.rfft(x_slice, n=N)
        noise_level = estimate_noise_level(X_initial)
        noise_levels[i] = noise_level

        # Apply adaptive filtering
        x_filtered = adaptive_filter(x_slice, noise_level)

        # Compute FFT of filtered signal
        X_m = np.fft.rfft(x_filtered, n=N)

        # Get magnitude spectrum and apply frequency mask
        magnitude_spectrum = np.abs(X_m)
        masked_spectrum = magnitude_spectrum.copy()
        masked_spectrum[~freq_mask] = 0

        # Find and interpolate peak
        peak_idx = np.argmax(masked_spectrum)
        interpolated_peak_idx = parabolic_interpolation(masked_spectrum, peak_idx)
        freq = interpolated_peak_idx * fs / N

        # Apply Kalman filtering to frequency estimate
        if prev_freq is None:
            prev_freq = freq

        freq, prev_var = kalman_update(freq, prev_freq, prev_var)
        prev_freq = freq

        # Store results
        freqs[i] = freq
        timestamps[i] = (start_idx + N / 2) / fs

    # Final cleanup
    valid_mask = timestamps > 0
    timestamps = timestamps[valid_mask]
    freqs = freqs[valid_mask]

    # Apply final median filter for smoothing
    kernel_size = min(5, len(freqs))
    if kernel_size % 2 == 0:
        kernel_size += 1
    freqs = signal.medfilt(freqs, kernel_size)

    return timestamps, freqs