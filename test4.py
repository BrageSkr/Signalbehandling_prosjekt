import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from test_ai4 import freq_detection


def generate_dynamic_signal(freq_changes, duration=2.0, fs=44100):
    """
    Generate a signal with frequency changes at specified times.

    Parameters:
    freq_changes - list of tuples [(time1, freq1), (time2, freq2), ...]
    duration - total signal duration in seconds
    fs - sampling frequency in Hz

    Returns:
    signal, time_vector, actual_freqs
    """
    t = np.linspace(0, duration, int(fs * duration))
    signal = np.zeros_like(t)
    actual_freqs = np.zeros_like(t)

    # Sort frequency changes by time
    freq_changes = sorted(freq_changes)

    # Generate signal for each segment
    for i in range(len(freq_changes)):
        start_time = freq_changes[i][0]
        freq = freq_changes[i][1]

        # Find end time (either next change or signal end)
        end_time = freq_changes[i + 1][0] if i < len(freq_changes) - 1 else duration

        # Create mask for this time segment
        mask = (t >= start_time) & (t < end_time)

        # Generate signal for this segment
        t_segment = t[mask] - start_time
        signal[mask] = np.sin(2 * np.pi * freq * t_segment)
        actual_freqs[mask] = freq

        # Add harmonics
        signal[mask] += 0.3 * np.sin(2 * np.pi * 2 * freq * t_segment)
        signal[mask] += 0.1 * np.sin(2 * np.pi * 3 * freq * t_segment)

    # Apply smooth transitions using a short window
    window_size = int(0.01 * fs)  # 10ms window
    signal = np.convolve(signal, np.hanning(window_size) / np.sum(np.hanning(window_size)),
                         mode='same')

    return signal, t, actual_freqs


def test_freq_detection_dynamic(freq_changes, duration=2.0, fs=44100, snr_db=20):
    """
    Test frequency detection with dynamic frequency changes.

    Parameters:
    freq_changes - list of tuples [(time1, freq1), (time2, freq2), ...]
    duration - signal duration in seconds
    fs - sampling frequency in Hz
    snr_db - signal to noise ratio in dB
    """
    # Generate test signal
    pure_signal, t, actual_freqs = generate_dynamic_signal(freq_changes, duration, fs)

    # Normalize signal
    pure_signal = pure_signal / np.max(np.abs(pure_signal))

    # Add noise
    signal_power = np.mean(pure_signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(pure_signal))
    noisy_signal = pure_signal + noise

    # Perform frequency detection
    timestamps, freqs = freq_detection(noisy_signal, fs)

    # Convert timestamps and frequencies to numpy arrays for comparison
    timestamps = np.array(timestamps)
    freqs = np.array(freqs)

    # Plotting
    plt.figure(figsize=(15, 10))

    # Plot 1: Frequency tracking
    plt.subplot(2, 1, 1)
    plt.plot(t, actual_freqs, 'r--', label='True Frequency', alpha=0.7)
    plt.plot(timestamps, freqs, 'b-', label='Detected Frequency')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Frequency Tracking Performance')
    plt.legend()
    plt.ylim(0, max(max(actual_freqs), max(freqs)) * 1.1)  # Set y-axis limit

    # Plot 2: Signal waveform with frequency changes marked
    plt.subplot(2, 1, 2)
    plt.plot(t, noisy_signal, 'b-', label=f'Signal (SNR={snr_db}dB)', alpha=0.7)

    # Mark frequency change points
    for change_time, freq in freq_changes:
        plt.axvline(x=change_time, color='r', linestyle='--', alpha=0.5)
        plt.text(change_time, np.max(noisy_signal), f'{freq}Hz',
                 rotation=90, verticalalignment='bottom')

    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Signal Waveform with Frequency Change Points')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Calculate and print performance metrics
    print("\nPerformance Metrics:")

    # Calculate response time for each frequency change
    for i in range(len(freq_changes) - 1):
        change_time = freq_changes[i][0]
        new_freq = freq_changes[i][1]

        # Find detected frequencies around the change point
        change_time_float = float(change_time)  # Convert to float explicitly
        mask = (timestamps >= change_time_float) & (timestamps <= change_time_float + 0.1)

        if np.any(mask):
            detected_freqs = freqs[mask]
            time_points = timestamps[mask]

            # Find point where frequency gets within 5% of target
            target_range = new_freq * 0.05
            settled_idx = np.where(np.abs(detected_freqs - new_freq) <= target_range)[0]

            if len(settled_idx) > 0:
                response_time = time_points[settled_idx[0]] - change_time_float
                print(f"Response time at {change_time}s: {response_time * 1000:.1f}ms")
            else:
                print(f"Did not settle within 100ms at {change_time}s")


# Example usage
if __name__ == "__main__":
    # Define frequency changes: (time, frequency)
    freq_changes = [
        (0.0, 440),  # Start with A4
        (0.5, 880),  # Jump to A5 at 0.5s
        (1.0, 660),  # Jump to E5 at 1.0s
        (1.5, 440)  # Back to A4 at 1.5s
    ]

    # Test with these frequency changes
    test_freq_detection_dynamic(
        freq_changes=freq_changes,
        duration=2.0,
        fs=44100,
        snr_db=20
    )