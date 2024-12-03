import numpy as np
import matplotlib.pyplot as plt
from final_algorithm import freq_detection

def generate_dynamic_signal(freq_changes, duration=2.0, fs=44100):
    t = np.linspace(0, duration, int(fs * duration))
    signal = np.zeros_like(t)
    actual_freqs = np.zeros_like(t)
    freq_changes = sorted(freq_changes)

    for i in range(len(freq_changes)):
        start_time = freq_changes[i][0]
        freq = freq_changes[i][1]
        end_time = freq_changes[i + 1][0] if i < len(freq_changes) - 1 else duration
        mask = (t >= start_time) & (t < end_time)
        t_segment = t[mask] - start_time
        signal[mask] = np.sin(2 * np.pi * freq * t_segment)
        actual_freqs[mask] = freq
        signal[mask] += 0.3 * np.sin(2 * np.pi * 2 * freq * t_segment)
        signal[mask] += 0.1 * np.sin(2 * np.pi * 3 * freq * t_segment)

    window_size = int(0.01 * fs)
    signal = np.convolve(signal, np.hanning(window_size) / np.sum(np.hanning(window_size)), mode='same')
    return signal, t, actual_freqs

def test_freq_detection_dynamic(freq_changes, duration=2.0, fs=44100, snr_db=20):
    # Generate signal
    pure_signal, t, actual_freqs = generate_dynamic_signal(freq_changes, duration, fs)
    pure_signal = pure_signal / np.max(np.abs(pure_signal))

    # Add noise
    signal_power = np.mean(pure_signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(pure_signal))
    noisy_signal = pure_signal + noise

    # Detect frequencies
    timestamps, freqs = freq_detection(noisy_signal, fs, N=2048)
    timestamps = np.array(timestamps)
    freqs = np.array(freqs)

    # Calculate middle sample deltas
    middle_deltas = []
    for i in range(len(freq_changes) - 1):
        start_time = freq_changes[i][0]
        end_time = freq_changes[i + 1][0]
        mask = (timestamps >= start_time) & (timestamps < end_time)
        if not np.any(mask):
            continue

        actual_freq = freq_changes[i][1]
        detected_freqs = freqs[mask]

        # Get middle samples
        mid_point = len(detected_freqs) // 2
        start_idx = mid_point - 1
        end_idx = mid_point + 1
        if start_idx < 0:
            start_idx = 0
            end_idx = min(10, len(detected_freqs))
        elif end_idx > len(detected_freqs):
            end_idx = len(detected_freqs)
            start_idx = max(0, end_idx - 10)

        middle_freqs = detected_freqs[start_idx:end_idx]
        delta = np.abs(middle_freqs - actual_freq)
        middle_deltas.append(delta)

    # Calculate statistics
    all_deltas = np.concatenate(middle_deltas)
    mean_delta = np.mean(all_deltas)
    max_delta = np.max(all_deltas)

    # Plot frequency tracking
    plt.figure(figsize=(12, 6))
    plt.plot(t, actual_freqs, 'r--', label='True Frequency', alpha=0.7)
    plt.plot(timestamps, freqs, 'b-', label='Detected Frequency')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Frequency Tracking Performance')
    plt.legend()
    plt.ylim(0, max(max(actual_freqs), max(freqs)) * 1.1)

    plt.tight_layout()
    plt.show()

    # Print performance metrics
    print("\nMiddle Samples Performance Metrics:")
    print(f"Mean Delta (Middle Samples): {mean_delta:.2f} Hz")
    print(f"Max Delta (Middle Samples): {max_delta:.2f} Hz")
    for i, deltas in enumerate(middle_deltas):
        print(f"Segment {i + 1} Mean Delta: {np.mean(deltas):.2f} Hz")

if __name__ == "__main__":
    freq_changes = [
        (0.0, 440.8),
        (0.5, 880.3),
        (1.0, 660.9),
        (1.5, 440.2)
    ]
    test_freq_detection_dynamic(
        freq_changes=freq_changes,
        duration=2.0,
        fs=44100,
        snr_db=20
    )