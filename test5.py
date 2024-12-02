import matplotlib.pyplot as plt
import numpy as np


def freq_detection_basic(x_n, fs, N=1024):
    timestamps = []
    freqs = []
    for window_start in range(0, len(x_n), N):
        window_end = min(window_start + N, len(x_n))
        x_slice = x_n[window_start:window_end]
        X_m = np.fft.rfft(x_slice, n=N)
        X_m[0] = 0
        peak_idx = np.argmax(np.abs(X_m))
        freqs.append(peak_idx / N * fs)
        timestamps.append(window_end / fs)
    return np.array(timestamps), np.array(freqs)


def freq_detection_interp(x_n, fs, N=1024):
    timestamps = []
    freqs = []
    for window_start in range(0, len(x_n), N):
        window_end = min(window_start + N, len(x_n))
        x_slice = x_n[window_start:window_end]
        X_m = np.fft.rfft(x_slice, n=N)
        X_m[0] = 0
        peak_idx = np.argmax(np.abs(X_m))

        if peak_idx > 0 and peak_idx < len(X_m) - 1:
            alpha = np.log(np.abs(X_m[peak_idx - 1]))
            beta = np.log(np.abs(X_m[peak_idx]))
            gamma = np.log(np.abs(X_m[peak_idx + 1]))
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            m_peak = peak_idx + p
        else:
            m_peak = peak_idx

        freqs.append(m_peak / N * fs)
        timestamps.append(window_end / fs)
    return np.array(timestamps), np.array(freqs)


# Test parameters
fs = 1000
N = 4 * fs
A = 1
n_points = 100
fixed_f = 650.5  # Fixed frequency between bins

delta_f_basic = np.zeros(n_points)
delta_f_interp = np.zeros(n_points)
f_est_basic_array = np.zeros(n_points)
f_est_interp_array = np.zeros(n_points)
snr = np.zeros(n_points)
variance_basic = np.zeros(n_points)
variance_interp = np.zeros(n_points)

for n in range(n_points):
    noise_var = 0.001 * (1 + n)
    snr[n] = 1 / (A ** 2 / noise_var)
    t_n = np.arange(N) / fs
    signal = A * np.sin(2 * np.pi * fixed_f * t_n)
    noise = np.random.normal(scale=np.sqrt(noise_var), size=N)
    x_n = signal + noise

    _, f_est_basic = freq_detection_basic(x_n, fs, N=2048)
    _, f_est_interp = freq_detection_interp(x_n, fs, N=2048)

    f_est_basic_array[n] = np.mean(f_est_basic)
    f_est_interp_array[n] = np.mean(f_est_interp)

    delta_f_basic[n] = np.abs(fixed_f - np.mean(f_est_basic))
    delta_f_interp[n] = np.abs(fixed_f - np.mean(f_est_interp))

    variance_basic[n] = np.std(f_est_basic)
    variance_interp[n] = np.std(f_est_interp)

total_error_basic = np.sum(delta_f_basic)
total_error_interp = np.sum(delta_f_interp)

plt.figure(figsize=(12, 12))

# Original plots
plt.subplot(3, 1, 1)
plt.semilogx(snr, f_est_basic_array, 'b-', label='Without Interpolation')
plt.semilogx(snr, f_est_interp_array, 'r-', label='With Interpolation')
plt.axhline(y=fixed_f, color='g', linestyle='--', label=f'True Frequency ({fixed_f} Hz)')
plt.fill_between(snr, f_est_basic_array - variance_basic,
                 f_est_basic_array + variance_basic, alpha=0.3, color='b')
plt.fill_between(snr, f_est_interp_array - variance_interp,
                 f_est_interp_array + variance_interp, alpha=0.3, color='r')
plt.grid(True)
plt.legend()
plt.ylabel('Frequency (Hz)')
plt.title('Frequency Estimates vs 1/SNR')

plt.subplot(3, 1, 2)
plt.loglog(snr, delta_f_basic, 'b.-', label=f'Without Interpolation (Total Error: {total_error_basic:.3f})')
plt.loglog(snr, delta_f_interp, 'r.-', label=f'With Interpolation (Total Error: {total_error_interp:.3f})')
plt.grid(True)
plt.xlabel('1/SNR')
plt.ylabel('Frequency Error (Hz)')
plt.legend()
plt.title('Frequency Error vs 1/SNR')

# Additional error plot
plt.subplot(3, 1, 3)
plt.plot(snr, delta_f_basic, 'b.-', label='Without Interpolation')
plt.plot(snr, delta_f_interp, 'r.-', label='With Interpolation')
plt.grid(True)
plt.xlabel('1/SNR')
plt.ylabel('Frequency Error (Hz)')
plt.legend()
plt.title('Linear Scale Frequency Error vs 1/SNR')

plt.tight_layout()
plt.show()