from test_ai5_music import freq_detection
import matplotlib.pyplot as plt
import numpy as np

# Test parameters
fs = 10_000  # Samples/second
N = 4*fs    # Samples for 4 secounds
f = 440    # Hertz
A = 1       # Volts


n_points = 100
delta_f = np.zeros(n_points)
f_estarray = np.zeros(n_points)
snr = np.zeros(n_points)
variance = np.zeros(n_points)

for n in range(n_points):
    noise_var = (1+n)
    snr[n] = 1/(A**2 / noise_var)
    t_n = np.arange(N)/fs
    signal = A*np.sin(2*np.pi*f*t_n)
    noise = np.random.normal(scale=np.sqrt(noise_var), size=N)
    x_n = signal + noise
    t_est, f_est = freq_detection(x_n, fs, 1024)
    f_estarray[n]= np.mean(f_est)
    variance[n] = np.std(f_est)
    delta_f[n] = np.abs(f - np.mean(f_est))


plt.close(1)
plt.figure(1, figsize=(10, 8))


plt.subplot(2, 1, 1)
plt.semilogx(snr, f_estarray)
plt.fill_between(snr,
                f_estarray - variance,
                f_estarray + variance,
                alpha=0.3)
plt.grid(True)
plt.xlabel('1/SNR')
plt.ylabel('Frequency Estimate (Hz)')
plt.title('Frequency Estimate vs 1/SNR')


plt.subplot(2, 1, 2)
plt.loglog(snr, delta_f, 'b.-')
plt.grid(True)
plt.xlabel('1/SNR')
plt.ylabel('Frequency Estimation Error (Hz)')
plt.title('Frequency Estimation Error vs 1/SNR')


plt.tight_layout()
plt.show()