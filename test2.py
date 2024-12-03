from onlyFilter import freq_detection
import matplotlib.pyplot as plt
import numpy as np

# Test parameters
fs = 10000
N = 4*fs
f = 440.5
A = 1

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
    t_est, f_est = freq_detection(x_n, fs, N=1028)
    f_estarray[n]= np.mean(f_est)
    variance[n] = np.std(f_est)
    delta_f[n] = np.abs(f - np.mean(f_est))

# Calculate error metrics
total_error = np.sum(delta_f)
max_error = np.max(delta_f)
mean_error = np.mean(delta_f)
rmse = np.sqrt(np.mean(delta_f**2))

print(f"Total Error: {total_error:.3f} Hz")
print(f"Maximum Error: {max_error:.3f} Hz")
print(f"Mean Error: {mean_error:.3f} Hz")
print(f"RMSE: {rmse:.3f} Hz")

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