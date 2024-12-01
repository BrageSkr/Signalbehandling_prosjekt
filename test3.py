from test_ai5_music import freq_detection
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

fs, audio_data = wavfile.read('sample_audio/A4_oboe.wav')
audio_data = audio_data[10000:-10000]
audio_data = audio_data.astype(float)
audio_data = audio_data / np.max(np.abs(audio_data))


n_points = 100
delta_f = np.zeros(n_points)
f_estarray = np.zeros(n_points)
variance = np.zeros(n_points)

signal_power = np.mean(audio_data**2)
true_freq = 440

inv_snr = np.logspace(-1, 2, n_points)
snr = 1/inv_snr


for n in range(n_points):
    noise_var = 1 / (snr[n]*2)
    noise = np.random.normal(scale=np.sqrt(noise_var), size=len(audio_data))
    x_n = audio_data + noise
    t_est, f_est = freq_detection(x_n, fs, 2048)
    f_estarray[n] = np.mean(f_est)
    variance[n] = np.std(f_est)
    delta_f[n] = np.abs(true_freq - f_estarray[n])

plt.close(1)
plt.figure(1, figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.semilogx(inv_snr, f_estarray)
plt.fill_between(inv_snr,f_estarray - variance,f_estarray + variance,alpha=0.3)
plt.grid(True)
plt.xlabel('1/SNR')
plt.ylabel('Frequency Estimate (Hz)')
plt.title('Frequency Estimate vs 1/SNR')
plt.xlim(0.1, 100)
plt.ylim(0, 15000)

plt.subplot(2, 1, 2)
plt.loglog(inv_snr, delta_f, 'b.-')
plt.grid(True)
plt.xlabel('1/SNR')
plt.ylabel('Frequency Estimation Error (Hz)')
plt.title('Frequency Estimation Error vs 1/SNR')
plt.xlim(0.1, 100)
plt.ylim(0.1, 10000)

plt.tight_layout()
plt.show()
