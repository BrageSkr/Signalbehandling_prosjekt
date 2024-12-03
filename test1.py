from baseline import freq_detection
import matplotlib.pyplot as plt
import numpy as np
#  parameters
fs = 10000  # Samples/second
N = 4*fs  # Samples
f = 300.5  # Hertz
f2= 600.8 #Hertz
A = 1  # Volts
noise_var = 1 # V^2


t_n = np.arange(N)/fs
signal = A*np.sin(2*np.pi*f*t_n)
noise = np.random.normal(scale=np.sqrt(noise_var), size=N)
x_n = signal+noise
x_n2 =  A*np.sin(2*np.pi*f2*t_n) + noise
x_total = np.concatenate((x_n, x_n2))

t_est, f_est = freq_detection(x_total, fs, N=1024)
t_est2, f_est2 = freq_detection(x_total, fs, N=2048)
plt.close(1)
plt.figure(1)

plt.plot(t_est, f_est, label="Frequency estimates (1024 samples)")
plt.plot(t_est2, f_est2, label="Frequency estimates (2048 samples)")
transition_time = N/fs
plt.plot([0,4,4, t_est[-1]], [f, f,f2,f2], label="True frequency")


plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('Frequency Detection Results')
plt.grid(True)
plt.legend(loc=(0, 0.80))

plt.xlim(3.5, 4.5)
plt.show()
print(f_est)