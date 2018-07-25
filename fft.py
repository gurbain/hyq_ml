import matplotlib.pyplot as plt
import numpy as np
from random import random

# Signal parameters
n = 1000
t_max = 2.0
t = np.linspace(0, t_max, n)
fs = n / t_max
k = np.arange(n)
T = n / fs

# Create signal
y = np.zeros(t.shape)
freq = [1, 5, 10, 20]
for f in freq:
    a = 5 * random()
    phi = np.pi * 2 * random()
    y += a * np.sin(2*np.pi*f*t + phi)

print y.shape

# Plot FFT
frq = k / T
frq = frq[range(n/2)]
y_pred_fft = np.fft.fft(y) / n
y_pred_fft = y_pred_fft[range(n/2)]


plt.plot(frq, abs(y_pred_fft))
plt.xlabel('Freq (Hz)')
plt.ylabel('|Y(freq)|')
plt.show()