#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
I = 1j

# Inputs
N=150
f0 = 15e6 # Hz
BW = 20e6 # Hz

# Create time array based on Nyquist rate or theorem
Fny = 2*(f0+BW/2) # Largest frequency to capture
Ts_ny = 1/Fny     # Nyquist rate
Ts_nyt = 1/(BW)   # Nyquist-Shannon Theorem

Ts = Ts_nyt

t = np.array([i*Ts for i in range(N)])

# Signal: rect pulse with with Rayleigh beamwidth = BW
T  = 1/BW         # For a Rect pulse
t0 = N/2*Ts

f_signal = np.where(abs(t-t0) <= T, np.exp(I*2*np.pi*f0*t), 0)

# Convert signal to Baseband
fmod = f_signal*np.exp(-I*2*np.pi*f0*t) # Note the negative kernel
f = fmod

# Plot time signal
fig, ax = plt.subplots(2, 1)
ax[0].plot(t,abs(f), '-k')
ax[0].plot(t,f.real, '-xr')
ax[0].set_title("time [s]")
ax[0].set_ylabel("amplitude [arb]")

# Calculate the Fourier Transform
Fs = fft(f)
Fsp = fftshift(Fs)/N
k = np.linspace(-1/(2*Ts), 1/(2*Ts), N)  # frequency

# Plot Fourier signal
ax[1].plot(k, np.abs(Fsp), '-k', label="amp")
ax[1].set_xlabel("frequency [1/s]")
ax[1].set_ylabel("amplitude [arb]")
