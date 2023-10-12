#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

# Inputs
N         = 500
f0        = 100e6 # Hz
BW        = 20e6  # Hz # used for the square pulse width
BW_sample = 50e6  # Hz # smallest f0/M > BW where M is an integer

# Create time array based on Nyquist rate or theorem
Fny           = 2*(f0+BW/2)   # Largest frequency to capture
Ts_ny         = 1/Fny         # Nyquist rate
Ts_nyt        = 1/(BW)        # Nyquist-Shannon Theorem
Ts_nyt_sample = 1/(BW_sample) # center spectrum wit f0 integer multiple of BW_ceter

dt_sample = Ts_ny/2
#dt_sample = Ts_nyt_sample

t = np.array([i*dt_sample for i in range(N)])

# Signal: rect pulse with with Rayleigh beamwidth = BW
T  = 1/BW         # For a Rect pulse
t0 = N/2*dt_sample

f_signal = np.where(abs(t-t0) <= T, np.exp(1j*2*np.pi*f0*t), 0)
# f_signal = np.zeros(N)
# f_signal[int(N/2)] = 100

f = f_signal
# Do not convert to baseband; use the sample replication proptery to convert

# Plot time signal
fig, ax = plt.subplots(2, 1)
ax[0].plot(t,abs(f), '-k')
ax[0].plot(t,f.real, '-xr')
ax[0].set_title("time [s]")
ax[0].set_ylabel("amplitude [arb]")

# Calculate the Fourier Transform
Fs = fft(f)
Fsp = fftshift(Fs)/N
k = np.linspace(-1/(2*dt_sample), 1/(2*dt_sample), N)  # frequency

# Plot Fourier signal
ax[1].plot(k, np.abs(Fsp)/max(abs(Fsp)), '-k', label="amp")
ax[1].set_xlabel("frequency [1/s]")
ax[1].set_ylabel("amplitude [arb]")
