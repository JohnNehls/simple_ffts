#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft, ifftshift

# Inputs
N         = 256
f0        = 10e6  # Hz
BW        = 1e6   # Hz # used for the square pulse width
F_sample  = 400e6  # Hz # smallest f0/M > BW where M is an integer

# Create time array based on Nyquist rate or theorem
Fny           = 2*(f0+BW/2)   # Largest frequency to capture
Ts_ny         = 1/Fny         # Nyquist rate
Ts_nyt        = 1/(BW)        # Nyquist-Shannon Theorem
Ts_nyt_sample = 1/(F_sample)  # center spectrum wit f0 integer multiple of BW_ceter

# dt_sample = Ts_ny
dt_sample = Ts_nyt_sample

t = np.array([i*dt_sample for i in range(N)])

# Signal: rect pulse with with Rayleigh beamwidth = BW
T  = 1/BW    # For a Rect pulse
t0 = t[int(N/2)]  # center pules

# start with complex signal
f_c = np.where(abs(t-t0) <= T, np.exp(1j*2*np.pi*f0*t), 0)

# soft clip
def soft_clip(signal, fact):
    return np.tanh(signal*fact)/fact

f = f_c.real

f = soft_clip(f,30)

# Do not convert to baseband; use the sample replication proptery to convert

# Plot time signal
fig, ax = plt.subplots(2, 1)
fig.suptitle('clipping and spectrum', fontsize=16)
ax[0].plot(t,f.real, '-xk', label="f.real")
ax[0].set_title("time [s]")
ax[0].set_ylabel("amplitude [arb]")
ax[0].legend()

# Calculate the Fourier Transform
Fs = fft(f)
Fsp = fftshift(Fs)/N
k = np.linspace(-1/(2*dt_sample), 1/(2*dt_sample), N)  # frequency

# Plot Fourier signal
ax[1].plot(k, np.abs(Fsp)/max(abs(Fsp)), '-k', label="amp")
ax[1].set_xlabel("frequency [1/s]")
ax[1].set_ylabel("amplitude [arb]")
ax[1].legend()

#################################################################
# "Convert" spectrum of real signal to spectrum for complex signal
# - zero the negative frequencies
# - double the positive frequencies to account for energy loss
# - ifft back to time domain
TMP = np.where(k < 0, 0, 2*Fsp)
tmp = ifft(ifftshift(TMP))*N

# Plot time signal
fig, ax = plt.subplots(2, 1)
fig.suptitle('convert to complex', fontsize=16)
ax[0].plot(t,f, '-k', label="in signal")
ax[0].plot(t,tmp.real, '--r', label="conv.real")
ax[0].plot(t,tmp.imag, '--g', label="conv.imag")
ax[0].set_title("time [s]")
ax[0].set_ylabel("amplitude [arb]")
ax[0].legend()

# Plot Fourier signal
ax[1].plot(k, np.abs(Fsp), '-k', label="abs(sig)")
ax[1].plot(k, np.abs(TMP), '--r', label="abs(conv)")
ax[1].set_xlabel("frequency [1/s]")
ax[1].set_ylabel("amplitude [arb]")
ax[1].legend()
