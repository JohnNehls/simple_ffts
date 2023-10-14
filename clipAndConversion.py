#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft, ifftshift

# Inputs
N         = 256
f0        = 10e6   # Hz
BW        = 5e6    # Hz # used for the square pulse width
F_sample  = 400e6  # Hz # smallest f0/M > BW where M is an integer
CLIP_CUTOFF = 0.7

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

def hard_clip(signal, val):
    '''Soft clip real signal'''
    signal = np.where(signal < -val, -val, signal)
    signal = np.where(signal >  val,  val, signal)
    return signal

def convert_signal_to_complex(signal):
    '''Convert real signal to complex'''
    S = fft(signal)
    S_convert = S*2
    S_convert[int(N/2):] = 0
    s_convert = ifft(S_convert)
    return s_convert

def hard_clip_complex(signal, cutoff):
    '''Clipp based on real signal and return complex signal'''
    s = signal.real                            # convert to real
    s_clip = hard_clip(s, cutoff)                # clip
    S_clip = convert_signal_to_complex(s_clip) # convert back to complex
    return S_clip

f = hard_clip(f_c.real, CLIP_CUTOFF)
f_convert = hard_clip_complex(f_c, CLIP_CUTOFF)

####### PLOT 1: show clipping and spectrum of real signal ########
# Plot time signal
fig, ax = plt.subplots(2, 1)
fig.suptitle('compression of real signal', fontsize=16)
ax[0].plot(t,f.real, '-xk', label="f.real")
ax[0].set_title("time [s]")
ax[0].set_ylabel("amplitude [arb]")
ax[0].grid()
ax[0].legend()

# Calculate the Fourier Transform
Fs = fft(f)
Fsp = fftshift(Fs)/N
k = np.linspace(-1/(2*dt_sample), 1/(2*dt_sample), N)  # frequency

# Plot Fourier signal
ax[1].plot(k, np.abs(Fsp)/max(abs(Fsp)), '-k', label="amp")
ax[1].set_xlabel("frequency [1/s]")
ax[1].set_ylabel("amplitude [arb]")
ax[1].grid()
ax[1].legend()

####### PLOT 2: compare with Real with Complex Conversion #########
F_convert = np.abs(fftshift(fft(f_convert))/N/2)

# # Plot time signal
fig, ax = plt.subplots(3, 1)
fig.suptitle('compare with complex conversion', fontsize=16)
ax[0].plot(t,f, '-k', label="in signal")
ax[0].plot(t,f_convert.real, '--r', label="conv.real")
ax[0].plot(t,f_convert.imag, '--g', label="conv.imag")
ax[0].set_title("time [s]")
ax[0].set_ylabel("amplitude [arb]")
ax[0].grid()
ax[0].legend(loc='upper right')

# Plot time error
ax[1].plot(t,abs(f-f_convert)/abs(f), '-k', label="error")
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("rel error [arb]")
ax[1].legend()
ax[1].grid()

# Plot Fourier signal
ax[2].plot(k, np.abs(Fsp), '-k', label="abs(amp)")
ax[2].plot(k, F_convert, '--r', label="abs(F_conv/2)")
ax[2].set_xlabel("frequency [1/s]")
ax[2].set_ylabel("amplitude [arb]")
ax[2].grid()
ax[2].legend()
