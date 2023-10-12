#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

error = 0.05 # 1 percent

###########################################################
## 1: plot to two and see the difference
## 2: Plot the error and see where they cross the 1% line
###########################################################
plt.rcParams['text.usetex'] = True

theta_array_deg = np.linspace(0, 180, int(1e6))
theta_array_rad = np.deg2rad(theta_array_deg)

fig, ax = plt.subplots(2,1)
# 1)
ax[0].plot(theta_array_deg, 2*np.sin(theta_array_rad/2), '-k',label='2 sin(\theta/2)')
ax[0].plot(theta_array_deg, theta_array_rad, '--r',label='\theta')
ax[0].grid()
ax[0].legend()
ax[0].set_ylabel("exact and approx")

# 2)
ax[1].plot(theta_array_deg, np.deg2rad(theta_array_deg) -
         2*np.sin(theta_array_rad/2), '-k')
ax[1].hlines(error, theta_array_deg.min(), theta_array_deg.max(), 'red','--')
ax[1].grid()
ax[1].set_xlabel("angle [deg]")
ax[1].set_ylabel("error")

###############################
## 3: Newton's Algorithm
###############################
def f(x, error=error):
    return abs(x - 2*np.sin(x/2)) - error

def df(x,error=error,dx=1e-12):
    return (f(x+dx, error) - f(x, error))/dx

x =np.deg2rad(179.00) # initial guess
iterations = 100      # number of interations

for i in range(iterations):
    # find the line
    m = df(x)
    b = f(x) - df(x)*x

    # x where line is zero
    x = -b/m
    print(i, np.rad2deg(x))
