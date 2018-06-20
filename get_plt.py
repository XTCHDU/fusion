# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def getplt():
    snr = np.load('snr.npy')
    rate = np.load('rate.npy')
    snr = snr[8:]
    rate = rate[8:]
    xnew = np.arange(-4, 30, 0.1)
    func = interpolate.interp1d(snr, rate, kind='cubic')
    ynew = func(xnew)
    fig, ax = plt.subplots()
    plt.plot(xnew, ynew)
    plt.grid()
    plt.ylim(0, 100)
    plt.xlabel("SNR (dB)")
    plt.ylabel("rate (%)")
    plt.title("Result")
    plt.show()

getplt()