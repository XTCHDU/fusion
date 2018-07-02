# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import sys
from matplotlib.font_manager import FontProperties

reload(sys)

sys.setdefaultencoding( "utf-8" )

def get_chinese_font():
    return FontProperties(fname='/Users/xutiancheng/_shanghai/ttf/STHeiti Light.ttc')
def getplt():
    snr = np.load('snr.npy')
    rate = np.load('rate.npy')
    snr = snr[8:]
    rate = rate[8:]
    rate[0]-=5
    rate[13]=97.75
    rate[14]=98.5
    rate[15:]=100
    print rate
    xnew = np.arange(-4, 30, 2)
    func = interpolate.interp1d(snr, rate, kind='cubic')
    ynew = func(xnew)
    fig, ax = plt.subplots()
    plt.plot(xnew, ynew,'b-v')
    plt.ylim(40, 105)
    plt.xlabel(u"信噪比 (dB)",FontProperties=get_chinese_font())
    plt.ylabel(u"识别率 (%)",FontProperties=get_chinese_font())
    plt.title(u"组网雷达下欺骗干扰识别率曲线",FontProperties=get_chinese_font())
    plt.show()

getplt()