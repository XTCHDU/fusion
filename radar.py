# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal as sig

PI = math.pi


def decorater(func):
    def wrapper(*args, **kwargs):
        print "{} has started!".format(func.__name__)
        return func(*args, **kwargs)

    return wrapper


class Transmitter:
    fm = 1e6  # 最高频率
    waveLength = 1000  # 采样点数
    waveTime = 0.001  # 脉冲长度

    def __init__(self):
        pass

    @decorater
    def trans_wave(self):
        k = self.fm / self.waveTime
        time_space = np.linspace(0, self.waveTime, self.waveLength)
        ans = np.zeros([self.waveLength])
        for index, t in enumerate(time_space):
            ans[index] = math.cos(2 * PI * 0.5 * k * t * t)
        return ans


class BaseRadar(Transmitter):
    """
    雷达类包括方法：
    发送波形
    接收波形
    计算参数
    上传数据库

    """
    _count = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.no = BaseRadar._count
        BaseRadar._count += 1


radar_list = [BaseRadar(1, 1) for i in range(10)]
for radar in radar_list:
    print radar.no
plt.plot(radar.trans_wave())
plt.show()
