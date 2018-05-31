# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt

PI = math.pi


class Transmitter:
    fm = 1e6           # 最高频率
    waveLength = 1000  # 采样点数
    waveTime = 0.001   # 脉冲长度

    def trans_wave(self):
        k = self.fm / self.waveTime
        time_space = np.linspace(0, self.waveTime, self.waveLength)
        ans = np.zeros([self.waveLength])
        for index, t in enumerate(time_space):
            ans[index] = math.cos(2 * PI * 0.5 * k * t * t)
        return ans


class BaseRadar(Transmitter):
    def __init__(self, x, y):
        self.x = x
        self.y = y


radar = BaseRadar(1, 1)
plt.plot(radar.trans_wave())
plt.show()
