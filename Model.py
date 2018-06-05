import numpy as np
from radar import decorater
import main
import random
import method
def scaler(x):
    x = np.array(x)
    xmin = np.min(x)
    xmax = np.max(x)
    x = (x-xmin)/(xmax-xmin)
    return x
class HammersteinWiener:
    B = []
    b = []
    h = []

    def __init__(self, B, b, h):
        self.B = B
        self.b = b
        self.h = h

    @decorater
    def run(self, x, norm = False):
        v = self.f(x)
        u = self.H(v)
        y = self.g(u)
        if norm:
            y = scaler(y)
        return y

    def setB(self, index, value):
        self.B[index] = value

    def setb(self, index, value):
        self.b[index] = value

    def seth(self, index, value):
        self.h[index] = value

    def f(self, x):
        # x = np.array(x)
        y = np.zeros(x.shape)
        for n, _x in enumerate(x):
            temp = 0
            for i in range(len(self.b)):
                temp += self.b[i] * pow(_x, 2 * i + 1)
            y[n] = temp
        return y

    def g(self, x):
        # x = np.array(x)
        y = np.zeros(x.shape)
        for n, _x in enumerate(x):
            temp = 0
            for i in range(len(self.B)):
                temp += self.B[i] * pow(_x, 2 * i + 1)
            y[n] = temp
        return y

    def H(self, x):
        # x = np.array(x)
        y = np.zeros(x.shape)
        for n, _x in enumerate(x):
            temp = 0
            for i in range(len(self.h)):
                temp += self.h[i] * x[n - i]
            y[n] = temp
        return y
    def para_init(self):
        self.b[0] = 1
        self.B[0] = 1
        self.h[0] = 1
    @decorater
    def find_way(self, x, y):
        MAX_PARAM_LOW = main.MAX_PARAM_LOW
        MAX_PARAM_HIGH = main.MAX_PARAM_HIGH
        est = HammersteinWiener(B=[random.uniform(MAX_PARAM_LOW, MAX_PARAM_HIGH) for _ in range(len(self.B))],
                                b=[random.uniform(MAX_PARAM_LOW, MAX_PARAM_HIGH) for _ in range(len(self.b))],
                                h=[random.uniform(MAX_PARAM_LOW, MAX_PARAM_HIGH) for _ in range(len(self.h))])
        est.setB(0, 1)
        est.setb(0, 1)
        est.seth(0, 1)



    def __str__(self):
        ans = ""
        ans += "\033[0;33mb =\033[0m {}\n".format(self.b)
        ans += "\033[0;33mB =\033[0m {}\n".format(self.B)
        ans += "\033[0;33mh =\033[0m {}\n".format(self.h)
        return ans
