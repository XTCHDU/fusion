import numpy as np
from radar import decorater

class HammersteinWiener:
    B = []
    b = []
    h = []

    def __init__(self, B, b, h):
        self.B = B
        self.b = b
        self.h = h
    @decorater
    def run(self, x):
        v = self.f(x)
        u = self.H(v)
        y = self.g(u)
        return y

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
    @decorater
    def find_way(self, x, y):
        est = HammersteinWiener(B=[0 for _ in range(len(self.B))], b=[0 for _ in range(len(self.b))],
                                h=[0 for _ in range(len(self.h))])

    def __str__(self):
        ans = ""
        ans += "\033[0;33mb =\033[0m {}\n".format(self.b)
        ans += "\033[0;33mB =\033[0m {}\n".format(self.B)
        ans += "\033[0;33mh =\033[0m {}\n".format(self.h)
        return ans