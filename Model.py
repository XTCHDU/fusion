import numpy as np

class HammersteinWiener:

    B = []
    b = []
    h = []
    def __init__(self,B,b,h):
        self.B = B
        self.b = b
        self.h = h

    def run(self, x):
        v = self.f(x)
        u = self.H(v)
        y = self.g(u)
        return y


    def f(self, x):
        #x = np.array(x)
        y = np.zeros(x.shape)
        for n,_x in enumerate(x):
            temp = 0
            for i in range(len(self.b)):
                temp += self.b[i]  * pow(_x, 2*i+1)
            y[n] = temp
        return y

    def g(self, x):
        #x = np.array(x)
        y = np.zeros(x.shape)
        for n,_x in enumerate(x):
            temp = 0
            for i in range(len(self.B)):
                temp += self.B[i]  * pow(_x, 2*i+1)
            y[n] = temp
        return y

    def H(self, x):
        #x = np.array(x)
        y = np.zeros(x.shape)
        for n,_x in enumerate(x):
            temp = 0
            for i in range(len(self.h)):
                temp += self.h[i] * x[n-i]
            y[n] = temp
        return y
