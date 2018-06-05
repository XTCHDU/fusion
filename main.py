# -*- coding: utf-8 -*-
from radar import BaseRadar
import datafusion
import random
import Model
import matplotlib.pyplot as plt
import method
import numpy as np
#import dnn

MAX_PARAM_LOW = -1
MAX_PARAM_HIGH = 1

if __name__ == "__main__":
    radar_list = [BaseRadar(random.randint(0, 200), random.randint(0, 200)) for i in range(10)]
    HW = Model.HammersteinWiener(B=[1, 0, 0], b=[1, 0, 0], h=[1, 0, 0])  ###
    for radar in radar_list:
        x = radar.trans_wave()




    x_list = []
    y_list = []
    epoch = 50000
    for _ in range(epoch):
        print(_)
        est = Model.HammersteinWiener(B=[random.uniform(MAX_PARAM_LOW, MAX_PARAM_HIGH) for _ in range(3)],
                                      b=[random.uniform(MAX_PARAM_LOW, MAX_PARAM_HIGH) for _ in range(3)],
                                      h=[random.uniform(MAX_PARAM_LOW, MAX_PARAM_HIGH) for _ in range(3)])
        est.para_init()
        x_list.append(np.append(x, est.run(x, norm = False)))
        y_list.append(est.B + est.b + est.h)

    x_list = np.array(x_list)
    y_list = np.array(y_list)
    np.save("./x_list.npy", x_list)
    np.save("./y_list.npy", y_list)

    #x_list = np.load("x_list.npy")
    #y_list = np.load("y_list.npy")
    ester = method.ML()
    ester.train(x_list[:epoch * 2 // 3], y_list[:epoch * 2 // 3])
    print(ester.mse(ester.predict(x_list[epoch * 2 // 3:]), y_list[epoch * 2 // 3:]))
    ester.save("./model/Gradient_{}.m".format(epoch))
