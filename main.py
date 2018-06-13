# -*- coding: utf-8 -*-
from radar import BaseRadar
import datafusion
import random
import Model
import matplotlib.pyplot as plt
import method
import numpy as np

import dnn
MAX_PARAM_LOW = -1
MAX_PARAM_HIGH = 1
RADAR_NUM = 1
TRUE_TARGET_NUM = 15
FALSE_TARGET_NUM = 15
FALSE_TRANS_POWER = 20

def test():
    x_list = []
    y_list = []
    epoch = 50000
    for _ in range(epoch):
        print(_)
        est = Model.HammersteinWiener(B=[random.uniform(MAX_PARAM_LOW, MAX_PARAM_HIGH) for _ in range(3)],
                                      b=[random.uniform(MAX_PARAM_LOW, MAX_PARAM_HIGH) for _ in range(3)],
                                      h=[random.uniform(MAX_PARAM_LOW, MAX_PARAM_HIGH) for _ in range(3)])
        est.para_init()
        x_list.append(np.append(x, est.run(x, norm=False)))
        y_list.append(est.B + est.b + est.h)

    x_list = np.array(x_list)
    y_list = np.array(y_list)
    np.save("./x_list.npy", x_list)
    np.save("./y_list.npy", y_list)

    # x_list = np.load("x_list.npy")
    # y_list = np.load("y_list.npy")
    ester = method.ML()
    ester.train(x_list[:epoch * 2 // 3], y_list[:epoch * 2 // 3])
    print(ester.mse(ester.predict(x_list[epoch * 2 // 3:]), y_list[epoch * 2 // 3:]))
    ester.save("./model/Gradient_{}.m".format(epoch))
def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

def get_ans(SNR):
    true_radar_list = [BaseRadar(random.randint(0, 200), random.randint(0, 200)) for i in range(TRUE_TARGET_NUM)]
    false_radar_list = [BaseRadar(random.randint(0, 200), random.randint(0, 200)) for i in range(FALSE_TARGET_NUM)]
    HW = Model.HammersteinWiener(B=[1, 0.5, 0.3], b=[1, -0.3, 0.5], h=[1, -0.5, 0.6])  ###
    receive_signal_to_param = {}
    origin_signal = true_radar_list[0].trans_wave()#原始信号

    ##产生真实回波信号，估计其参数
    for index in range(RADAR_NUM):
        for index_radar, radar in enumerate(true_radar_list):
            x = origin_signal * (FALSE_TRANS_POWER+random.normalvariate(0, 1))
            noise = wgn(x, SNR)
            x = x+noise
            receive_signal_to_param["radar{}true{}".format(index,index_radar)] = dnn.dnn_output(np.append(origin_signal,x))

    ##产生虚假回波信号，估计其参数

    for index in range(RADAR_NUM):
        for index_radar ,radar in enumerate(false_radar_list):
            x = HW.run(origin_signal) * FALSE_TRANS_POWER
            noise = wgn(x, SNR)
            x = x+noise
            receive_signal_to_param["radar{}false{}".format(index,index_radar)] = dnn.dnn_output(np.append(origin_signal,x))
    ## 数据中心进行判断
    center = datafusion.Center()
    center.load_data(receive_signal_to_param)
    ans = center.run()

if __name__ == "__main__":
    x = []
    y = []
    for snr in range(-20,24,4):
        y.append(get_ans(snr))
        x.append(snr)
    np.save("snr.npy",x)
    np.save("rate.npy",y)
    plt.figure()
    plt.plot(x,y)
    plt.show()
