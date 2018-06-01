# -*- coding: utf-8 -*-
from radar import BaseRadar
import datafusion
import random
import Model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    radar_list = [BaseRadar(random.randint(0, 200), random.randint(0, 200)) for i in range(10)]
    HW = Model.HammersteinWiener(B=[1, 0, 0], b=[1, 0, 0], h=[1, 0, 0])  ###
    for radar in radar_list:
        x = radar.trans_wave()
    HW.find_way(0,0)
