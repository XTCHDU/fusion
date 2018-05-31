# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Center:
    """
    数据中心类包括方法：
    接收参数列表
    计算重复参数个数
    """
    def __init__(self):
       self.data = []

    def load_data(self,param_list):
        self.data.append(param_list)

    def search(self):
        data = self.data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        kmeans = KMeans(10)
        data = kmeans.fit_transform(data)
        return data

