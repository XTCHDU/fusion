# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import re

class Center:
    """
    数据中心类包括方法：
    接收参数列表
    计算重复参数个数
    """
    def __init__(self):
       self.data = pd.DataFrame()
    def getKey(self):
        temp = self.data.index.tolist()
        ans = []
        for key in temp:
            if re.search(r'false',key):
                ans.append(1)
            else:
                ans.append(0)
        return ans
    def load_data(self,param_list):
        self.data = pd.DataFrame(param_list).T

    def search(self):
        data = self.data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        kmeans = KMeans(2)
        data = kmeans.fit_transform(data)
        return kmeans.labels_

    def run(self):
        true_data = self.getKey()
        est_data = self.search()
        count = 0
        for i in range(len(true_data)):
            if true_data[i] == est_data[i]:
                count += 1
        rate = 1.0 * count / len(true_data)
        if rate < 0.5:
            est_data = [1-x for x in est_data]
            rate = 1-rate
        print "分辨准确率为 %.3f%%"%(rate*100)
        for i in range(len(true_data)):
            print "第{}组 ，正确为{} , 估计为{}".format(i+1, ("欺骗干扰信号" if true_data[i]==1 else "真实回波信号" ) , ("欺骗干扰信号" if est_data[i]==1 else "真实回波信号" ))
            print "估计正确" if true_data[i]==est_data[i] else "估计错误"
            print ""
        return rate*100