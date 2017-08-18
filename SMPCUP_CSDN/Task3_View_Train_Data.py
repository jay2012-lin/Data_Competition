#-*- coding:UTF-8 -*-
# author:jaylin
# time:17-7-8 下午10:25
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

task3_trian_data = '../data/SMPCUP2017_TrainingData_Task3.txt'

def view_task3_train_data():
    train_data = pd.read_csv(task3_trian_data,sep='',index_col=0,header=None)
    # print train_data
    bins = np.linspace(0,0.015,11)
    # print bins
    plt.hist(train_data.values,bins=bins)
    plt.show()
    print train_data.describe()


view_task3_train_data()