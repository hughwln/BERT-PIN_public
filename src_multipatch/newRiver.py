#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 6, 2022
Describe: Load and process the New River data set.
@author: Yi Hu
Email: yhu28@ncsu.edu
"""
import pandas as pd

import os
from os import walk
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import kmeans_pytorch
from sys import platform

if platform == "linux" or platform == "linux2":
    import pickle5 as pickle
elif platform == "win32":
    import pickle as pickle

class newRiver():

    def __init__(self):
        self.dfs = []

    def read_data_transformer(self, nuser=1000, k=100):
        if platform == "linux" or platform == "linux2":
            filepath = '/home/yhu28/Downloads/BadSMsFixedv2/'
            t_path = '~/Documents/Code/Data/new_river/'
        elif platform == "darwin":
            filepath = 'C:\\Users\\yhu28\\Downloads\\BadSMsFixedv2\\'
            t_path = 'C:\\Users\\yhu28\\Documents\\Code\\Data\\new_river\\'
        elif platform == "win32":
            filepath = 'C:\\Users\\yhu28\\Downloads\\BadSMsFixedv2\\'
            t_path = 'C:\\Users\\yhu28\\Documents\\Code\\Data\\new_river\\'

        t_file = t_path + 'temp_Boone.csv'

        d = pd.read_csv(t_file)
        d.index = pd.to_datetime(d["Date"])
        d = d.loc['2017-12-27':'2020-12-27']

        t = d['Temp'].astype(np.float32)

        # df = pd.DataFrame(index=t.index)

        n = 0
        for (dirpath, dirnames, filenames) in walk(filepath):
            if 'figs' in dirpath:
                continue
            if 'probSMs' in dirpath:
                continue
            if 'removedSMsFixed' in dirpath:
                continue
            for file in filenames:
                if file[-3:] != 'pkl':
                    continue

                fullname = os.path.join(dirpath, file)
                with open(fullname, 'rb') as f:
                    data = pickle.load(f)
                    data = data.loc['2017-12-27':'2020-12-27']
                    missing = data['usage'].isna()
                    n_missing = missing.sum()
                    if n_missing/len(missing) > 0.5:
                        continue
                    elif len(data.index) != len(t.index):
                        continue
                    else:
                        # df[file[3:11]] = data['usage']
                        self.dfs.append(data['usage'])

                        n += 1
                        print(n)

        df = pd.concat(self.dfs, axis=1)
        print("aggregated ", n, " users.")
        df = df.fillna(0)

        data = []
        for i in range(k):
            sample = df.sample(n=nuser, axis='columns')
            sample = sample.sum(axis=1)
            sample = pd.concat([sample, t], axis=1)
            data.append(sample)

        dataset = pd.concat(data, axis=0)

        trainDataFile = '../data/raw_data/NewRiver/NonCVR_LT_2000_2000.csv'
        dataset.to_csv(trainDataFile)


if __name__ == '__main__':
    dataloader = newRiver()
    dataloader.read_data_transformer(nuser=2000, k=2000)
