#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  26 2023

@author: Yi Hu
"""

import config
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from numpy import genfromtxt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def fill_nan(feeder, dim):
    n_user = np.size(feeder, axis=1)
    # fill NaN
    for i_day in range(365*2):
        for i_user in range(n_user):
            user_data = feeder[i_day*dim:(i_day+1)*dim, i_user]
            n_NaN = np.sum(np.isnan(user_data))
            if n_NaN>0 and n_NaN<5:
                nans, x= nan_helper(user_data)
                user_data[nans] = np.interp(x(nans), x(~nans), user_data[~nans])
                feeder[i_day*dim:(i_day+1)*dim, i_user] = user_data
            else:
                nans, x= nan_helper(user_data)
                user_data[nans] = 0.0
                feeder[i_day*dim:(i_day+1)*dim, i_user] = user_data
    return feeder
    

def gen_new_river_dataset(file_pth, dim_daily, step, temperature, cvr_info):
    cvr_day_idx = np.array((cvr_info.n_days-1) * dim_daily, dtype=int) # start index of CVR days
    
    norm_near_cvr_idx = np.concatenate((np.arange(150,240,1,dtype=int), np.arange(540,580,1,dtype=int))) * dim_daily
    norm_near_cvr_idx_base = norm_near_cvr_idx+4
    for i in range(12):
        norm_near_cvr_idx = np.concatenate((norm_near_cvr_idx, norm_near_cvr_idx_base))
        norm_near_cvr_idx_base = norm_near_cvr_idx_base + 4
    
    cvr_step_of_day = np.array(dim_daily/24 * cvr_info.n_hours + cvr_info.n_min/reso_min, dtype=int)
    cvr_step_of_all = (cvr_info.n_days-1) * dim_daily + cvr_step_of_day
    
    for i in range(int(len(file_pth)/2)):
        raw_2019 = genfromtxt(file_pth[2*i], delimiter=',')
        raw_2020 = genfromtxt(file_pth[2*i+1], delimiter=',')
        feeder = np.concatenate((raw_2019, raw_2020),axis=0)
        feeder = fill_nan(feeder, dim_daily)
        feeder = np.sum(feeder, axis=1, keepdims=True)
        if i == 0:
            raw_data = feeder
        else:
            raw_data = np.concatenate((raw_data,feeder), axis=1)
    
    # normalize data
    user_max = np.amax(np.nan_to_num(raw_data, nan=0))
    raw_data = raw_data/user_max # normalize by P_max
    
    # split data into daily profile
    num_user = np.size(raw_data, axis=1)
    max_size = int(365*2 * num_user * dim_daily / step)
    norm_set = np.zeros([max_size, dim_daily*4]) # buffer
    norm_near_cvr_set = np.zeros([len(norm_near_cvr_idx), dim_daily*4])
    cvr_set = np.zeros([len(cvr_info)*num_user, dim_daily*4])
    pt_norm = 0 # pointer for normal samples
    pt_cvr = 0 # pointer for cvr samples
    pt_norm_near_cvr = 0 # pointer for normal days near cvr event
    for i_st in range(0, (365*2-1)*dim_daily-step, step):
        T = temperature[i_st : (i_st+dim_daily)] # daily temperature
        for idx_user in range(np.size(raw_data, axis=1)):
            load_gt = raw_data[i_st : (i_st + dim_daily), idx_user]
            if np.sum(np.isnan(load_gt))==0 and np.all(load_gt): # wipe out Nan and 0 samples
                dist2crv = cvr_step_of_all - i_st # distance to CVR event start index of whole years
                if i_st in cvr_day_idx: # CVR event day
                    # mask for CVR period
                    event = np.where(cvr_day_idx==i_st)[0][0]
                    mask = np.zeros(dim_daily, dtype=bool)
                    mask[cvr_step_of_day[event]:cvr_step_of_day[event]+config.DIM_PATCH] = True
                    load_patched = load_gt * ~mask
                    cvr_set[pt_cvr,:] = np.concatenate((load_patched, mask, T, load_gt), axis=0, dtype=float).reshape(1, dim_daily*4)
                    pt_cvr += 1
                    
                elif np.any((dist2crv >= -dim_daily) & (dist2crv <= 0)): 
                    # make sure no overlap with CVR event in normal dataset, 12 is 3-hours cvr duration
                    continue
                else:
                    if config.USE_CENTRAL_MASK:
                        patch_start = dim_daily//2 - config.DIM_PATCH//2
                    else:
                        patch_start = np.random.randint(1, dim_daily-config.DIM_PATCH-1)
                    
                    mask = np.zeros(dim_daily, dtype=bool)
                    mask[patch_start:patch_start + config.DIM_PATCH] = True # hole-True
                    load_patched = load_gt * ~mask
                    norm_set[pt_norm, :] = np.concatenate((load_patched, mask, T, load_gt), axis=0, dtype=float).reshape(1, dim_daily*4)
                    pt_norm += 1
                    
                    if i_st in norm_near_cvr_idx:
                        patch_start = int(dim_daily/24*14) # start from 14:00
                        mask = np.zeros(dim_daily, dtype=bool)
                        mask[patch_start:patch_start + config.DIM_PATCH] = True # hole-True
                        load_patched = load_gt * ~mask
                        norm_near_cvr_set[pt_norm_near_cvr,:] =\
                            np.concatenate((load_patched, mask, T, load_gt), axis=0, dtype=float).reshape(1, dim_daily*4)
                        pt_norm_near_cvr += 1
                    
    norm_set = norm_set[0:pt_norm, :]
    norm_near_cvr_set = norm_near_cvr_set[0:pt_norm_near_cvr]
    print("Total normal samples:", pt_norm)
    print("Total CVR samples:", pt_cvr)
    print("Total normal day near CVR event:", pt_norm_near_cvr)
    
    # split whole dataset into train/dev/test set
    index_random = np.arange(pt_norm)
    np.random.shuffle(index_random)
    
    idx_start = 0
    idx_end = int(pt_norm*config.TRAIN_SET_SIZE)
    train_choice = index_random[idx_start:idx_end]
    
    idx_start = int(pt_norm*config.TRAIN_SET_SIZE)
    idx_end = int(pt_norm*(config.TRAIN_SET_SIZE + config.DEV_SET_SIZE))
    dev_choice = index_random[idx_start:idx_end]
    
    train_set = []
    dev_set = []
    test_set = []
    for i in range(pt_norm):
        if i in train_choice:
            train_set.append(norm_set[i,:])
        elif i in dev_choice:
            dev_set.append(norm_set[i,:])
        else:
            test_set.append(norm_set[i,:])
            
    train_set = np.array(train_set)
    dev_set = np.array(dev_set)               
    test_set = np.array(test_set)

    return train_set, dev_set, test_set, cvr_set, norm_near_cvr_set, user_max

if config.GEN_DATASET:
    
    if config.FEEDER == 'Fayetteville':
        step = config.STEP
        dim_daily = config.DIM_INPUT # daily data dimension
        n_days = 1
        reso_min = 1440/dim_daily
        
        # read raw data
        raw_data = pd.read_csv("../data/raw_data/Fayetteville_2021/Fayetteville_2021_new_filled.csv",index_col='Datetime') 
        temp = raw_data.Temperature.values
        temperature = NormalizeData(temp)
        power = (raw_data.IA.values*raw_data.VAN.values + \
                 raw_data.IB.values*raw_data.VBN.values + \
                 raw_data.IC.values*raw_data.VCN.values)
        # CVR event information
        cvr_info = pd.read_csv('../data/raw_data/Fayetteville_2021/CSV_events_Fayetteville.csv')
        cvr_idx_start = cvr_info.Index_start.values
        cvr_day_idx = np.array((cvr_info.n_days-1) * dim_daily, dtype=int)
        cvr_step_of_day = cvr_info.Index_start.values % dim_daily # cvr event index of a day
        cvr_len = cvr_info.Index_end.values - cvr_info.Index_start.values + 1
        np.random.seed(0) # seed everything for reproducible purpose
        
        
        
        # # winter case
        # norm_near_cvr_idx = np.concatenate((np.arange(0,60,1,dtype=int),
        #                                     np.arange(300,360,1,dtype=int))) * dim_daily
        # norm_near_cvr_idx_base = norm_near_cvr_idx + 1
        # for i in range(12):
        #     norm_near_cvr_idx = np.concatenate((norm_near_cvr_idx, norm_near_cvr_idx_base))
        #     norm_near_cvr_idx_base += 1
          
        # summer case
        norm_near_cvr_idx = np.arange(240,300,1,dtype=int) * dim_daily
        norm_near_cvr_idx_base = norm_near_cvr_idx + 1
        for i in range(12):
            norm_near_cvr_idx = np.concatenate((norm_near_cvr_idx, norm_near_cvr_idx_base))
            norm_near_cvr_idx_base += 1
            
            
        # normalize data
        max_load = np.amax(np.nan_to_num(power, nan=0))
        power = power/max_load # normalize by P_max
        
        # split data into daily profile
        num_user = 1
        max_size = int(365 * num_user * dim_daily / step)
        norm_set = np.zeros([max_size, dim_daily*4]) # input/temperature/gt
        datase_cvr = np.zeros([max_size, dim_daily*4]) 
        norm_near_cvr_set = np.zeros([len(norm_near_cvr_idx), dim_daily*4])
        pt_norm = 0 # pointer for normal samples
        pt_cvr = 0 # pointer for cvr samples
        pt_norm_near_cvr = 0
        
        # visualize temperature distrubution purpose
        T_avg_norm = np.zeros([max_size, 3]) # number of sample / daily_temp / cvr-temp
        T_avg_cvr = np.zeros([max_size, 3])
        
        for i_st in range(0, (365-n_days)*dim_daily-step, step):
            T = temperature[i_st : (i_st + dim_daily)] # temperature data
            load_gt = power[i_st : (i_st + dim_daily)]
            if np.sum(np.isnan(load_gt))==0 and np.all(load_gt): # wipe out Nan and 0 samples
                # make sure no overlap with CVR event
                dist2crv = cvr_idx_start - i_st
                if i_st in cvr_day_idx:
                    event = np.where(cvr_day_idx==i_st)[0][0]
                    mask = np.zeros(dim_daily, dtype=bool)
                    patch_start = cvr_step_of_day[event]
                    mask[patch_start:patch_start + cvr_len[event]] = True
                    load_patched = load_gt * ~mask
                    datase_cvr[pt_cvr,:] = np.concatenate((load_patched, mask, T, load_gt), axis=0, dtype=float).reshape(1, dim_daily*4)
                    T_avg_cvr[pt_cvr, 0] = pt_cvr
                    T_avg_cvr[pt_cvr, 1] = np.mean(T)
                    T_avg_cvr[pt_cvr, 2] = np.mean(T[patch_start:patch_start + cvr_len[event]])
                    pt_cvr += 1
                elif np.any((dist2crv >= -n_days*dim_daily) & (dist2crv <= 12*3)): 
                    # make sure no overlap with CVR event in normal dataset
                    continue
                else:
                    mask = np.zeros(dim_daily, dtype=bool)
                    patch_len = np.random.randint(int(dim_daily/12), int(dim_daily/6))
                    if config.USE_CENTRAL_MASK:
                        patch_start = (dim_daily - patch_len)//2
                    else:
                        patch_start = np.random.randint(1, dim_daily-patch_len-2)
                    mask[patch_start:patch_start + patch_len] = True # hole-True
                    load_patched = load_gt * ~mask
                    norm_set[pt_norm, :] = np.concatenate((load_patched, mask, T, load_gt), axis=0, dtype=float).reshape(1, dim_daily*4)
                    T_avg_norm[pt_norm, 0] = pt_norm
                    T_avg_norm[pt_norm, 1] = np.mean(T)
                    T_avg_norm[pt_norm, 2] = np.mean(T[patch_start:patch_start + patch_len])
                    pt_norm += 1
                    
                    if i_st in norm_near_cvr_idx:
                        patch_start = int(dim_daily/24*5) # start from 13:00-14:00
                        patch_len = np.random.randint(int(dim_daily/12), int(dim_daily/6))
                        mask = np.zeros(dim_daily, dtype=bool)
                        mask[patch_start:patch_start + patch_len] = True # hole-True
                        load_patched = load_gt * ~mask
                        norm_near_cvr_set[pt_norm_near_cvr,:] =\
                            np.concatenate((load_patched, mask, T, load_gt), axis=0, dtype=float).reshape(1, dim_daily*4)
                        pt_norm_near_cvr += 1
        
        norm_set = norm_set[0:pt_norm, :]
        datase_cvr = datase_cvr[0:pt_cvr, :]
        T_avg_norm = T_avg_norm[0:pt_norm, :]
        T_avg_cvr = T_avg_cvr[0:pt_cvr, :]
        
        print("Total normal samples:", pt_norm)
        print("Total CVR samples:", pt_cvr)
        norm_near_cvr_set = norm_near_cvr_set[0:pt_norm_near_cvr]
        print("Total normal day near CVR event:", pt_norm_near_cvr)
        # split whole dataset into train/dev/test set
        index_random = np.arange(pt_norm)
        np.random.shuffle(index_random)
        
        idx_start = 0
        idx_end = int(pt_norm*config.TRAIN_SET_SIZE)
        train_choice = index_random[idx_start:idx_end]
        
        idx_start = int(pt_norm*config.TRAIN_SET_SIZE)
        idx_end = int(pt_norm*(config.TRAIN_SET_SIZE + config.DEV_SET_SIZE))
        dev_choice = index_random[idx_start:idx_end]
        
        train_set = []
        dev_set = []
        test_set = []
        for i in range(pt_norm):
            if i in train_choice:
                train_set.append(norm_set[i,:])
            elif i in dev_choice:
                dev_set.append(norm_set[i,:])
            else:
                test_set.append(norm_set[i,:])
                
        train_set = np.array(train_set)
        dev_set = np.array(dev_set)               
        test_set = np.array(test_set)
        # np.random.shuffle(norm_set)
        # idx_start = 0
        # idx_end = int(pt_norm*config.TRAIN_SET_SIZE)
        # train_set = norm_set[idx_start:idx_end,:]
        
        # idx_start = int(pt_norm*config.TRAIN_SET_SIZE)
        # idx_end = int(pt_norm*(config.TRAIN_SET_SIZE + config.DEV_SET_SIZE))
        # dev_set = norm_set[idx_start:idx_end,:]
        
        # idx_start = int(pt_norm*(config.TRAIN_SET_SIZE + config.DEV_SET_SIZE))
        # test_set  = norm_set[idx_start:,:]
        
        np.save('../data/train_set_Fayetteville_all'+'_step'+str(step)+'.npy', train_set)
        np.save('../data/dev_set_Fayetteville_all' +'_step'+str(step)+'.npy', dev_set)
        np.save('../data/test_set_Fayetteville_all' +'_step'+str(step)+'.npy', test_set)
        np.save('../data/cvr_set_Fayetteville_all' +'_step'+str(step)+'.npy', datase_cvr)
        np.save('../data/norm_near_cvr_set_Fayetteville_all' +'_step'+str(step)+'.npy', norm_near_cvr_set)
        np.save('../data/max_load_Fayetteville_all' +'_step'+str(step)+'.npy', max_load)
                
        print("Fayetteville_2021 Dataset processing finished!")
        
    elif config.FEEDER == 'Pecan2015':
        raw_data = np.load('../data/raw_data/Pecan/Pecan2015/'+config.GROUP+config.USERS+config.RESO+'.npy')
        max_load = np.amax(raw_data)
        temperature = np.genfromtxt('../data/raw_data/Pecan/temperature_Pecan2015_1h.csv', delimiter=',')
        t_lr = np.linspace(0, int(8760*config.DIM_INPUT/24), 8760)
        t_hr = np.linspace(0, int(8760*config.DIM_INPUT/24), int(8760*config.DIM_INPUT/24))
        fnc_intp = interp1d(t_lr, temperature, kind="cubic")
        temperature = fnc_intp(t_hr)
        temperature = temperature / np.amax(temperature)
        power = raw_data/max_load
        max_load = np.amax(power)
        dim_daily = config.DIM_INPUT
        step = config.STEP
        
        max_size = int(365 * dim_daily / step)
        norm_set = np.zeros([max_size, dim_daily*4]) # input/temperature/gt
        pt_norm = 0 # pointer for normal samples
        for i_st in range(0, (365-1)*dim_daily-step, step):
            T = temperature[i_st : (i_st+dim_daily)] # daily temperature
            load_gt = power[i_st : (i_st+dim_daily)]
            if config.USE_CENTRAL_MASK:
                patch_start = dim_daily//2 - config.DIM_PATCH//2
            else:
                peak_index = max(load_gt.argmax(axis=0) - config.DIM_PATCH // 2, 1)
                patch_start = min(peak_index, dim_daily - config.DIM_PATCH - 1)
                # patch_start = np.random.randint(1, dim_daily - config.DIM_PATCH-1)
            mask = np.zeros(dim_daily, dtype=bool)
            mask[patch_start:patch_start + config.DIM_PATCH] = True # hole-True
            load_patched = load_gt * ~mask
            norm_set[pt_norm, :] = np.concatenate((load_patched, mask, T, load_gt), axis=0, dtype=float).reshape(1, dim_daily*4)
            pt_norm += 1
        print("Total normal samples:", pt_norm)
        for i in range(pt_norm):
            max_val = max(norm_set[i,:])
            if max_val == 0:
                print('Bad data in Normal set at:', i)
        # split whole dataset into train/dev/test set, 
        # do not use np.random.shuffle() to total array, will output 0 valus randomly, seems like a bug in numpy
        
        
        if config.N_SAMPLE != 'all':
            print('Randomly select ' + str(config.N_SAMPLE) + ' samples...')
            pt_norm = config.N_SAMPLE
            
        index_random = np.arange(pt_norm)
        np.random.shuffle(index_random)
        idx_start = 0
        idx_end = int(pt_norm*config.TRAIN_SET_SIZE)
        train_choice = index_random[idx_start:idx_end]

        idx_start = int(pt_norm*config.TRAIN_SET_SIZE)
        idx_end = int(pt_norm*(config.TRAIN_SET_SIZE + config.DEV_SET_SIZE))
        dev_choice = index_random[idx_start:idx_end]

        # for i in range(pt_norm):
        #     if i == 0:
        #         train_choice = index_random[i:i+700]
        #         dev_choice = index_random[i+700:i + 900]
        #     elif i % 1000 == 0:
        #         if pt_norm - i < 1000:
        #             d = pt_norm - i
        #             train_choice_ = index_random[i:i + int(d*0.7)]
        #             dev_choice_ = index_random[i + int(d*0.7):i + int(d*0.9)]
        #         else:
        #             train_choice_ = index_random[i:i + 700]
        #             dev_choice_ = index_random[i + 700:i + 900]
        #
        #         train_choice = np.concatenate((train_choice, train_choice_))
        #         dev_choice = np.concatenate((dev_choice, dev_choice_))
        
        train_set = []
        dev_set = []
        test_set = []
        for i in range(pt_norm):
            if i in train_choice:
                train_set.append(norm_set[i,:])
            elif i in dev_choice:
                dev_set.append(norm_set[i,:])
            else:
                test_set.append(norm_set[i,:])
                
        train_set = np.array(train_set)
        dev_set = np.array(dev_set)               
        test_set = np.array(test_set)

        TRAIN_SET_PTH = '../data/train_set_'+config.FEEDER+'_'+config.GROUP+config.USERS+config.RESO+'_step'+ str(config.STEP) + '.npy'
        TEST_SET_PTH  = '../data/test_set_'+config.FEEDER+'_'+config.GROUP+config.USERS+config.RESO+'_step'+str(config.STEP) + '.npy'
        DEV_SET_PTH   = '../data/dev_set_'+config.FEEDER+'_'+config.GROUP+config.USERS+config.RESO+'_step'+ str(config.STEP) + '.npy'
        PEAK_LOAD_PTH = '../data/max_load_'+config.FEEDER+'_'+config.GROUP+config.USERS+config.RESO+'_step'+ str(config.STEP) + '.npy'
        
        np.save(TRAIN_SET_PTH, train_set)
        np.save(TEST_SET_PTH, test_set)
        np.save(DEV_SET_PTH, dev_set)
        np.save(PEAK_LOAD_PTH, max_load)

    elif config.FEEDER == 'Wilson':
        rawdata_dict = pd.read_excel('../data/raw_data/Wilson data/N_residential.xlsx', index_col=0, sheet_name=None, usecols=[0, 7])
        # for sheet, data in rawdata_dict:
        #     rawdata = rawdata_dict[sheet]
        rawdata = pd.concat([rawdata_dict[sheet] for sheet in rawdata_dict.keys()], axis=1)
        rawnp = rawdata.to_numpy()

    elif config.FEEDER == "NewRiver":
        # raw_data = pd.read_csv('../data/raw_data/NewRiver/GANData0.csv')
        # real_loads = raw_data.iloc[:, 1:-1].to_numpy()
        # real_loads = real_loads.sum(axis=1)
        # real_T = raw_data.iloc[:, -1].to_numpy()

        raw_data = pd.read_csv('../data/FeederLevel_LT_' + str(config.nuser) + '_' + str(config.nsample) + '.csv', index_col=0)

        real_loads = raw_data.iloc[:, 0].to_numpy()
        real_T = raw_data.iloc[:, -1].to_numpy()
        pmax = np.amax(real_loads)
        pmin = np.amin(real_loads)
        tmax = np.amax(real_T)

        step = config.STEP
        dim_daily = config.DIM_INPUT

        cvr_set = []
        raw_data.iloc[:, 0] = raw_data.iloc[:, 0] / np.amax(real_loads)
        raw_data.iloc[:, 1] = raw_data.iloc[:, 1] / np.amax(real_T)
        raw_data.index = pd.to_datetime(raw_data.index)

        real_loads = raw_data.iloc[:, 0].to_numpy()
        power = real_loads
        temperature = raw_data.iloc[:, -1].to_numpy()
        max_load = np.amax(power)
        max_size = int(real_loads.shape[0] / step)
        peak_dist = []
        patch_values = np.array([])

        norm_set = np.zeros([max_size, dim_daily * 4])  # input/mask/temperature/gt
        pt_norm = 0  # pointer for normal samples
        for i_st in range(0, real_loads.shape[0] - dim_daily, step):
            T = temperature[i_st: (i_st + dim_daily)]  # daily temperature
            load_gt = power[i_st: (i_st + dim_daily)]
            peak_dist.append(np.amax(load_gt))

            if config.MASK_STRATEGY == "Central":
                patch_start = dim_daily // 2 - config.DIM_PATCH // 2
            elif config.MASK_STRATEGY == "Peak":
                peak_index = max(load_gt.argmax(axis=0) - config.DIM_PATCH // 2, 1)
                patch_start = min(peak_index, dim_daily - config.DIM_PATCH - 1)
            elif config.MASK_STRATEGY == "Random":
                patch_start = np.random.randint(1, dim_daily - config.DIM_PATCH - 1)
            mask = np.zeros(dim_daily, dtype=bool)
            mask[patch_start:patch_start + config.DIM_PATCH] = True  # hole-True
            load_patched = load_gt * ~mask
            norm_set[pt_norm, :] = np.concatenate((load_patched, mask, T, load_gt), axis=0, dtype=float).reshape(1,
                                                                                                                 dim_daily * 4)
            pt_norm += 1

            # patch_values = np.concatenate((patch_values, load_gt[patch_start:patch_start + config.DIM_PATCH]))

        print("Total normal samples:", pt_norm)
        for i in range(pt_norm):
            max_val = max(norm_set[i, :])
            if max_val == 0:
                print('Bad data in Normal set at:', i)
        # split whole dataset into train/dev/test set,
        # do not use np.random.shuffle() to total array, will output 0 valus randomly, seems like a bug in numpy

        # show the distribution of daily peak values.
        figure = plt.figure(2)
        violin_parts = plt.violinplot([[element * pmax for element in peak_dist],
                                       [element * pmax for element in real_loads]],
                                      showmedians=True, showextrema=False, vert=False)
        violin_parts['bodies'][0].set_facecolor('blue')
        violin_parts['bodies'][1].set_facecolor('green')
        plt.yticks(range(1, 3), labels=['Daily Peak Values', 'Load Values'])
        plt.xlabel("Power (kw)")
        # plt.show()
        fn = '../plot/' + config.TAG + '/peak_violinplot.png'
        figure.savefig(fn, dpi=300)
        # =============================================

        if config.N_SAMPLE != 'all':
            print('Randomly select ' + str(config.N_SAMPLE) + ' samples...')
            pt_norm = config.N_SAMPLE

        # pt_norm = int(pt_norm * 0.1)
        # norm_set = norm_set[:pt_norm]

        index_random = np.arange(pt_norm)
        # np.random.shuffle(index_random)

        idx_start = 0
        idx_end = int(pt_norm*config.TRAIN_SET_SIZE)
        train_choice = index_random[idx_start:idx_end]

        idx_start = int(pt_norm*config.TRAIN_SET_SIZE)
        idx_end = int(pt_norm*(config.TRAIN_SET_SIZE + config.DEV_SET_SIZE))
        dev_choice = index_random[idx_start:idx_end]

        train_set = []
        dev_set = []
        test_set = []
        for i in range(pt_norm):
            if i in train_choice:
                train_set.append(norm_set[i, :])
            elif i in dev_choice:
                dev_set.append(norm_set[i, :])
            else:
                test_set.append(norm_set[i, :])

        train_set = np.array(train_set)
        dev_set = np.array(dev_set)
        test_set = np.array(test_set)
        cvr_set = np.array(cvr_set)

        TRAIN_SET_PTH = '../data/train_set_' + config.FEEDER + '_' + config.GROUP + config.USERS + config.RESO + '_step' + str(
            config.STEP) + '.npy'
        TEST_SET_PTH = '../data/test_set_' + config.FEEDER + '_' + config.GROUP + config.USERS + config.RESO + '_step' + str(
            config.STEP) + '.npy'
        DEV_SET_PTH = '../data/dev_set_' + config.FEEDER + '_' + config.GROUP + config.USERS + config.RESO + '_step' + str(
            config.STEP) + '.npy'
        PEAK_LOAD_PTH = '../data/max_load_' + config.FEEDER + '_' + config.GROUP + config.USERS + config.RESO + '_step' + str(
            config.STEP) + '.npy'
        CVR_SET_PTH = '../data/cvr_set_' + config.FEEDER + '_' + config.GROUP + config.USERS + config.RESO + '_step' + str(
            config.STEP) + '.npy'

        np.save(TRAIN_SET_PTH, train_set)
        np.save(TEST_SET_PTH, test_set)
        np.save(DEV_SET_PTH, dev_set)
        np.save(PEAK_LOAD_PTH, max_load)
        np.save(CVR_SET_PTH, cvr_set)

    elif config.FEEDER == 'Pecan2015_IEEE123':
        train_set = np.genfromtxt('../data/raw_data/Pecan/train_set_Pecan2015_IEEE123.csv', delimiter=',')
        dev_set = np.genfromtxt('../data/raw_data/Pecan/dev_set_Pecan2015_IEEE123.csv', delimiter=',')
        test_set = np.genfromtxt('../data/raw_data/Pecan/test_set_Pecan2015_IEEE123.csv', delimiter=',')
        datase_cvr = np.genfromtxt('../data/raw_data/Pecan/dr_set_Pecan20150817_IEEE123.csv', delimiter=',')
        max_load = np.genfromtxt('../data/raw_data/Pecan/max_load_Pecan2015_IEEE123.csv', delimiter=',')
        
        np.save('../data/train_set_Pecan2015_IEEE123', train_set)
        np.save('../data/dev_set_Pecan2015_IEEE123', dev_set)
        np.save('../data/test_set_Pecan2015_IEEE123', test_set)
        np.save('../data/cvr_set_Pecan2015_IEEE123', datase_cvr)
        np.save('../data/max_load_Pecan2015_IEEE123', max_load)
        
    else:
        step = config.STEP
        dim_daily = config.DIM_INPUT # daily data dimension
        reso_min = 1440/dim_daily
        
        # read temperature data
        temp = pd.read_csv("../data/raw_data/temp_Boone.csv",index_col='Date') 
        temp_2019_2020 = temp.Temp['2019-01-01 00:00':'2020-12-31 00:00'].values
        temperature_norm = NormalizeData(temp_2019_2020)
        # CVR event information
        cvr_info = pd.read_csv('../data/raw_data/cvr_info.csv',index_col='event')
        
        np.random.seed(0) # seed everything for reproducible purpose
        
        # # generate all feeder together
        DATA_PTH = '../data/raw_data/2019_2020_NewRiver'
        LOAD_SIZE = ['large','medium','small']
        # for load_size in LOAD_SIZE:
        #     print('Processing ' + load_size + ' users for all feeders')
        #     file_pth = [DATA_PTH + '/BlowingRock_power_' + load_size + '_2019.csv',
        #                 DATA_PTH + '/BlowingRock_power_' + load_size + '_2020.csv',
        #                 DATA_PTH + '/Deerfield_power_' + load_size + '_2019.csv',
        #                 DATA_PTH + '/Deerfield_power_' + load_size + '_2020.csv',
        #                 DATA_PTH + '/Shadowline_power_' + load_size + '_2019.csv',
        #                 DATA_PTH + '/Shadowline_power_' + load_size + '_2020.csv']
        #     train_set, dev_set, test_set, cvr_set, max_load = \
        #         gen_new_river_dataset(file_pth, dim_daily, step, temperature_norm, cvr_info)
        #     np.save('../data/train_set_NewRiver_all_'+ load_size +'_step'+str(step)+'.npy', train_set)
        #     np.save('../data/dev_set_NewRiver_all_'+ load_size +'_step'+str(step)+'.npy', dev_set)
        #     np.save('../data/test_set_NewRiver_all_'+ load_size +'_step'+str(step)+'.npy', test_set)
        #     np.save('../data/cvr_set_NewRiver_all_'+ load_size +'_step'+str(step)+'.npy', cvr_set)
        #     np.save('../data/max_load_NewRiver_all_'+ load_size +'_step'+str(step)+'.npy', max_load)
        
        # generate each feeder seperately
        FEEDER_NAME = ['BlowingRock','Deerfield','Shadowline']
        LOAD_SIZE = ['large','medium','small']
        for feeder_name in FEEDER_NAME:
            for load_size in LOAD_SIZE:
                print('Processing ' + load_size + ' users for feeders:' + feeder_name)
                file_pth = [DATA_PTH + '/' + feeder_name +'_power_' + load_size + '_2019.csv',
                            DATA_PTH + '/' + feeder_name +'_power_' + load_size + '_2020.csv']
                train_set, dev_set, test_set, cvr_set, norm_near_cvr_set, max_load = \
                    gen_new_river_dataset(file_pth, dim_daily, step, temperature_norm, cvr_info)
                np.save('../data/train_set_NewRiver_'+ feeder_name+ '_' + load_size +'_step'+str(step)+'.npy', train_set)
                np.save('../data/dev_set_NewRiver_'+ feeder_name+ '_' + load_size +'_step'+str(step)+'.npy', dev_set)
                np.save('../data/test_set_NewRiver_'+ feeder_name+ '_' + load_size +'_step'+str(step)+'.npy', test_set)
                np.save('../data/cvr_set_NewRiver_'+ feeder_name+ '_' + load_size +'_step'+str(step)+'.npy', cvr_set)
                np.save('../data/norm_near_cvr_set_NewRiver_'+ feeder_name+ '_' + load_size +'_step'+str(step)+'.npy', norm_near_cvr_set)
                np.save('../data/max_load_NewRiver_'+ feeder_name+ '_' + load_size +'_step'+str(step)+'.npy', max_load)
                
        print("Dataset processing finished!")
    
    
class Profile_Dataset(Dataset):
    def __init__(self, path, dim_input):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataset = np.load(path)

        self.dim_input = dim_input
        self.dataset = torch.from_numpy(self.dataset).float().to(self.device)
        
    def __getitem__(self, index):
        # load_patched, mask, T, load_gt
        model_input = self.dataset[index, 0:self.dim_input*1].reshape([1, self.dim_input])
        temperature = self.dataset[index, self.dim_input*2:self.dim_input*3].reshape([1, self.dim_input])
        load_gt = self.dataset[index, self.dim_input*3:].reshape([1, self.dim_input])
        mask = self.dataset[index, self.dim_input*1:self.dim_input*2].bool().reshape([1, self.dim_input])
        
        # model_input = torch.from_numpy(model_input).float().to(self.device)
        # mask = torch.from_numpy(mask).bool().to(self.device)
        # load_gt = torch.from_numpy(load_gt).float().to(self.device)

        return model_input, temperature, mask, load_gt

    def __len__(self):
        return len(self.dataset)

class IEEE123_Dataset(Dataset):
    def __init__(self, path, dim_input):
        super().__init__()
        self.dataset = np.load(path)
        self.dim_input = dim_input
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, index):
        
        model_input = self.dataset[index, 0:self.dim_input*3].reshape(3, self.dim_input)
        load_dr = self.dataset[index, self.dim_input*3:self.dim_input*4].reshape(1, self.dim_input)
        load_gt = self.dataset[index, self.dim_input*4:].reshape(1, self.dim_input)
        mask = self.dataset[index, self.dim_input*1:self.dim_input*2].reshape(1, self.dim_input)
        
        model_input = torch.from_numpy(model_input).float().to(self.device)
        mask = torch.from_numpy(mask).float().to(self.device)
        load_gt = torch.from_numpy(load_gt).float().to(self.device)
        load_dr = torch.from_numpy(load_dr).float().to(self.device)

        return model_input, mask, load_gt, load_dr

    def __len__(self):
        return len(self.dataset)

    
trainset = Profile_Dataset(path = config.TRAIN_SET_PTH, dim_input = config.DIM_INPUT)
trainloader = DataLoader(trainset, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=True)
trainloader_eval = DataLoader(trainset,batch_size=1, num_workers=0, shuffle=False)

devset = Profile_Dataset(path = config.DEV_SET_PTH, dim_input = config.DIM_INPUT)
devloader = DataLoader(devset,batch_size=config.BATCH_SIZE, num_workers=0, shuffle=True)
devloader_eval = DataLoader(devset,batch_size=1, num_workers=0, shuffle=False)

testset = Profile_Dataset(path = config.TEST_SET_PTH, dim_input = config.DIM_INPUT)
testloader = DataLoader(testset, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=True)



if config.FEEDER == 'Fayetteville':
    cvrset = Profile_Dataset(path = config.CVR_SET_PTH, dim_input = config.DIM_INPUT)
    cvrloader = DataLoader(cvrset, batch_size=1, num_workers=0, shuffle=False)
    norm_cvr_set = Profile_Dataset(path = config.NEAR_CVR_SET_PTH, dim_input = config.DIM_INPUT)
    norm_near_cvr_loader = DataLoader(norm_cvr_set, batch_size=1, num_workers=0, shuffle=False)
elif config.FEEDER == 'Pecan2015':
    cvrset = None
    cvrloader = None
    norm_near_cvr_loader = None
elif config.FEEDER == 'NewRiver':
    cvrset = Profile_Dataset(path=config.CVR_SET_PTH, dim_input=config.DIM_INPUT)
    cvrloader = DataLoader(cvrset, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=False)
    norm_near_cvr_loader = None
elif config.FEEDER == 'Pecan2015_IEEE123':
    cvrset = IEEE123_Dataset(path = config.CVR_SET_PTH, dim_input = config.DIM_INPUT)
    cvrloader = DataLoader(cvrset, batch_size=1, num_workers=0, shuffle=False)
    norm_near_cvr_loader = None
else:
    cvrset = Profile_Dataset(path = config.CVR_SET_PTH, dim_input = config.DIM_INPUT)
    cvrloader = DataLoader(cvrset, batch_size=1, num_workers=0, shuffle=False)
    norm_cvr_set = Profile_Dataset(path = config.NEAR_CVR_SET_PTH, dim_input = config.DIM_INPUT)
    norm_near_cvr_loader = DataLoader(norm_cvr_set, batch_size=1, num_workers=0, shuffle=False)



"""
dataset analysis
"""
# # the histogram of normal daily average temperature
# fig = plt.figure(1, figsize=(10, 7))
# plt.clf()
# gs = fig.add_gridspec(2, 1)

# ax = fig.add_subplot(gs[0, 0])
# ax.hist(T_avg_norm[:,2], 50, density=False, facecolor='g', alpha=0.75)
# plt.ylabel('Number of sample')
# plt.title('Normal day')
# plt.grid(True)

# # the histogram of cvr daily average temperature
# ax = fig.add_subplot(gs[1, 0])
# ax.hist(T_avg_cvr[:,2], 50, density=False, facecolor='g', alpha=0.75)

# plt.xlabel('Daily temperature')
# plt.ylabel('Number of sample')
# plt.title('CVR day')
# plt.grid(True)
# plt.show()
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# fn = config.FEEDER + '_dailyT_histg.svg'
# fig.savefig(fn)


# # the histogram of normal daily average temperature
# fig = plt.figure(1, figsize=(10, 7))
# plt.clf()
# gs = fig.add_gridspec(2, 1)

# ax = fig.add_subplot(gs[0, 0])
# ax.hist(T_avg_norm[:,2], 50, density=False, facecolor='g', alpha=0.75)
# plt.xlim(-0.02, 1.02)
# plt.ylabel('Number of sample')
# plt.title('Normal period')
# plt.grid(True)


# # the histogram of cvr daily average temperature
# ax = fig.add_subplot(gs[1, 0])
# ax.hist(T_avg_cvr[:,2], 50, density=False, facecolor='b', alpha=0.75)
# plt.xlim(-0.02, 1.02)
# plt.xlabel('Daily temperature')
# plt.ylabel('Number of sample')
# plt.title('CVR period')
# plt.grid(True)
# plt.show()
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# fn = config.FEEDER + '_eventT_histg.svg'
# fig.savefig(fn)