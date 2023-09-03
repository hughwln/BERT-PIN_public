#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  26 2023

@author: Yi Hu
"""
import torch
import os

TRAIN_SET_SIZE = 0.7 # slice percentage of training set
DEV_SET_SIZE = 0.15
TEST_SET_SIZE = 0.15
GEN_DATASET = True
EVAL_MODE = False
EVAL_TEST_ONLY = False
TRAIN_BENCHMARK = False
USE_DELTA_P = False # predict deltaP and reconstruct P manually
USE_GLOBAL_MSE_LOSS = False # calculate MSE using global loss
USE_CENTRAL_MASK = True # generate mask at the middle of daily profile
USE_LOCAL_GAN_LOSS = True # 5 hours most 


FEEDER = 'NewRiver' # ['BlowingRock','Deerfield','Shadowline','all','Fayetteville', 'Pecan2015']
# -------Newriver dataset config
LOAD_SIZE = 'all' # ['large','medium','small','all']
GROUP = 'group5_' # [1,2,3,4,5]
USERS = '300users_' #[10,50,100,200,300]
DATE = '0902_multipatch_test_central'

RESO = '15min' #[1h,30min,15min,5min,1min]
NUM_H = 4
# ---------------------------------
STEP = NUM_H * 24 * 7 # move forward interval each slice, this step should be round-hour
DIM_INPUT = 24 * NUM_H * 7
DIM_PATCH = 4 * NUM_H
LOCAL_GAN_STEPS = int(DIM_INPUT/24*5) # 5 hours Local GAN loss window


N_SAMPLE = 'all' # [1000, 2000, 3000, ... 'all']
N_START = 0 # random patch start index
CH_INPUT = 3 # 0-load(with hole)/1-mask/2-temperature
N_EPOCH = 200 + 1
SAVE_PER_EPO = 10
BATCH_SIZE = 16
LR = 3e-4
NF_GEN = 64
NF_DIS = 8
W_P2P = 0.5
W_GAN = 0.02
W_FEA = 0.2
DROPRATE = 0.0


if FEEDER == 'Pecan2015':
    TAG = DATE+'_'+FEEDER+'_'+GROUP+USERS+RESO+'_step'+str(STEP)
elif FEEDER == 'Pecan2015_IEEE123':
    TAG = DATE+'_Pecan2015_IEEE123'
else:
    TAG = DATE+'_'+FEEDER+'_'+LOAD_SIZE+'_step'+str(STEP)
COMMENTS = 'Coarse L1loss, , GIN1D, fixed mask'

CUDA = True if torch.cuda.is_available() else False
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

DIR = "../plot/" + TAG
if not os.path.isdir(DIR): os.makedirs(DIR)
DIR = "../checkpoint/" + TAG
if not os.path.isdir(DIR): os.makedirs(DIR)
DIR = "../eval/" + TAG + '/examples'
if not os.path.isdir(DIR): os.makedirs(DIR)
DIR = "../eval/" + TAG + '/violin'
if not os.path.isdir(DIR): os.makedirs(DIR)
DIR = "../eval/" + TAG + '/src'
if not os.path.isdir(DIR): os.makedirs(DIR)
DIR = "../eval/" + TAG + '/metrics'
if not os.path.isdir(DIR): os.makedirs(DIR)

if FEEDER == 'Fayetteville':
    TRAIN_SET_PTH = '../data/train_set_'+FEEDER+'_'+LOAD_SIZE+'_step'+ str(STEP) + '.npy'
    TEST_SET_PTH  = '../data/test_set_'+FEEDER+'_'+LOAD_SIZE+'_step'+ str(STEP) + '.npy'
    DEV_SET_PTH   = '../data/dev_set_'+FEEDER+'_'+LOAD_SIZE+'_step'+ str(STEP) + '.npy'
    CVR_SET_PTH   = '../data/cvr_set_'+FEEDER+'_'+LOAD_SIZE+'_step'+ str(STEP) + '.npy'
    PEAK_LOAD_PTH = '../data/max_load_'+FEEDER+'_'+LOAD_SIZE+'_step'+ str(STEP) + '.npy'
    NEAR_CVR_SET_PTH = '../data/norm_near_cvr_set_Fayetteville_all' +'_step'+str(STEP)+'.npy'
    
elif FEEDER == 'Pecan2015':
    TRAIN_SET_PTH = '../data/train_set_'+FEEDER+'_'+GROUP+USERS+RESO+'_step'+ str(STEP) + '.npy'
    TEST_SET_PTH  = '../data/test_set_'+FEEDER+'_'+GROUP+USERS+RESO+'_step'+str(STEP) + '.npy'
    DEV_SET_PTH   = '../data/dev_set_'+FEEDER+'_'+GROUP+USERS+RESO+'_step'+ str(STEP) + '.npy'
    PEAK_LOAD_PTH = '../data/max_load_'+FEEDER+'_'+GROUP+USERS+RESO+'_step'+ str(STEP) + '.npy'

elif FEEDER == 'NewRiver':
    TRAIN_SET_PTH = '../data/train_set_'+FEEDER+'_'+GROUP+USERS+RESO+'_step'+ str(STEP) + '.npy'
    TEST_SET_PTH  = '../data/test_set_'+FEEDER+'_'+GROUP+USERS+RESO+'_step'+str(STEP) + '.npy'
    DEV_SET_PTH   = '../data/dev_set_'+FEEDER+'_'+GROUP+USERS+RESO+'_step'+ str(STEP) + '.npy'
    PEAK_LOAD_PTH = '../data/max_load_'+FEEDER+'_'+GROUP+USERS+RESO+'_step'+ str(STEP) + '.npy'
    CVR_SET_PTH = '../data/cvr_set_' + FEEDER + '_' + GROUP + USERS + RESO + '_step' + str(STEP) + '.npy'

elif FEEDER == 'Pecan2015_IEEE123':
    TRAIN_SET_PTH = '../data/train_set_Pecan2015_IEEE123.npy'
    TEST_SET_PTH  = '../data/test_set_Pecan2015_IEEE123.npy'
    DEV_SET_PTH   = '../data/dev_set_Pecan2015_IEEE123.npy'
    CVR_SET_PTH   = '../data/cvr_set_Pecan2015_IEEE123.npy'
    PEAK_LOAD_PTH = '../data/max_load_Pecan2015_IEEE123.npy'
    
else:
    TRAIN_SET_PTH = '../data/train_set_NewRiver_'+FEEDER+'_'+LOAD_SIZE+'_step'+ str(STEP) + '.npy'
    TEST_SET_PTH  = '../data/test_set_NewRiver_'+FEEDER+'_'+LOAD_SIZE+'_step'+ str(STEP) + '.npy'
    DEV_SET_PTH   = '../data/dev_set_NewRiver_'+FEEDER+'_'+LOAD_SIZE+'_step'+ str(STEP) + '.npy'
    CVR_SET_PTH   = '../data/cvr_set_NewRiver_'+FEEDER+'_'+LOAD_SIZE+'_step'+ str(STEP) + '.npy'
    NEAR_CVR_SET_PTH = '../data/norm_near_cvr_set_NewRiver_'+FEEDER+'_'+LOAD_SIZE+'_step'+ str(STEP) + '.npy'
    PEAK_LOAD_PTH = '../data/max_load_NewRiver_'+FEEDER+'_'+LOAD_SIZE+'_step'+ str(STEP) + '.npy'

