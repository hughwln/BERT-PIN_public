#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 4 2023

@author: Yi Hu
"""
import time
import torch
import torch.nn as nn
import config
import numpy as np
import model
from config import *

from dataset import testloader, trainloader, devloader, cvrloader
import evaluation
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.fftpack import fft
from torch.autograd import Variable
import torch.nn.functional as F


class SNDisLoss(torch.nn.Module):
    """
    The loss for sngan discriminator
    """
    def __init__(self, weight=1):
        super(SNDisLoss, self).__init__()
        self.weight = weight

    def forward(self, pos, neg):
        #return self.weight * (torch.sum(F.relu(-1+pos)) + torch.sum(F.relu(-1-neg)))/pos.size(0)
        return self.weight * (torch.mean(F.relu(1.-pos)) + torch.mean(F.relu(1.+neg)))


class SNGenLoss(torch.nn.Module):
    """
    The loss for sngan generator
    """
    def __init__(self, weight=1):
        super(SNGenLoss, self).__init__()
        self.weight = weight

    def forward(self, neg):
        return - self.weight * torch.mean(neg)


def calc_loss(pre, gt, mask, criterion, K):
    dim_day = 24 * config.NUM_H
    n_day = int(config.DIM_INPUT / dim_day)
    bs = gt.size(0)

    local_entropy_loss = 0
    global_entropy_loss = 0
    for n in range(n_day):
        pre_day = pre[:, n * dim_day: (n + 1) * dim_day]
        gt_day = gt[:, n * dim_day: (n + 1) * dim_day]
        mask_np = mask[:, n * dim_day: (n+1) * dim_day].cpu().detach().numpy()
        for j in range(bs):
            if mask is not None:
                patch = np.where(mask_np[j, :] == 1.0)
                if len(patch[0]) == 0:
                    continue
                patch_bgn = patch[0][0] - 1
                patch_end = patch[0][-1] + 1

            loss_1 = criterion(pre_day[j, patch_bgn:patch_end+1], gt_day[j, patch_bgn:patch_end+1])
            local_entropy_loss += loss_1

            loss_1 = criterion(pre_day[j], gt_day[j])
            global_entropy_loss += loss_1

    local_entropy_loss = local_entropy_loss / bs
    global_entropy_loss = global_entropy_loss / bs

    loss = global_entropy_loss + 5 * local_entropy_loss
    return loss

def calc_benchmark_loss(pre, gt, mask):
    mask_np = mask.cpu().detach().numpy()
    mse = nn.MSELoss(reduction='sum')
    global_mse_loss = mse(pre, gt)
    # if mask is not None:
    #     # pre = pre.masked_fill(mask == False, float(0.0))
    #     # gt = gt.masked_fill(mask == False, float(0.0))
    #     patch_bgn = np.where(mask_np[0, :] == 1.0)[0][0] - 1
    #     patch_end = np.where(mask_np[0, :] == 1.0)[0][-1] + 1
    #
    # patch_pre = pre[:, patch_bgn:patch_end+1]
    # patch_gt = gt[:, patch_bgn:patch_end+1]
    # mse_loss = mse(patch_pre, patch_gt)

    est = gt * ~mask + pre * mask
    mse_loss = mse(gt, est)

    # loss = 5 * mse_loss ** 0.5
    return global_mse_loss + 5 * mse_loss

def train_bert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    K = 202
    heads = 2
    transformer = model.Transformer(src_vocab_size=K+heads, embed_size=K+heads, heads=heads, num_layers=2,
                                    forward_expansion=heads, max_length=config.DIM_INPUT, device=device).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config.LR, betas=(0.9, 0.99))

    criterion = torch.nn.CrossEntropyLoss()

    # ------------------------------------Training------------------------------------
    start_t = time.strftime("%Y/%m/%d %H:%M:%S")
    loss_train_rec = []
    loss_eval_rec = []
    err_rec = []

    print("start train BERT.")
    for epoch in range(config.N_EPOCH):
        train_loss_list = []
        for i, data in enumerate(trainloader):
            transformer.train()
            model_input, temperature, mask, gt = data
            model_input = model_input.reshape((model_input.shape[0], model_input.shape[2])).to(device)
            mask = mask.reshape((mask.shape[0], mask.shape[2])).to(device)
            gt = gt.reshape((gt.shape[0], gt.shape[2])).to(device)
            temperature = temperature.reshape((temperature.shape[0], temperature.shape[2])).to(device)

            model_input = torch.round(model_input * K).type(torch.int)
            temperature = torch.round(temperature * K).type(torch.int)
            gt = torch.round(gt * K).type(torch.int)
            gt = gt.reshape((gt.shape[0], gt.shape[1]), -1).type(torch.LongTensor).to(device)

            bs = gt.size(0)

            output = transformer(model_input, temperature, mask)

            loss = calc_loss(output, gt, mask, criterion, K)
            train_loss_list.append(loss.item())

            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            # grad_norm = nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # t = time.strftime("%Y/%m/%d %H:%M:%S")
            # print("epoch ", epoch, " step ", i, "/", len(trainloader), " ====== train loss ", loss.item())

        if epoch % 10 == 0:
            torch.save(transformer.state_dict(),
                       '../checkpoint/' + config.TAG + '/encoder_' + str(epoch) + '.pth')

        train_loss = np.mean(train_loss_list)
        '[ave_mpe, ave_rmse, ave_pk_err, ave_vl_err, ave_egy_err, ave_fce]'
        eval_loss, errs = evaluation.eval_set(transformer, K, criterion, device)
        loss_train_rec.append(train_loss)
        loss_eval_rec.append(eval_loss)
        err_rec.append(errs)
        t = time.strftime("%Y/%m/%d %H:%M:%S")
        print("epoch ", epoch, " ====== train loss ", train_loss, " eval loss ", eval_loss, t)
        print("         errs: ", errs)
        evaluation.plot_samples(transformer, epoch, K, device)
        evaluation.plot_loss(loss_train_rec, loss_eval_rec, epoch, 'BERT')
        evaluation.plot_err(err_rec, epoch, 'BERT')
        epoch += 1

    TRAINLOSS_PTH = '../eval/'+ config.TAG + '/BERT_train_loss.npy'
    np.save(TRAINLOSS_PTH, loss_train_rec)
    EVALLOSS_PTH = '../eval/' + config.TAG + '/BERT_eval_loss.npy'
    np.save(EVALLOSS_PTH, loss_eval_rec)
    EVALERR_PTH = '../eval/' + config.TAG + '/BERT_eval_err.npy'
    np.save(EVALERR_PTH, err_rec)

def train_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    sae = model.SAE(name='SAE').to(device)
    lstm = model.LSTM(in_channels=CH_INPUT, name='LSTM').to(device)

    opt_sae = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.99))
    opt_lstm = torch.optim.Adam(lstm.parameters(), lr=LR, betas=(0.9, 0.99))

    models = [sae, lstm]
    opts = [opt_sae, opt_lstm]

    mse_loss = torch.nn.MSELoss()

    # ------------------------------------Training------------------------------------
    start_t = time.strftime("%Y/%m/%d %H:%M:%S")
    loss_train_rec = [[] for n in range(len(models))]
    loss_eval_rec = [[] for n in range(len(models))]
    err_rec = [[] for n in range(len(models))]

    for epoch in range(config.N_EPOCH):
        train_loss_list = [[] for n in range(len(models))]
        for i, data in enumerate(trainloader):
            for j in range(len(models)):
                models[j].train()
                model_input, temperature, mask, gt = data
                bs = gt.size(0)
                model_input = model_input.reshape((model_input.shape[0], model_input.shape[2])).to(device)
                mask = mask.reshape((mask.shape[0], mask.shape[2])).to(device)
                temperature = temperature.reshape((temperature.shape[0], temperature.shape[2])).to(device)
                gt = gt.reshape((gt.shape[0], gt.shape[2])).to(device)

                opts[j].zero_grad()
                if models[j].name == 'SAE':
                    mask_np = mask.cpu().detach().numpy()
                    for k in range(bs):
                        load_np = model_input[k, :].cpu().detach().numpy()
                        patch_bgn = np.where(mask_np[k, :] == 1.0)[0][0]
                        patch_end = np.where(mask_np[k, :] == 1.0)[0][-1]
                        m_ = 0.5 * (load_np[patch_bgn - 1] + load_np[patch_end + 1])
                        model_input_sae = model_input
                        model_input_sae[k, patch_bgn:patch_end + 1] = \
                            m_ * torch.ones_like(model_input[k, patch_bgn:patch_end + 1])
                    pre = models[j](model_input_sae[:, :])
                    pre = model_input_sae[:, :] + pre * mask
                else:
                    pre = models[j](model_input, mask, temperature)
                    pre = model_input[:, :] + pre * mask

                loss_train = calc_benchmark_loss(pre, gt, mask)
                train_loss_list[j].append(loss_train.item())
                loss_train.backward()
                opts[j].step()

        for j in range(len(models)):
            loss_mean = np.mean(train_loss_list[j])
            loss_train_rec[j].append(loss_mean)
            models[j].save_checkpoint(epoch)

            '[ave_mpe, ave_rmse, ave_pk_err, ave_vl_err, ave_egy_err, ave_fce]'
            eval_loss, errs = evaluation.eval_set(models[j], 1, mse_loss, device)
            loss_eval_rec[j].append(eval_loss)
            err_rec[j].append(errs)
            t = time.strftime("%Y/%m/%d %H:%M:%S")
            print("epoch ", epoch, " ====== ", models[j].name, "train loss ", loss_mean, " eval loss ", eval_loss, t)
            evaluation.plot_samples(models[j], epoch, 1, device)
            evaluation.plot_loss(loss_train_rec[j], loss_eval_rec[j], epoch, models[j].name)
            evaluation.plot_err(err_rec[j], epoch, models[j].name)
        epoch += 1

    for j in range(len(models)):
        TRAINLOSS_PTH = '../eval/' + config.TAG + '/' + models[j].name + '_train_loss.npy'
        np.save(TRAINLOSS_PTH, loss_train_rec[j])
        EVALLOSS_PTH = '../eval/' + config.TAG + '/' + models[j].name + '_eval_loss.npy'
        np.save(EVALLOSS_PTH, loss_eval_rec[j])
        EVALERR_PTH = '../eval/' + config.TAG + '/' + models[j].name + '_eval_err.npy'
        np.save(EVALERR_PTH, err_rec[j])

def train_sae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    sae = model.SAE(name='SAE').to(device)

    opt_sae = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.99))

    mse_loss = torch.nn.MSELoss()

    # ------------------------------------Training------------------------------------
    start_t = time.strftime("%Y/%m/%d %H:%M:%S")
    loss_train_rec = []
    loss_eval_rec = []
    err_rec = []

    for epoch in range(config.N_EPOCH):
        train_loss_list = []
        for i, data in enumerate(trainloader):
            sae.train()
            model_input, temperature, mask, gt = data
            bs = gt.size(0)
            model_input = model_input.reshape((model_input.shape[0], model_input.shape[2])).to(device)
            mask = mask.reshape((mask.shape[0], mask.shape[2])).to(device)
            temperature = temperature.reshape((temperature.shape[0], temperature.shape[2])).to(device)
            gt = gt.reshape((gt.shape[0], gt.shape[2])).to(device)

            opt_sae.zero_grad()

            # mask_np = mask.cpu().detach().numpy()

            model_input_sae = model_input
            dim_day = 24 * config.NUM_H
            n_day = int(config.DIM_INPUT / dim_day)
            for n in range(n_day):
                input_day = model_input[:, n * dim_day: (n + 1) * dim_day]
                mask_np = mask[:, n * dim_day: (n + 1) * dim_day].cpu().detach().numpy()

                for k in range(bs):
                    patch = np.where(mask_np[k, :] == 1.0)
                    if len(patch[0]) == 0:
                        continue
                    patch_bgn = patch[0][0]
                    patch_end = patch[0][-1]
                    load_np = input_day[k, :].cpu().detach().numpy()
                    m_ = 0.5 * (load_np[patch_bgn - 1] + load_np[patch_end + 1])
                    model_input_sae[k, patch_bgn + n * dim_day: n * dim_day + patch_end + 1] = \
                        m_ * torch.ones_like(model_input[k, patch_bgn + n * dim_day: n * dim_day + patch_end + 1])
            pre = sae(model_input_sae[:, :])
            pre = model_input_sae[:, :] + pre * mask

            loss_train = calc_benchmark_loss(pre, gt, mask)
            train_loss_list.append(loss_train.item())
            loss_train.backward()
            opt_sae.step()

        loss_mean = np.mean(train_loss_list)
        loss_train_rec.append(loss_mean)
        sae.save_checkpoint(epoch)

        '[ave_mpe, ave_rmse, ave_pk_err, ave_vl_err, ave_egy_err, ave_fce]'
        eval_loss, errs = evaluation.eval_set(sae, 1, mse_loss, device)
        loss_eval_rec.append(eval_loss)
        err_rec.append(errs)
        t = time.strftime("%Y/%m/%d %H:%M:%S")
        print("epoch ", epoch, " ====== ", sae.name, "train loss ", loss_mean, " eval loss ", eval_loss, t)
        evaluation.plot_samples(sae, epoch, 1, device)
        evaluation.plot_loss(loss_train_rec, loss_eval_rec, epoch, sae.name)
        evaluation.plot_err(err_rec, epoch, sae.name)

        epoch += 1

    TRAINLOSS_PTH = '../eval/' + config.TAG + '/' + sae.name + '_train_loss.npy'
    np.save(TRAINLOSS_PTH, loss_train_rec)
    EVALLOSS_PTH = '../eval/' + config.TAG + '/' + sae.name + '_eval_loss.npy'
    np.save(EVALLOSS_PTH, loss_eval_rec)
    EVALERR_PTH = '../eval/' + config.TAG + '/' + sae.name + '_eval_err.npy'
    np.save(EVALERR_PTH, err_rec)

def train_lstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    lstm = model.LSTM(name='LSTM').to(device)

    opt_lstm = torch.optim.Adam(lstm.parameters(), lr=LR, betas=(0.9, 0.99))

    mse_loss = torch.nn.MSELoss()

    # ------------------------------------Training------------------------------------
    start_t = time.strftime("%Y/%m/%d %H:%M:%S")
    loss_train_rec = []
    loss_eval_rec = []
    err_rec = []

    for epoch in range(11):
        train_loss_list = []
        for i, data in enumerate(trainloader):
            lstm.train()
            model_input, temperature, mask, gt = data
            bs = gt.size(0)
            model_input = model_input.reshape((model_input.shape[0], model_input.shape[2])).to(device)
            mask = mask.reshape((mask.shape[0], mask.shape[2])).to(device)
            temperature = temperature.reshape((temperature.shape[0], temperature.shape[2])).to(device)
            gt = gt.reshape((gt.shape[0], gt.shape[2])).to(device)

            opt_lstm.zero_grad()

            pre = lstm(model_input, mask, temperature)
            pre = model_input[:, :] + pre * mask

            loss_train = calc_benchmark_loss(pre, gt, mask)
            train_loss_list.append(loss_train.item())
            loss_train.backward()
            opt_lstm.step()

        loss_mean = np.mean(train_loss_list)
        loss_train_rec.append(loss_mean)
        lstm.save_checkpoint(epoch)

        '[ave_mpe, ave_rmse, ave_pk_err, ave_vl_err, ave_egy_err, ave_fce]'
        eval_loss, errs = evaluation.eval_set(lstm, 1, mse_loss, device)
        loss_eval_rec.append(eval_loss)
        err_rec.append(errs)
        t = time.strftime("%Y/%m/%d %H:%M:%S")
        print("epoch ", epoch, " ====== ", lstm.name, "train loss ", loss_mean, " eval loss ", eval_loss, t)
        evaluation.plot_samples(lstm, epoch, 1, device)
        evaluation.plot_loss(loss_train_rec, loss_eval_rec, epoch, lstm.name)
        evaluation.plot_err(err_rec, epoch, lstm.name)

        epoch += 1

    TRAINLOSS_PTH = '../eval/' + config.TAG + '/' + lstm.name + '_train_loss.npy'
    np.save(TRAINLOSS_PTH, loss_train_rec)
    EVALLOSS_PTH = '../eval/' + config.TAG + '/' + lstm.name + '_eval_loss.npy'
    np.save(EVALLOSS_PTH, loss_eval_rec)
    EVALERR_PTH = '../eval/' + config.TAG + '/' + lstm.name + '_eval_err.npy'
    np.save(EVALERR_PTH, err_rec)

def train_gin():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gin = model.GenGIN(in_ch=CH_INPUT, n_fea=NF_GEN, name='GIN')
    disc = model.DisGIN(n_fea=config.NF_DIS)
    opt_gin = torch.optim.Adam(gin.parameters(), lr=LR, betas=(0.9, 0.99))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=config.LR, betas=(0.9, 0.99))

    mse_loss = torch.nn.MSELoss()

    l1_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()
    gan_loss = SNGenLoss()
    dis_loss = SNDisLoss()

    if config.CUDA:
        gin.cuda()
        disc.cuda()

    loss_train_rec = []
    loss_eval_rec = []
    err_rec = []

    # ------------------------------------Training------------------------------------
    start_t = time.time()
    for epoch in range(config.N_EPOCH):
        for i, data in enumerate(trainloader):
            model_input, temperature, mask, gt = data
            bs = gt.size(0)

            # train discriminator
            opt_disc.zero_grad()

            model_input = torch.cat([model_input, mask, temperature], dim=1)
            pre, pre_coarse, _ = gin(model_input)

            if not config.USE_GLOBAL_MSE_LOSS:
                pre = model_input[:, 0, :].unsqueeze(1) + pre * mask
                pre_coarse = model_input[:, 0, :].unsqueeze(1) + pre_coarse * mask

            if config.USE_LOCAL_GAN_LOSS:
                idx_st = int(config.DIM_INPUT // 2 - config.NUM_H * 2.5)
                idx_end = idx_st + int(config.NUM_H * 5)
                pos = torch.cat([gt[:, :, idx_st:idx_end],
                                 mask[:, :, idx_st:idx_end],
                                 torch.full_like(mask[:, :, idx_st:idx_end], 1.)], dim=1)
                neg = torch.cat([pre[:, :, idx_st:idx_end],
                                 mask[:, :, idx_st:idx_end],
                                 torch.full_like(mask[:, :, idx_st:idx_end], 1.)], dim=1)
                pos_neg = torch.cat([pos, neg], dim=0)
            else:
                pos = torch.cat([gt, mask, torch.full_like(mask, 1.)], dim=1)
                neg = torch.cat([pre, mask, torch.full_like(mask, 1.)], dim=1)
                pos_neg = torch.cat([pos, neg], dim=0)

            pred_pos_neg, _ = disc(pos_neg)
            pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
            d_loss = dis_loss(pred_pos, pred_neg)
            d_loss.backward(retain_graph=True)
            opt_disc.step()

            # train generator
            opt_disc.zero_grad()
            disc.zero_grad()
            opt_gin.zero_grad()
            gin.zero_grad()

            pred_neg, fea_neg = disc(neg)  # Fake samples
            _, fea_pos = disc(pos)  # Fake samples

            g_loss = gan_loss(pred_neg)
            fea_loss = mse_loss(fea_pos, fea_neg)
            r_loss = l1_loss(gt, pre)
            r_loss_coarse = l1_loss(gt, pre_coarse)
            loss_train = r_loss_coarse + config.W_GAN * g_loss + \
                         config.W_P2P * r_loss + config.W_FEA * fea_loss
            # loss_train = config.W_GAN * g_loss + config.W_P2P * r_loss + config.W_FEA * fea_loss

            loss_train.backward()
            opt_gin.step()

            loss_train_rec.append(loss_train.cpu().detach().numpy())
            # print("epoch ", epoch, " step ", i, "/", len(trainloader))

        gin.save_checkpoint(epoch)
        '[ave_mpe, ave_rmse, ave_pk_err, ave_vl_err, ave_egy_err, ave_fce]'
        eval_loss, errs = evaluation.eval_set(gin, 1, mse_loss, device)
        loss_eval_rec.append(eval_loss)
        err_rec.append(errs)
        t = time.strftime("%Y/%m/%d %H:%M:%S")
        print("epoch ", epoch, " ====== ", gin.name, "train loss ", np.mean(loss_train_rec), " eval loss ", eval_loss, t)
        evaluation.plot_samples(gin, epoch, 1, device)
        evaluation.plot_loss(loss_train_rec, loss_eval_rec, epoch, gin.name)
        evaluation.plot_err(err_rec, epoch, gin.name)

        epoch += 1

    TRAINLOSS_PTH = '../eval/' + config.TAG + '/' + gin.name + '_train_loss.npy'
    np.save(TRAINLOSS_PTH, loss_train_rec)
    EVALLOSS_PTH = '../eval/' + config.TAG + '/' + gin.name + '_eval_loss.npy'
    np.save(EVALLOSS_PTH, loss_eval_rec)
    EVALERR_PTH = '../eval/' + config.TAG + '/' + gin.name + '_eval_err.npy'
    np.save(EVALERR_PTH, err_rec)

def test_7days_week():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = 202
    heads = 2
    transformer = model.Transformer(src_vocab_size=K + heads, embed_size=K + heads, heads=heads, num_layers=2,
                                    forward_expansion=heads, max_length=DIM_INPUT, device=device).to(device)
    transformer_day = model.Transformer(src_vocab_size=K + heads, embed_size=K + heads, heads=heads, num_layers=2,
                                    forward_expansion=heads, max_length=24 * NUM_H, device=device).to(device)
    if config.USE_CENTRAL_MASK:
        transformer.load_state_dict(
            torch.load('../checkpoint/0615_BLIN_multipatch_central_NewRiver_all_step672/encoder_30.pth'))
        transformer_day.load_state_dict(
            torch.load('../checkpoint/0419_feeder2000_200_global_local_mse_mid_NewRiver_all_step96/encoder_100.pth'))
    else:
        transformer.load_state_dict(
            torch.load('../checkpoint/0615_BLIN_multipatch_peak_NewRiver_all_step672/encoder_50.pth'))
        transformer_day.load_state_dict(torch.load('../checkpoint/0417_feeder2000_200_global_local_peak_mse_NewRiver_all_step96/encoder_100.pth'))

    bert_mpe_buff = []
    bert_rmse_buff = []
    bert_pk_err_buff = []
    bert_vl_err_buff = []
    bert_egy_err_buff = []
    bert_fce_buff = []
    bert_mpe_buff_day = []
    bert_rmse_buff_day = []
    bert_pk_err_buff_day = []
    bert_vl_err_buff_day = []
    bert_egy_err_buff_day = []
    bert_fce_buff_day = []

    dim_day = 24 * NUM_H

    for i, data in enumerate(testloader):
        model_input, temperature, mask, gt = data
        model_input = model_input.reshape((model_input.shape[0], model_input.shape[2])).to(device)
        mask = mask.reshape((mask.shape[0], mask.shape[2])).to(device)
        gt = gt.reshape((gt.shape[0], gt.shape[2])).to(device)
        temperature = temperature.reshape((temperature.shape[0], temperature.shape[2])).to(device)
        bs = gt.size(0)
        mask_np = mask.cpu().detach().numpy()


        # BERT
        bert_input = model_input.clone()
        bert_input = torch.round(bert_input * K).type(torch.int)
        temperature = torch.round(temperature * K).type(torch.int)
        bert_output = transformer(bert_input, temperature, mask)
        bert_output = bert_output.argmax(dim=-1)
        bert_output = bert_output / K

        # BERT day
        bert_output_days = []
        for n in range(int(DIM_INPUT / dim_day)):
            day_output = transformer_day(bert_input[:, n * dim_day: (n + 1) * dim_day],
                                         temperature[:, n * dim_day: (n + 1) * dim_day],
                                         mask[:, n * dim_day: (n + 1) * dim_day])
            day_output = day_output.argmax(dim=-1)
            day_output = day_output / K
            day_output = model_input[:, n * dim_day: (n + 1) * dim_day] + day_output * mask[:, n * dim_day: (n + 1) * dim_day]
            bert_output_days.append(day_output)
        bert_output_days = torch.cat(bert_output_days, dim=1)

        pre_np_bert = bert_output.cpu().detach().numpy()
        pre_np_bert_day = bert_output_days.cpu().detach().numpy()
        input_np = model_input.cpu().detach().numpy()
        gt_np = gt.cpu().detach().numpy()

        fig = plt.figure(1, figsize=(3 * 3, 2 * 3))
        plt.clf()
        gs = fig.add_gridspec(3, 1)
        for j in range(3):
            ax = fig.add_subplot(gs[j, 0])
            buff = np.concatenate((pre_np_bert[j, :], gt_np[j, :]))

            y_min = np.amin(buff)
            y_max = np.amax(buff)

            for n in range(int(DIM_INPUT / dim_day)):
                patch = np.where(mask_np[j, n * dim_day: (n + 1) * dim_day] == 1.0)
                if len(patch[0]) != 0:
                    patch_bgn = patch[0][0] - 1 + n * dim_day
                    patch_end = patch[0][-1] + 1 + n * dim_day
                    x = np.arange(patch_bgn, patch_end + 1)
                    # draw output data of the test set
                    ax.plot(x, pre_np_bert[j, patch_bgn:patch_end + 1], 'r', linewidth=1, label='Week')
                    ax.plot(x, pre_np_bert_day[j, patch_bgn:patch_end + 1], 'b', linewidth=1, label='7days')
                    rect = plt.Rectangle((patch_bgn, y_min), patch_end - patch_bgn + 1, y_max - y_min,
                                         facecolor="k", alpha=0.1)
                    ax.add_patch(rect)
                # draw gt of the test set
                ax.plot(np.arange(n * dim_day, (n + 1) * dim_day), gt_np[j, n * dim_day: (n + 1) * dim_day], 'g',
                        linewidth=1, label='GT')
                plt.ylim(y_min, y_max)
                plt.xticks([])
                plt.yticks([])
            legend_without_duplicate_labels(ax)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.pause(0.001)  # pause a bit so that plots are updated

        fn = '../plot/' + TAG + '/samples_pre' + str(i) + '.png'
        fig.savefig(fn, dpi=300)

        # calculate errs
        for idx in range(bs):
            mpe, rmse, pk_err, vl_err, egy_err, fft_err = 0, 0, 0, 0, 0, 0
            mpe_day, rmse_day, pk_err_day, vl_err_day, egy_err_day, fft_err_day = 0, 0, 0, 0, 0, 0
            dim_day = 24 * NUM_H
            for n in range(int(DIM_INPUT / dim_day)):
                pre_day = pre_np_bert[idx, n * dim_day: (n + 1) * dim_day]
                pre_day_day = pre_np_bert_day[idx, n * dim_day: (n + 1) * dim_day]
                gt_day = gt_np[idx, n * dim_day: (n + 1) * dim_day]
                mask_day = mask_np[idx, n * dim_day: (n + 1) * dim_day]
                patch = np.where(mask_day == 1.0)
                if len(patch[0]) == 0:
                    continue
                patch_bgn = patch[0][0]
                patch_end = patch[0][-1] + 1
                rmse += mean_squared_error(pre_day[patch_bgn:patch_end], gt_day[patch_bgn:patch_end]) ** 0.5
                # percentage error: (gt - pre) / gt
                pe = abs(gt_day[patch_bgn:patch_end] - pre_day[patch_bgn:patch_end]) / gt_day[patch_bgn:patch_end]
                mpe += np.mean(pe)
                # energy error
                egy_err += abs(np.sum(pre_day[patch_bgn:patch_end]) - np.sum(gt_day[patch_bgn:patch_end])) / np.sum(
                    gt_day[patch_bgn:patch_end])
                # peak error, valley error, peak-valley error
                pk_err_, vl_err_, pkvy_err_ = evaluation.peak_valley_err(pre_day[patch_bgn:patch_end], gt_day[patch_bgn:patch_end])
                pk_err += pk_err_
                vl_err += vl_err_
                # frequency components error
                f_pre = abs(fft(pre_day[patch_bgn:patch_end]))
                f_gt = abs(fft(gt_day[patch_bgn:patch_end]))
                fft_err += np.sum(abs(f_gt - f_pre)) / (patch_end - patch_bgn + 1)

                rmse_day += mean_squared_error(pre_day_day[patch_bgn:patch_end], gt_day[patch_bgn:patch_end]) ** 0.5
                # percentage error: (gt - pre) / gt
                pe_day = abs(gt_day[patch_bgn:patch_end] - pre_day_day[patch_bgn:patch_end]) / gt_day[patch_bgn:patch_end]
                mpe_day += np.mean(pe_day)
                # energy error
                egy_err_day += abs(np.sum(pre_day_day[patch_bgn:patch_end]) - np.sum(gt_day[patch_bgn:patch_end])) / np.sum(
                    gt_day[patch_bgn:patch_end])
                # peak error, valley error, peak-valley error
                pk_err__day, vl_err__day, pkvy_err__day = evaluation.peak_valley_err(pre_day_day[patch_bgn:patch_end],
                                                                         gt_day[patch_bgn:patch_end])
                pk_err_day += pk_err__day
                vl_err_day += vl_err__day
                # frequency components error
                f_pre_day = abs(fft(pre_day_day[patch_bgn:patch_end]))
                fft_err_day += np.sum(abs(f_gt - f_pre_day)) / (patch_end - patch_bgn + 1)

            bert_mpe_buff.append(mpe)
            bert_rmse_buff.append(rmse)
            bert_pk_err_buff.append(pk_err)
            bert_vl_err_buff.append(vl_err)
            bert_egy_err_buff.append(egy_err)
            bert_fce_buff.append(fft_err)

            bert_mpe_buff_day.append(mpe_day)
            bert_rmse_buff_day.append(rmse_day)
            bert_pk_err_buff_day.append(pk_err_day)
            bert_vl_err_buff_day.append(vl_err_day)
            bert_egy_err_buff_day.append(egy_err_day)
            bert_fce_buff_day.append(fft_err_day)



        if i > 10:
            break

    figure = plt.figure(2)

    violin_parts = plt.violinplot([bert_mpe_buff,
                                   bert_rmse_buff,
                                   bert_pk_err_buff,
                                   bert_vl_err_buff,
                                   bert_egy_err_buff,
                                   bert_fce_buff], showmedians=True, showextrema=False)
    plt.xticks(range(1, 7), labels=['mpe',
                                     'rmse',
                                     'pk_err',
                                     'vl_err',
                                     'egy_err',
                                     'fce_err'])
    plt.show()
    fn = '../plot/' + TAG + '/violinplot.png'
    figure.savefig(fn, dpi=300)
    print("error week")
    print(np.mean(bert_mpe_buff), np.mean(bert_rmse_buff), np.mean(bert_pk_err_buff), np.mean(bert_vl_err_buff), np.mean(bert_egy_err_buff), np.mean(bert_fce_buff))
    print("error day")
    print(np.mean(bert_mpe_buff_day), np.mean(bert_rmse_buff_day), np.mean(bert_pk_err_buff_day), np.mean(bert_vl_err_buff_day),
          np.mean(bert_egy_err_buff_day), np.mean(bert_fce_buff_day))

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def test():
    bm = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = 202
    heads = 2
    transformer = model.Transformer(src_vocab_size=K + heads, embed_size=K + heads, heads=heads, num_layers=2,
                                    forward_expansion=heads, max_length=DIM_INPUT, device=device).to(device)

    if config.USE_CENTRAL_MASK:
        transformer.load_state_dict(
            torch.load('../checkpoint/0615_BLIN_multipatch_central_NewRiver_all_step672/encoder_30.pth'))
    else:
        transformer.load_state_dict(
            torch.load('../checkpoint/0615_BLIN_multipatch_peak_NewRiver_all_step672/encoder_50.pth'))
    bert_mpe_buff = []
    bert_rmse_buff = []
    bert_pk_err_buff = []
    bert_vl_err_buff = []
    bert_egy_err_buff = []
    bert_fce_buff = []

    if bm:
        sae = model.SAE(name='SAE').to(device)
        lstm = model.LSTM(in_channels=CH_INPUT, name='LSTM').to(device)
        gin = model.GenGIN(in_ch=CH_INPUT, n_fea=NF_GEN, name='GIN')

        if config.USE_CENTRAL_MASK:
            sae.load_state_dict(
                torch.load('../checkpoint/0615_SAE_multipatch_central_NewRiver_all_step672/SAE.pth'))
            lstm.load_state_dict(
                torch.load('../checkpoint/0615_LSTM_multipatch_central_NewRiver_all_step672/LSTM.pth'))
            gin.load_state_dict(torch.load('../checkpoint/0616_GIN_multipatch_central_NewRiver_all_step672/GIN.h5'))
        else:
            sae.load_state_dict(
                torch.load('../checkpoint/0615_SAE_multipatch_peak_NewRiver_all_step672/SAE.pth'))
            lstm.load_state_dict(
                torch.load('../checkpoint/0615_LSTM_multipatch_peak_NewRiver_all_step672/LSTM.pth'))
            gin.load_state_dict(torch.load('../checkpoint/0616_GIN_multipatch_peak_NewRiver_all_step672/GIN.h5'))

        sae_mpe_buff = []
        sae_rmse_buff = []
        sae_pk_err_buff = []
        sae_vl_err_buff = []
        sae_egy_err_buff = []
        sae_fce_buff = []
        lstm_mpe_buff = []
        lstm_rmse_buff = []
        lstm_pk_err_buff = []
        lstm_vl_err_buff = []
        lstm_egy_err_buff = []
        lstm_fce_buff = []
        gin_mpe_buff = []
        gin_rmse_buff = []
        gin_pk_err_buff = []
        gin_vl_err_buff = []
        gin_egy_err_buff = []
        gin_fce_buff = []

    for i, data in enumerate(testloader):
        model_input, temperature, mask, gt = data
        model_input = model_input.reshape((model_input.shape[0], model_input.shape[2])).to(device)
        mask = mask.reshape((mask.shape[0], mask.shape[2])).to(device)
        gt = gt.reshape((gt.shape[0], gt.shape[2])).to(device)
        temperature = temperature.reshape((temperature.shape[0], temperature.shape[2])).to(device)
        bs = gt.size(0)
        mask_np = mask.cpu().detach().numpy()

        if bm:
            # LSTM
            lstm_output = lstm(model_input, mask, temperature)
            lstm_output = model_input[:, :] + lstm_output * mask

        if bm:
            # GIN
            gin_input = torch.cat([model_input.unsqueeze(1), mask.unsqueeze(1), temperature.unsqueeze(1)], dim=1)
            pre, pre_coarse, _ = gin(gin_input)
            output = model_input.unsqueeze(1) + pre * mask.unsqueeze(1)
            gin_output = output.reshape((output.shape[0], output.shape[2])).to(device)

        # BERT
        bert_input = model_input.clone()
        bert_input = torch.round(bert_input * K).type(torch.int)
        temperature = torch.round(temperature * K).type(torch.int)
        bert_output = transformer(bert_input, temperature, mask)
        blin_output = bert_output.argmax(dim=-1)
        blin_output = blin_output / K
        blin_output = model_input[:, :] + blin_output * mask

        # SAE
        if bm:
            model_input_sae = model_input.clone()
            dim_day = 24 * NUM_H
            n_day = int(DIM_INPUT / dim_day)
            for n in range(n_day):
                input_day = model_input[:, n * dim_day: (n + 1) * dim_day]
                mask_np_sae = mask[:, n * dim_day: (n + 1) * dim_day].cpu().detach().numpy()

                for k in range(gt.size(0)):
                    patch = np.where(mask_np_sae[k, :] == 1.0)
                    if len(patch[0]) == 0:
                        continue
                    patch_bgn = patch[0][0]
                    patch_end = patch[0][-1]
                    load_np = input_day[k, :].cpu().detach().numpy()
                    m_ = 0.5 * (load_np[patch_bgn - 1] + load_np[patch_end + 1])
                    model_input_sae[k, patch_bgn + n * dim_day: n * dim_day + patch_end + 1] = \
                        m_ * torch.ones_like(model_input[k, patch_bgn + n * dim_day: n * dim_day + patch_end + 1])
            pre = sae(model_input_sae[:, :])
            sae_output = model_input_sae[:, :] + pre * mask

        pre_np_bert = blin_output.cpu().detach().numpy()
        if bm:
            pre_np_sae = sae_output.cpu().detach().numpy()
            pre_np_lstm = lstm_output.cpu().detach().numpy()
            pre_np_gin = gin_output.cpu().detach().numpy()
        input_np = model_input.cpu().detach().numpy()
        gt_np = gt.cpu().detach().numpy()

        fig = plt.figure(1, figsize=(3 * 3, 2 * 3))
        plt.clf()
        gs = fig.add_gridspec(3, 1)
        for j in range(3):
            ax = fig.add_subplot(gs[j, 0])
            buff = np.concatenate((pre_np_bert[j, :], gt_np[j, :]))
            if bm:
                buff = np.concatenate((buff, pre_np_sae[j, :], pre_np_lstm[j, :], pre_np_gin[j, :]))

            y_min = np.amin(buff)
            y_max = np.amax(buff)
            dim_day = 24 * NUM_H
            for n in range(int(DIM_INPUT / dim_day)):
                patch = np.where(mask_np[j, n * dim_day: (n + 1) * dim_day] == 1.0)
                if len(patch[0]) != 0:
                    patch_bgn = patch[0][0] - 1 + n * dim_day
                    patch_end = patch[0][-1] + 1 + n * dim_day
                    x = np.arange(patch_bgn, patch_end + 1)
                    # draw output data of the test set
                    ax.plot(x, pre_np_bert[j, patch_bgn:patch_end + 1], 'r', linewidth=1, label='BERT-PIN')
                    if bm:
                        ax.plot(x, pre_np_sae[j, patch_bgn:patch_end + 1], 'b', linewidth=1, label='SAE')
                        ax.plot(x, pre_np_lstm[j, patch_bgn:patch_end + 1], 'y', linewidth=1, label='LSTM')
                        ax.plot(x, pre_np_gin[j, patch_bgn:patch_end + 1], 'm', linewidth=1, label='Load-PIN')
                    rect = plt.Rectangle((patch_bgn, y_min), patch_end - patch_bgn + 1, y_max - y_min,
                                         facecolor="k", alpha=0.1)
                    ax.add_patch(rect)
                # draw gt of the test set
                ax.plot(np.arange(n * dim_day, (n + 1) * dim_day), gt_np[j, n * dim_day: (n + 1) * dim_day], 'g',
                        linewidth=1, label='GT')
                plt.ylim(y_min, y_max)
                plt.xticks([])
                plt.yticks([])
            legend_without_duplicate_labels(ax)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.pause(0.001)  # pause a bit so that plots are updated

        fn = '../plot/' + TAG + '/samples_pre' + str(i) + '.png'
        fig.savefig(fn, dpi=300)

        # calculate errs
        for idx in range(bs):
            mpe, rmse, pk_err, vl_err, egy_err, fft_err = 0, 0, 0, 0, 0, 0
            if bm:
                mpe_sae, rmse_sae, pk_err_sae, vl_err_sae, egy_err_sae, fft_err_sae = 0, 0, 0, 0, 0, 0
                mpe_lstm, rmse_lstm, pk_err_lstm, vl_err_lstm, egy_err_lstm, fft_err_lstm = 0, 0, 0, 0, 0, 0
                mpe_gin, rmse_gin, pk_err_gin, vl_err_gin, egy_err_gin, fft_err_gin = 0, 0, 0, 0, 0, 0
            dim_day = 24 * NUM_H
            for n in range(int(DIM_INPUT / dim_day)):
                pre_day = pre_np_bert[idx, n * dim_day: (n + 1) * dim_day]
                gt_day = gt_np[idx, n * dim_day: (n + 1) * dim_day]
                mask_day = mask_np[idx, n * dim_day: (n + 1) * dim_day]
                patch = np.where(mask_day == 1.0)
                if len(patch[0]) == 0:
                    continue
                patch_bgn = patch[0][0]
                patch_end = patch[0][-1] + 1
                rmse += mean_squared_error(pre_day[patch_bgn:patch_end], gt_day[patch_bgn:patch_end]) ** 0.5
                # percentage error: (gt - pre) / gt
                pe = abs(gt_day[patch_bgn:patch_end] - pre_day[patch_bgn:patch_end]) / gt_day[patch_bgn:patch_end]
                mpe += np.mean(pe)
                # energy error
                egy_err += abs(np.sum(pre_day[patch_bgn:patch_end]) - np.sum(gt_day[patch_bgn:patch_end])) / np.sum(
                    gt_day[patch_bgn:patch_end])
                # peak error, valley error, peak-valley error
                pk_err_, vl_err_, pkvy_err_ = evaluation.peak_valley_err(pre_day[patch_bgn:patch_end], gt_day[patch_bgn:patch_end])
                pk_err += pk_err_
                vl_err += vl_err_
                # frequency components error
                f_pre = abs(fft(pre_day[patch_bgn:patch_end]))
                f_gt = abs(fft(gt_day[patch_bgn:patch_end]))
                fft_err += np.sum(abs(f_gt - f_pre)) / (patch_end - patch_bgn + 1)

                if bm:
                    pre_day_sae = pre_np_sae[idx, n * dim_day: (n + 1) * dim_day]
                    pre_day_lstm = pre_np_lstm[idx, n * dim_day: (n + 1) * dim_day]
                    pre_day_gin = pre_np_gin[idx, n * dim_day: (n + 1) * dim_day]
                    rmse_sae += mean_squared_error(pre_day_sae[patch_bgn:patch_end],
                                                  gt_day[patch_bgn:patch_end]) ** 0.5
                    rmse_lstm += mean_squared_error(pre_day_lstm[patch_bgn:patch_end],
                                                   gt_day[patch_bgn:patch_end]) ** 0.5
                    rmse_gin += mean_squared_error(pre_day_gin[patch_bgn:patch_end],
                                                    gt_day[patch_bgn:patch_end]) ** 0.5
                    pe_sae = abs(gt_day[patch_bgn:patch_end] - pre_day_sae[patch_bgn:patch_end]) / gt_day[patch_bgn:patch_end]
                    pe_lstm = abs(gt_day[patch_bgn:patch_end] - pre_day_lstm[patch_bgn:patch_end]) / gt_day[patch_bgn:patch_end]
                    pe_gin = abs(gt_day[patch_bgn:patch_end] - pre_day_gin[patch_bgn:patch_end]) / gt_day[patch_bgn:patch_end]
                    mpe_sae += np.mean(pe_sae)
                    mpe_lstm += np.mean(pe_lstm)
                    mpe_gin += np.mean(pe_gin)
                    egy_err_sae += abs(
                        np.sum(pre_day_sae[patch_bgn:patch_end]) - np.sum(
                            gt_day[patch_bgn:patch_end])) / np.sum(
                        gt_day[patch_bgn:patch_end])
                    egy_err_lstm += abs(
                        np.sum(pre_day_lstm[patch_bgn:patch_end]) - np.sum(
                            gt_day[patch_bgn:patch_end])) / np.sum(
                        gt_day[patch_bgn:patch_end])
                    egy_err_gin += abs(
                        np.sum(pre_day_gin[patch_bgn:patch_end]) - np.sum(
                            gt_day[patch_bgn:patch_end])) / np.sum(
                        gt_day[patch_bgn:patch_end])
                    pk_err_sae_, vl_err_sae_, pkvy_err_sae = evaluation.peak_valley_err(
                        pre_day_sae[patch_bgn:patch_end],
                        gt_day[patch_bgn:patch_end])
                    pk_err_lstm_, vl_err_lstm_, pkvy_err_lstm = evaluation.peak_valley_err(
                        pre_day_lstm[patch_bgn:patch_end],
                        gt_day[patch_bgn:patch_end])
                    pk_err_gin_, vl_err_gin_, pkvy_err_gin = evaluation.peak_valley_err(
                        pre_day_gin[patch_bgn:patch_end],
                        gt_day[patch_bgn:patch_end])
                    pk_err_sae += pk_err_sae_
                    vl_err_sae += vl_err_sae_
                    pk_err_lstm += pk_err_lstm_
                    vl_err_lstm += vl_err_lstm_
                    pk_err_gin += pk_err_gin_
                    vl_err_gin += vl_err_gin_
                    f_pre_sae = abs(fft(pre_day_sae[patch_bgn:patch_end]))
                    f_pre_lstm = abs(fft(pre_day_lstm[patch_bgn:patch_end]))
                    f_pre_gin = abs(fft(pre_day_gin[patch_bgn:patch_end]))
                    fft_err_sae += np.sum(abs(f_gt - f_pre_sae)) / (patch_end - patch_bgn + 1)
                    fft_err_lstm += np.sum(abs(f_gt - f_pre_lstm)) / (patch_end - patch_bgn + 1)
                    fft_err_gin += np.sum(abs(f_gt - f_pre_gin)) / (patch_end - patch_bgn + 1)

            bert_mpe_buff.append(mpe)
            bert_rmse_buff.append(rmse)
            bert_pk_err_buff.append(pk_err)
            bert_vl_err_buff.append(vl_err)
            bert_egy_err_buff.append(egy_err)
            bert_fce_buff.append(fft_err)

            if bm:
                sae_mpe_buff.append(mpe_sae)
                sae_rmse_buff.append(rmse_sae)
                sae_pk_err_buff.append(pk_err_sae)
                sae_vl_err_buff.append(vl_err_sae)
                sae_egy_err_buff.append(egy_err_sae)
                sae_fce_buff.append(fft_err_sae)
                lstm_mpe_buff.append(mpe_lstm)
                lstm_rmse_buff.append(rmse_lstm)
                lstm_pk_err_buff.append(pk_err_lstm)
                lstm_vl_err_buff.append(vl_err_lstm)
                lstm_egy_err_buff.append(egy_err_lstm)
                lstm_fce_buff.append(fft_err_lstm)
                gin_mpe_buff.append(mpe_gin)
                gin_rmse_buff.append(rmse_gin)
                gin_pk_err_buff.append(pk_err_gin)
                gin_vl_err_buff.append(vl_err_gin)
                gin_egy_err_buff.append(egy_err_gin)
                gin_fce_buff.append(fft_err_gin)

        if i > 10:
            break

    figure = plt.figure(2)
    if bm:
        violin_parts = plt.violinplot([bert_mpe_buff, sae_mpe_buff, lstm_mpe_buff, gin_mpe_buff,
                                       bert_rmse_buff, sae_rmse_buff, lstm_rmse_buff, gin_rmse_buff,
                                       bert_pk_err_buff, sae_pk_err_buff, lstm_pk_err_buff, gin_pk_err_buff,
                                       bert_vl_err_buff, sae_vl_err_buff, lstm_vl_err_buff, gin_vl_err_buff,
                                       bert_egy_err_buff, sae_egy_err_buff, lstm_egy_err_buff, gin_egy_err_buff,
                                       bert_fce_buff, sae_fce_buff, lstm_fce_buff, gin_fce_buff], showmedians=True,
                                      showextrema=False)
        for n in range(len(violin_parts['bodies'])):
            pc = violin_parts['bodies'][n]
            if n % 4 == 0:
                pc.set_facecolor('red')
                pc.set_edgecolor('red')
            elif n % 4 == 1:
                pc.set_facecolor('blue')
                pc.set_edgecolor('blue')
            elif n % 4 == 2:
                pc.set_facecolor('yellow')
                pc.set_edgecolor('yellow')
            elif n % 4 == 3:
                pc.set_facecolor('magenta')
                pc.set_edgecolor('magenta')
        plt.xticks(range(1, 25), labels=['', 'mpe', '', '', '',
                                         'rmse', '', '', '',
                                         'pk_err', '', '', '',
                                         'vl_err', '', '', '',
                                         'egy_err', '', '', '',
                                         'fce_err', '', ''])
    else:
        violin_parts = plt.violinplot([bert_mpe_buff,
                                       bert_rmse_buff,
                                       bert_pk_err_buff,
                                       bert_vl_err_buff,
                                       bert_egy_err_buff,
                                       bert_fce_buff], showmedians=True, showextrema=False)
        plt.xticks(range(1, 7), labels=['mpe',
                                         'rmse',
                                         'pk_err',
                                         'vl_err',
                                         'egy_err',
                                         'fce_err'])
    plt.show()
    fn = '../plot/' + TAG + '/violinplot.png'
    figure.savefig(fn, dpi=300)
    if bm:
        print(np.mean(bert_mpe_buff), np.mean(sae_mpe_buff), np.mean(lstm_mpe_buff), np.mean(gin_mpe_buff), '\n',
              np.mean(bert_rmse_buff), np.mean(sae_rmse_buff), np.mean(lstm_rmse_buff), np.mean(gin_rmse_buff), '\n',
              np.mean(bert_pk_err_buff), np.mean(sae_pk_err_buff), np.mean(lstm_pk_err_buff), np.mean(gin_pk_err_buff),
              '\n',
              np.mean(bert_vl_err_buff), np.mean(sae_vl_err_buff), np.mean(lstm_vl_err_buff), np.mean(gin_vl_err_buff),
              '\n',
              np.mean(bert_egy_err_buff), np.mean(sae_egy_err_buff), np.mean(lstm_egy_err_buff),
              np.mean(gin_egy_err_buff), '\n',
              np.mean(bert_fce_buff), np.mean(sae_fce_buff), np.mean(lstm_fce_buff), np.mean(gin_fce_buff))
    else:
        print(np.mean(bert_mpe_buff), np.mean(bert_rmse_buff), np.mean(bert_pk_err_buff), np.mean(bert_vl_err_buff), np.mean(bert_egy_err_buff), np.mean(bert_fce_buff))




if __name__ == "__main__":
    # train_bert()
    # train_sae()
    # train_lstm()
    # train_gin()
    test()
    # test_7days_week()






