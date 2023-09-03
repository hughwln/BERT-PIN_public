#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  26 2023

@author: Yi Hu
"""

import matplotlib.pyplot as plt
from config import *
from dataset import testloader, trainloader, devloader
import numpy as np
from train import calc_loss, calc_benchmark_loss
from sklearn.metrics import mean_squared_error
from scipy.fftpack import fft

def peak_valley_err(pred, gt):
    pred_max = np.amax(pred)
    gt_max = np.amax(gt)
    pred_min = np.amin(pred)
    gt_min = np.amin(gt)
    peak_err = abs(pred_max - gt_max)
    valley_err = abs(pred_min - gt_min)
    pk_valley_err = abs((pred_max - pred_min) - (gt_max - gt_min))

    return peak_err, valley_err, pk_valley_err

def plot_samples(transformer, epoch, K, device):
    transformer.eval()
    num_samples = 5

    for i, data in enumerate(trainloader, 0):
        model_input, temperature, mask, gt = data
        model_input = model_input.reshape((model_input.shape[0], model_input.shape[2])).to(device)
        mask = mask.reshape((mask.shape[0], mask.shape[2])).to(device)
        gt = gt.reshape((gt.shape[0], gt.shape[2])).to(device)
        temperature = temperature.reshape((temperature.shape[0], temperature.shape[2])).to(device)

        if transformer.name == 'BERT':
            model_input = torch.round(model_input * K).type(torch.int)
            temperature = torch.round(temperature * K).type(torch.int)
            output = transformer(model_input, temperature, mask)
            output = output.argmax(dim=-1)
            output = output / K
            model_input = model_input / K
        elif transformer.name == 'SAE':
            model_input_sae = model_input
            dim_day = 24 * NUM_H
            n_day = int(DIM_INPUT / dim_day)
            for n in range(n_day):
                input_day = model_input[:, n * dim_day: (n + 1) * dim_day]
                mask_np = mask[:, n * dim_day: (n + 1) * dim_day].cpu().detach().numpy()

                for k in range(gt.size(0)):
                    patch = np.where(mask_np[k, :] == 1.0)
                    if len(patch[0]) == 0:
                        continue
                    patch_bgn = patch[0][0]
                    patch_end = patch[0][-1]
                    load_np = input_day[k, :].cpu().detach().numpy()
                    m_ = 0.5 * (load_np[patch_bgn - 1] + load_np[patch_end + 1])
                    model_input_sae[k, patch_bgn + n * dim_day: n * dim_day + patch_end + 1] = \
                        m_ * torch.ones_like(model_input[k, patch_bgn + n * dim_day: n * dim_day + patch_end + 1])
            pre = transformer(model_input_sae[:, :])
            output = model_input_sae[:, :] + pre * mask
        elif transformer.name == 'LSTM':
            output = transformer(model_input, mask, temperature)
            output = model_input[:, :] + output * mask
        elif transformer.name == 'GIN':
            gin_input = torch.cat([model_input.unsqueeze(1), mask.unsqueeze(1), temperature.unsqueeze(1)], dim=1)
            pre, pre_coarse, _ = transformer(gin_input)
            output = model_input.unsqueeze(1) + pre * mask.unsqueeze(1)
            output = output.reshape((output.shape[0], output.shape[2])).to(device)

        pre_np = output.cpu().detach().numpy()
        input_np = model_input.cpu().detach().numpy()
        gt_np = gt.cpu().detach().numpy()

        break

    for i, data in enumerate(testloader, 0):
        model_input, temperature, mask_test, gt = data
        model_input = model_input.reshape((model_input.shape[0], model_input.shape[2])).to(device)
        mask_test = mask_test.reshape((mask_test.shape[0], mask_test.shape[2])).to(device)
        gt = gt.reshape((gt.shape[0], gt.shape[2])).to(device)
        temperature = temperature.reshape((temperature.shape[0], temperature.shape[2])).to(device)

        if transformer.name == 'BERT':
            model_input = torch.round(model_input * K).type(torch.int)
            temperature = torch.round(temperature * K).type(torch.int)
            output = transformer(model_input, temperature, mask)
            output = output.argmax(dim=-1)
            output = output / K
            model_input = model_input / K
        elif transformer.name == 'SAE':
            model_input_sae = model_input
            dim_day = 24 * NUM_H
            n_day = int(DIM_INPUT / dim_day)
            for n in range(n_day):
                input_day = model_input[:, n * dim_day: (n + 1) * dim_day]
                mask_np = mask[:, n * dim_day: (n + 1) * dim_day].cpu().detach().numpy()

                for k in range(gt.size(0)):
                    patch = np.where(mask_np[k, :] == 1.0)
                    if len(patch[0]) == 0:
                        continue
                    patch_bgn = patch[0][0]
                    patch_end = patch[0][-1]
                    load_np = input_day[k, :].cpu().detach().numpy()
                    m_ = 0.5 * (load_np[patch_bgn - 1] + load_np[patch_end + 1])
                    model_input_sae[k, patch_bgn + n * dim_day: n * dim_day + patch_end + 1] = \
                        m_ * torch.ones_like(model_input[k, patch_bgn + n * dim_day: n * dim_day + patch_end + 1])
            pre = transformer(model_input_sae[:, :])
            output = model_input_sae[:, :] + pre * mask
        elif transformer.name == 'LSTM':
            output = transformer(model_input, mask, temperature)
            output = model_input[:, :] + output * mask
        elif transformer.name == 'GIN':
            gin_input = torch.cat([model_input.unsqueeze(1), mask.unsqueeze(1), temperature.unsqueeze(1)], dim=1)
            pre, pre_coarse, _ = transformer(gin_input)
            output = model_input.unsqueeze(1) + pre * mask.unsqueeze(1)
            output = output.reshape((output.shape[0], output.shape[2])).to(device)

        pre_np_test = output.cpu().detach().numpy()
        input_np_test = model_input.cpu().detach().numpy()
        gt_np_test = gt.cpu().detach().numpy()

        break

    fig = plt.figure(1, figsize=(3 * num_samples, 2 * num_samples))
    plt.clf()
    gs = fig.add_gridspec(num_samples, 1)
    mask_np = mask.cpu().detach().numpy()
    mask_test_np = mask_test.cpu().detach().numpy()
    for i in range(0, num_samples):
        ax = fig.add_subplot(gs[i, 0])
        buff = np.concatenate((pre_np_test[i, :], gt_np_test[i, :]))
        y_min_test = np.amin(buff)
        y_max_test = np.amax(buff)
        dim_day = 24 * NUM_H
        for n in range(int(DIM_INPUT / dim_day)):
            patch = np.where(mask_test_np[i, n*dim_day: (n+1)*dim_day] == 1.0)
            if len(patch[0]) != 0:
                patch_bgn = patch[0][0] - 1 + n*dim_day
                patch_end = patch[0][-1] + 1 + n*dim_day
                x = np.arange(patch_bgn, patch_end + 1)
                # draw output data of the test set
                ax.plot(x, pre_np_test[i, patch_bgn:patch_end + 1], 'r', linewidth=1)
                rect = plt.Rectangle((patch_bgn, y_min_test), patch_end - patch_bgn + 1, y_max_test - y_min_test,
                                     facecolor="k", alpha=0.1)
                ax.add_patch(rect)
            # draw gt of the test set
            ax.plot(np.arange(n * dim_day, (n + 1) * dim_day), gt_np_test[i, n*dim_day: (n+1)*dim_day], 'g', linewidth=1)
            plt.ylim(y_min_test, y_max_test)
            plt.xticks([])
            plt.yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.pause(0.001)  # pause a bit so that plots are updated

    if epoch % 10 == 0:
        fn = '../plot/' + TAG + '/' + transformer.name +'samples_epoch' + str(epoch) + '.png'
        fig.savefig(fn)

    transformer.train()

def eval_set(transformer, K, criterion, device):
    transformer.eval()
    loss_test_rec = []

    rmse_buff = []
    pk_err_buff = []
    vl_err_buff = []
    mpe_buff = []
    egy_err_buff = []
    fce_buff = []

    for i, data in enumerate(devloader, 0):
        model_input, temperature, mask, gt = data
        model_input = model_input.reshape((model_input.shape[0], model_input.shape[2])).to(device)
        mask = mask.reshape((mask.shape[0], mask.shape[2])).to(device)
        gt = gt.reshape((gt.shape[0], gt.shape[2])).to(device)
        temperature = temperature.reshape((temperature.shape[0], temperature.shape[2])).to(device)
        bs = gt.size(0)

        if transformer.name == 'BERT':
            model_input = torch.round(model_input * K).type(torch.int)
            temperature = torch.round(temperature * K).type(torch.int)
            gt = torch.round(gt * K).type(torch.int)
            gt = gt.reshape((gt.shape[0], gt.shape[1]), -1).type(torch.LongTensor).to(device)
            output = transformer(model_input, temperature, mask)
            loss = calc_loss(output, gt, mask, criterion, K)
            output = output.argmax(dim=-1)
            output = output / K
            gt = gt / K
        elif transformer.name == 'SAE':
            model_input_sae = model_input
            dim_day = 24 * NUM_H
            n_day = int(DIM_INPUT / dim_day)
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
            pre = transformer(model_input_sae[:, :])
            output = model_input_sae[:, :] + pre * mask
            loss = calc_benchmark_loss(output, gt, mask)
        elif transformer.name == 'LSTM':
            output = transformer(model_input, mask, temperature)
            output = model_input[:, :] + output * mask
            loss = calc_benchmark_loss(output, gt, mask)
        elif transformer.name == 'GIN':
            gin_input = torch.cat([model_input.unsqueeze(1), mask.unsqueeze(1), temperature.unsqueeze(1)], dim=1)
            pre, pre_coarse, _ = transformer(gin_input)
            output = model_input.unsqueeze(1) + pre * mask.unsqueeze(1)
            output = output.reshape((output.shape[0], output.shape[2])).to(device)
            loss = calc_benchmark_loss(output, gt, mask)

        loss_test_rec.append(loss.item())

        # Accuracy
        mask_np = mask.cpu().detach().numpy()
        pre_np = output.cpu().detach().numpy()
        gt_np = gt.cpu().detach().numpy()

        for idx in range(bs):
            mpe, rmse, pk_err, vl_err, egy_err, fft_err = 0, 0, 0, 0, 0, 0
            dim_day = 24 * NUM_H
            for n in range(int(DIM_INPUT / dim_day)):
                pre_day = pre_np[idx, n * dim_day: (n + 1) * dim_day]
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
                pk_err_, vl_err_, pkvy_err_ = peak_valley_err(pre_day[patch_bgn:patch_end], gt_day[patch_bgn:patch_end])
                pk_err += pk_err_
                vl_err += vl_err_
                # frequency components error
                f_pre = abs(fft(pre_day[patch_bgn:patch_end]))
                f_gt = abs(fft(gt_day[patch_bgn:patch_end]))
                fft_err += np.sum(abs(f_gt - f_pre)) / (patch_end - patch_bgn + 1)

            mpe_buff.append(mpe)
            rmse_buff.append(rmse)
            pk_err_buff.append(pk_err)
            vl_err_buff.append(vl_err)
            egy_err_buff.append(egy_err)
            fce_buff.append(fft_err)

    ave_loss = np.mean(loss_test_rec)

    ave_mpe = np.mean(mpe_buff)
    ave_rmse = np.mean(rmse_buff)
    ave_pk_err = np.mean(pk_err_buff)
    ave_vl_err = np.mean(vl_err_buff)
    ave_egy_err = np.mean(egy_err_buff)
    ave_fce = np.mean(fce_buff)

    transformer.train()

    return ave_loss, [ave_mpe, ave_rmse, ave_pk_err, ave_vl_err, ave_egy_err, ave_fce]

def plot_loss(loss_train_rec, loss_eval_rec, epoch, name):

    fig = plt.figure(1, figsize=(5, 5))
    plt.clf()

    plt.plot(loss_train_rec, 'r', linewidth=1, label="train_loss")
    plt.plot(loss_eval_rec, 'b', linewidth=1, label="eval_loss")
    plt.legend()

    # plt.pause(0.001)  # pause a bit so that plots are updated

    if epoch % 10 == 0:
        fn = '../plot/' + TAG + '/' + name +'loss' + str(epoch) + '.png'
        fig.savefig(fn)

def plot_err(err_rec, epoch, name):
    errs = np.transpose(err_rec)
    fig = plt.figure(1, figsize=(5, 5))
    plt.clf()
    '[ave_mpe, ave_rmse, ave_pk_err, ave_vl_err, ave_egy_err, ave_fce]'
    plt.plot(errs[0], linewidth=1, label="mpe")
    plt.plot(errs[1], linewidth=1, label="rmse")
    plt.plot(errs[2], linewidth=1, label="pk_err")
    plt.plot(errs[3], linewidth=1, label="vl_err")
    plt.plot(errs[4], linewidth=1, label="egy_err")
    plt.plot(errs[5], linewidth=1, label="fce")
    plt.legend()

    # plt.pause(0.001)  # pause a bit so that plots are updated

    if epoch % 10 == 0:
        fn = '../plot/' + TAG + '/' + name +'err' + str(epoch) + '.png'
        fig.savefig(fn)
