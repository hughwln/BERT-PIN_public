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
            mask_np = mask.cpu().detach().numpy()
            for k in range(gt.size(0)):
                load_np = model_input[k, :].cpu().detach().numpy()
                patch_bgn = np.where(mask_np[k, :] == 1.0)[0][0]
                patch_end = np.where(mask_np[k, :] == 1.0)[0][-1]
                m_ = 0.5 * (load_np[patch_bgn - 1] + load_np[patch_end + 1])
                model_input_sae = model_input
                model_input_sae[k, patch_bgn:patch_end + 1] = \
                    m_ * torch.ones_like(model_input[k, patch_bgn:patch_end + 1])
            output = transformer(model_input_sae[:, :])
            output = model_input_sae[:, :] + output * mask
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
            mask_np = mask.cpu().detach().numpy()
            for k in range(gt.size(0)):
                load_np = model_input[k, :].cpu().detach().numpy()
                patch_bgn = np.where(mask_np[k, :] == 1.0)[0][0]
                patch_end = np.where(mask_np[k, :] == 1.0)[0][-1]
                m_ = 0.5 * (load_np[patch_bgn - 1] + load_np[patch_end + 1])
                model_input_sae = model_input
                model_input_sae[k, patch_bgn:patch_end + 1] = \
                    m_ * torch.ones_like(model_input[k, patch_bgn:patch_end + 1])
            output = transformer(model_input_sae[:, :])
            output = model_input_sae[:, :] + output * mask
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

    fig = plt.figure(1, figsize=(2 * num_samples, 2 * num_samples))
    plt.clf()
    gs = fig.add_gridspec(num_samples, num_samples)
    mask_np = mask.cpu().detach().numpy()
    mask_test_np = mask_test.cpu().detach().numpy()
    for i in range(0, num_samples):
        buff = np.concatenate((pre_np[i, :], gt_np[i, :]))

        patch_bgn = np.where(mask_np[i, :] == 1.0)[0][0] - 1
        patch_end = np.where(mask_np[i, :] == 1.0)[0][-1] + 1
        x = np.arange(patch_bgn, patch_end + 1)
        y_min = np.amin(buff)
        y_max = np.amax(buff)
        # y_min = 0.0
        # y_max = 1.0

        ax = fig.add_subplot(gs[i, 0])
        # draw input data
        ax.plot(np.arange(0, patch_bgn+1),
                input_np[i, 0:patch_bgn+1],
                'b', linewidth=1)
        ax.plot(np.arange(patch_end, np.size(input_np, axis=1)),
                input_np[i, patch_end:],
                'b', linewidth=1)

        plt.ylim(y_min, y_max)
        plt.xticks([])
        plt.yticks([])
        rect = plt.Rectangle((patch_bgn, y_min), patch_end - patch_bgn + 1, y_max - y_min,
                             facecolor="k", alpha=0.1)
        ax.add_patch(rect)

        # draw output data
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(np.arange(0, patch_bgn+1),
                input_np[i, 0:patch_bgn+1],
                'r', linewidth=1)
        ax.plot(np.arange(patch_end, np.size(input_np, axis=1)),
                input_np[i, patch_end:],
                'r', linewidth=1)
        ax.plot(x, pre_np[i, patch_bgn:patch_end + 1], 'k', linewidth=1)
        # ax.plot(x, pre_np[i, patch_bgn - 1:patch_end + 2], 'k', linewidth=1)
        plt.ylim(y_min, y_max)
        plt.xticks([])
        plt.yticks([])
        rect = plt.Rectangle((patch_bgn, y_min), patch_end - patch_bgn + 1, y_max - y_min,
                             facecolor="k", alpha=0.1)
        ax.add_patch(rect)

        # draw gt
        ax = fig.add_subplot(gs[i, 2])
        ax.plot(gt_np[i, :], 'g', linewidth=1)
        ax.plot(x, gt_np[i, patch_bgn:patch_end + 1], 'k', linewidth=1)
        plt.ylim(y_min, y_max)
        plt.xticks([])
        plt.yticks([])
        rect = plt.Rectangle((patch_bgn, y_min), patch_end - patch_bgn + 1, y_max - y_min,
                             facecolor="k", alpha=0.1)
        ax.add_patch(rect)

        # draw output data of the test set
        buff_test = np.concatenate((pre_np_test[i, :], gt_np_test[i, :]))
        y_min_test = np.amin(buff_test)
        y_max_test = np.amax(buff_test)
        # y_min_test = 0.0
        # y_max_test = 1.0
        patch_bgn = np.where(mask_test_np[i, :] == 1.0)[0][0] - 1
        patch_end = np.where(mask_test_np[i, :] == 1.0)[0][-1] + 1
        x = np.arange(patch_bgn, patch_end + 1)

        ax = fig.add_subplot(gs[i, 3])
        ax.plot(np.arange(0, patch_bgn+1),
                gt_np_test[i, 0:patch_bgn+1],
                'b', linewidth=1)
        ax.plot(np.arange(patch_end, np.size(gt_np_test, axis=1)),
                gt_np_test[i, patch_end:],
                'b', linewidth=1)
        ax.plot(x, pre_np_test[i, patch_bgn:patch_end + 1], 'k', linewidth=1)

        # ax.plot(x, pre_np_test[i, patch_bgn - 1:patch_end + 2], 'k', linewidth=1)
        plt.ylim(y_min_test, y_max_test)
        plt.xticks([])
        plt.yticks([])
        rect = plt.Rectangle((patch_bgn, y_min_test), patch_end - patch_bgn + 1, y_max_test - y_min_test,
                             facecolor="k", alpha=0.1)
        ax.add_patch(rect)

        # draw gt of the test set
        ax = fig.add_subplot(gs[i, 4])
        ax.plot(gt_np_test[i, :], 'g', linewidth=1)
        ax.plot(x, gt_np_test[i, patch_bgn:patch_end + 1], 'k', linewidth=1)
        plt.ylim(y_min_test, y_max_test)
        plt.xticks([])
        plt.yticks([])
        rect = plt.Rectangle((patch_bgn, y_min_test), patch_end - patch_bgn + 1, y_max_test - y_min_test,
                             facecolor="k", alpha=0.1)
        ax.add_patch(rect)

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
        if transformer.name == 'GIN':
            model_input = torch.cat([model_input, mask, temperature], dim=1)
        else:
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
            mask_np = mask.cpu().detach().numpy()
            for k in range(gt.size(0)):
                load_np = model_input[k, :].cpu().detach().numpy()
                patch_bgn = np.where(mask_np[k, :] == 1.0)[0][0]
                patch_end = np.where(mask_np[k, :] == 1.0)[0][-1]
                m_ = 0.5 * (load_np[patch_bgn - 1] + load_np[patch_end + 1])
                model_input_sae = model_input
                model_input_sae[k, patch_bgn:patch_end + 1] = \
                    m_ * torch.ones_like(model_input[k, patch_bgn:patch_end + 1])
            output = transformer(model_input_sae[:, :])
            output = model_input_sae[:, :] + output * mask
            loss = calc_benchmark_loss(output, gt, mask)
        elif transformer.name == 'LSTM':
            output = transformer(model_input, mask, temperature)
            output = model_input[:, :] + output * mask
            loss = calc_benchmark_loss(output, gt, mask)
        elif transformer.name == 'GIN':
            pre, pre_coarse, _ = transformer(model_input)
            output = model_input[:, 0, :].unsqueeze(1) + pre * mask.unsqueeze(1)
            output = output.reshape((output.shape[0], output.shape[2])).to(device)
            loss = calc_benchmark_loss(output, gt, mask)

        loss_test_rec.append(loss.item())

        # Accuracy
        mask_np = mask.cpu().detach().numpy()
        pre_np = output.cpu().detach().numpy()
        gt_np = gt.cpu().detach().numpy()

        for idx in range(bs):
            patch_bgn = np.where(mask_np[idx, :] == 1.0)[0][0]
            patch_end = np.where(mask_np[idx, :] == 1.0)[0][-1] + 1
            rmse = mean_squared_error(pre_np[idx, patch_bgn:patch_end], gt_np[idx, patch_bgn:patch_end]) ** 0.5
            # percentage error: (gt - pre) / gt
            pe = abs(gt_np[idx, patch_bgn:patch_end] - pre_np[idx, patch_bgn:patch_end]) / gt_np[idx, patch_bgn:patch_end]
            mpe = np.mean(pe)
            # energy error
            egy_err = abs(np.sum(pre_np[idx, patch_bgn:patch_end]) - np.sum(gt_np[idx, patch_bgn:patch_end])) / np.sum(
                gt_np[idx, patch_bgn:patch_end])
            # peak error, valley error, peak-valley error
            pk_err, vl_err, pkvy_err = peak_valley_err(pre_np[idx, patch_bgn:patch_end], gt_np[idx, patch_bgn:patch_end])
            # frequency components error
            f_pre = abs(fft(pre_np[idx, patch_bgn:patch_end]))
            f_gt = abs(fft(gt_np[idx, patch_bgn:patch_end]))
            fft_err = np.sum(abs(f_gt - f_pre)) / (patch_end - patch_bgn + 1)

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
