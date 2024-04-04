#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 2023

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


def blin_2nd_iter(transformer, bert_input, temperature, mask, k):
    for i in range(int(config.DIM_PATCH/2)):
        blin_output = transformer(bert_input, temperature, mask)
        mask_np = mask.cpu().detach().numpy()
        for j in range(bert_input.size(0)):
            patch_bgn = np.where(mask_np[j, :] == 1.0)[0][0]
            patch_end = np.where(mask_np[j, :] == 1.0)[0][-1]
            bert_input[j, patch_bgn] = torch.topk(blin_output[j, patch_bgn], k=k)[1][-1]
            bert_input[j, patch_end] = torch.topk(blin_output[j, patch_end], k=k)[1][-1]
            mask[j, patch_bgn] = False
            mask[j, patch_end] = False

    return bert_input

def blin_2nd_iter_improved(transformer, bert_input, temperature, mask, k=2, e=0.1):
    # find the fork position
    for j in range(bert_input.size(0)):
        bert_output = transformer(bert_input[j:j+1, :], temperature[j:j+1, :], mask[j:j+1, :])
        mask_np = mask[j:j+1, :].cpu().detach().numpy()
        patch_bgn = np.where(mask_np[0, :] == 1.0)[0][0]
        patch_end = np.where(mask_np[0, :] == 1.0)[0][-1]
        p_left = patch_bgn
        p_right = patch_end
        left_fork = patch_bgn + int(config.DIM_PATCH/2)
        right_fork = patch_end - int(config.DIM_PATCH/2)
        while(p_left < left_fork):
            bgn_top1 = torch.topk(bert_output[0, p_left], k=1)[0][-1]
            bgn_topk = torch.topk(bert_output[0, p_left], k=2)[0][-1]
            if bgn_top1 - bgn_topk < e:
                left_fork = p_left
                bert_input[j, left_fork] = torch.topk(bert_output[0, p_left], k=2)[1][-1]

                break
            else:
                p_left += 1

        while(p_right > right_fork):
            end_top1 = torch.topk(bert_output[0, p_right], k=1)[0][-1]
            end_topk = torch.topk(bert_output[0, p_right], k=2)[0][-1]
            if end_top1 - end_topk < e:
                right_fork = p_right
                bert_input[j, right_fork] = torch.topk(bert_output[0, p_right], k=2)[1][-1]

                break
            else:
                p_right -= 1

        bert_input[j, patch_bgn: left_fork] = bert_output[0, patch_bgn: left_fork].argmax(dim=-1)
        bert_input[j, right_fork + 1: patch_end + 1] = bert_output[0, right_fork + 1: patch_end + 1].argmax(dim=-1)

        for i in range(left_fork + 1 - patch_bgn, int(config.DIM_PATCH/2)):
            shift_left = torch.zeros_like(bert_input[j:j+1, :])
            shift_left[:, :-i] = bert_input[j:j+1, i:].clone()
            shift_left[:, -i:] = bert_input[j:j+1, :i].clone()
            temp_left = torch.zeros_like(temperature[j:j+1, :])
            temp_left[:, :-i] = temperature[j:j+1, i:].clone()
            temp_left[:, -i:] = temperature[j:j+1, :i].clone()
            blin_output_left = transformer(shift_left * ~mask[j:j+1, :], temp_left, mask[j:j+1, :])

            bert_input[j, patch_bgn + i] = torch.topk(blin_output_left[0, patch_bgn], k=k)[1][0]
            show = bert_input[j, :]

        for i in range(patch_end - right_fork + 1, int(config.DIM_PATCH / 2)):
            shift_right = torch.zeros_like(bert_input[j:j+1, :])
            shift_right[:, i:] = bert_input[j:j + 1, :-i].clone()
            shift_right[:, :i] = bert_input[j:j + 1, -i:].clone()
            temp_right = torch.zeros_like(temperature[j:j+1, :])
            temp_right[:, i:] = temperature[j:j + 1, :-i].clone()
            temp_right[:, :i] = temperature[j:j + 1, -i:].clone()
            blin_output_right = transformer(shift_right * ~mask[j:j+1, :], temp_right, mask[j:j+1, :])

            bert_input[j, patch_end - i] = torch.topk(blin_output_right[0, patch_end], k=k)[1][0]
            show = bert_input[j, :]


    return bert_input

def blin_2nd_iter_noniter(transformer, bert_input, temperature, mask, k=2, e=0.1):
    # find the fork position
    for j in range(bert_input.size(0)):
        bert_output = transformer(bert_input[j:j+1, :], temperature[j:j+1, :], mask[j:j+1, :])
        mask_np = mask[j:j+1, :].cpu().detach().numpy()
        patch_bgn = np.where(mask_np[0, :] == 1.0)[0][0]
        patch_end = np.where(mask_np[0, :] == 1.0)[0][-1]
        p_left = patch_bgn
        p_right = patch_end
        left_fork = patch_bgn + int(config.DIM_PATCH/2)
        right_fork = patch_end - int(config.DIM_PATCH/2)
        while(p_left < left_fork):
            bgn_top1 = torch.topk(bert_output[0, p_left], k=1)[0][-1]
            bgn_topk = torch.topk(bert_output[0, p_left], k=2)[0][-1]
            if bgn_top1 - bgn_topk < e:
                left_fork = p_left
                bert_input[j, left_fork] = torch.topk(bert_output[0, p_left], k=2)[1][-1]

                break
            else:
                p_left += 1

        while(p_right > right_fork):
            end_top1 = torch.topk(bert_output[0, p_right], k=1)[0][-1]
            end_topk = torch.topk(bert_output[0, p_right], k=2)[0][-1]
            if end_top1 - end_topk < e:
                right_fork = p_right
                bert_input[j, right_fork] = torch.topk(bert_output[0, p_right], k=2)[1][-1]

                break
            else:
                p_right -= 1

        bert_input[j, patch_bgn: left_fork] = bert_output[0, patch_bgn: left_fork].argmax(dim=-1)
        bert_input[j, right_fork + 1: patch_end + 1] = bert_output[0, right_fork + 1: patch_end + 1].argmax(dim=-1)

        i = left_fork + 1 - patch_bgn
        shift_left = torch.zeros_like(bert_input[j:j+1, :])
        shift_left[:, :-i] = bert_input[j:j+1, i:].clone()
        shift_left[:, -i:] = bert_input[j:j+1, :i].clone()
        temp_left = torch.zeros_like(temperature[j:j+1, :])
        temp_left[:, :-i] = temperature[j:j+1, i:].clone()
        temp_left[:, -i:] = temperature[j:j+1, :i].clone()
        blin_output_left = transformer(shift_left * ~mask[j:j+1, :], temp_left, mask[j:j+1, :])

        for p in range(patch_bgn, patch_bgn + int(config.DIM_PATCH / 2) - i):
            aaa = blin_output_left[0, p]
            bbb = torch.topk(aaa, k=k)[1][0]
            bert_input[j, p + i] = bbb
        show = bert_input[j, :]

        i = patch_end - right_fork + 1
        shift_right = torch.zeros_like(bert_input[j:j+1, :])
        shift_right[:, i:] = bert_input[j:j + 1, :-i].clone()
        shift_right[:, :i] = bert_input[j:j + 1, -i:].clone()
        temp_right = torch.zeros_like(temperature[j:j+1, :])
        temp_right[:, i:] = temperature[j:j + 1, :-i].clone()
        temp_right[:, :i] = temperature[j:j + 1, -i:].clone()
        blin_output_right = transformer(shift_right * ~mask[j:j+1, :], temp_right, mask[j:j+1, :])

        for p in range(patch_bgn + int(config.DIM_PATCH / 2) + i, patch_end + 1):
            aaa = blin_output_left[0, p]
            bbb = torch.topk(aaa, k=k)[1][0]
            bert_input[j, p - i] = bbb
        show = bert_input[j, :]


    return bert_input

def blin_combine(blin_output, blin2i_output, gt):
    for i in range(gt.size(0)):
        for j in range(gt.size(1)):
            dis1 = abs(blin_output[i, j] - gt[i, j])
            dis2 = abs(blin2i_output[i, j] - gt[i, j])
            if dis1 > dis2:
                blin_output[i, j] = blin2i_output[i, j]

    return blin_output

def reconciliation(bert_output, blin_output, blin2i_output, mask, e=0.1):
    for i in range(int(config.DIM_PATCH/2)):
        mask_np = mask.cpu().detach().numpy()
        for j in range(bert_output.size(0)):
            patch_bgn = np.where(mask_np[j, :] == 1.0)[0][0]
            patch_end = np.where(mask_np[j, :] == 1.0)[0][-1]
            bgn_top1 = torch.topk(bert_output[j, patch_bgn], k=1)[0][-1]
            bgn_topk = torch.topk(bert_output[j, patch_bgn], k=2)[0][-1]
            end_top1 = torch.topk(bert_output[j, patch_end], k=1)[0][-1]
            end_topk = torch.topk(bert_output[j, patch_end], k=2)[0][-1]
            if bgn_top1 - bgn_topk < e:
                blin_output[j, patch_bgn] = blin2i_output[j, patch_bgn]

            if end_top1 - end_topk < e:
                blin_output[j, patch_end] = blin2i_output[j, patch_end]

            mask[j, patch_bgn] = False
            mask[j, patch_end] = False

    return blin_output

def test():
    second = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = 202
    heads = 2
    transformer = model.Transformer(src_vocab_size=K + heads, embed_size=K + heads, heads=heads, num_layers=2,
                                    forward_expansion=heads, max_length=DIM_INPUT, device=device).to(device)
    if config.USE_CENTRAL_MASK:
        transformer.load_state_dict(torch.load('../checkpoint/0419_feeder2000_200_global_local_mse_mid_NewRiver_all_step96/encoder_100.pth'))
    else:
        transformer.load_state_dict(torch.load('../checkpoint/0417_feeder2000_200_global_local_peak_mse_NewRiver_all_step96/encoder_100.pth'))

    bert_mpe_buff = []
    bert_rmse_buff = []
    bert_pk_err_buff = []
    bert_vl_err_buff = []
    bert_egy_err_buff = []
    bert_fce_buff = []

    if second:
        blin2_mpe_buff = []
        blin2_rmse_buff = []
        blin2_pk_err_buff = []
        blin2_vl_err_buff = []
        blin2_egy_err_buff = []
        blin2_fce_buff = []

        blin2i_mpe_buff = []
        blin2i_rmse_buff = []
        blin2i_pk_err_buff = []
        blin2i_vl_err_buff = []
        blin2i_egy_err_buff = []
        blin2i_fce_buff = []

        blin2add_mpe_buff = []
        blin2add_rmse_buff = []
        blin2add_pk_err_buff = []
        blin2add_vl_err_buff = []
        blin2add_egy_err_buff = []
        blin2add_fce_buff = []

    top2_ite = []
    top2_reconciliation = []
    reconciliation_parameters = [0.4, 0.3, 0.2, 0.1, 0.01, 0.0]
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
        blin_output = bert_output.argmax(dim=-1)
        blin_output = blin_output / K
        blin_output = model_input[:, :] + blin_output * mask

        if second:
            k = 2
            blin2_output = torch.topk(bert_output, k=k, dim=-1)[1]    # torch.return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))
            blin2_output = blin2_output[:, :, -1]
            blin2_output = blin2_output / K
            blin2_output = model_input[:, :] + blin2_output * mask
            # blin2i_output = blin_2nd_iter(transformer, bert_input.clone(), temperature.clone(), mask.clone(), k=k)
            # blin2i_output = blin_2nd_iter_improved(transformer, bert_input.clone(), temperature.clone(), mask.clone(), k=k, e=0.5)
            blin2i_output = blin_2nd_iter_noniter(transformer, bert_input.clone(), temperature.clone(), mask.clone(),
                                                   k=k, e=0.8)
            blin2i_output = blin2i_output / K
            blin2i_output = model_input[:, :] + blin2i_output * mask
            blin2_add_output = blin_combine(blin_output.clone(), blin2i_output.clone(), gt.clone())
            # blin2_add_output = reconciliation(bert_output.clone(), blin_output.clone(), blin2i_output.clone(), mask.clone(), e=reconciliation_parameters[0])
            reconciliation_1 = reconciliation(bert_output.clone(), blin_output.clone(), blin2i_output.clone(), mask.clone(), e=reconciliation_parameters[1])
            reconciliation_2 = reconciliation(bert_output.clone(), blin_output.clone(), blin2i_output.clone(),
                                                   mask.clone(), e=reconciliation_parameters[2])
            reconciliation_3 = reconciliation(bert_output.clone(), blin_output.clone(), blin2i_output.clone(),
                                                   mask.clone(), e=reconciliation_parameters[3])
            reconciliation_4 = reconciliation(bert_output.clone(), blin_output.clone(), blin2i_output.clone(),
                                                   mask.clone(), e=reconciliation_parameters[4])
            reconciliation_5 = reconciliation(bert_output.clone(), blin_output.clone(), blin2i_output.clone(),
                                                  mask.clone(), e=reconciliation_parameters[5])

        pre_np_bert = blin_output.cpu().detach().numpy()
        if second:
            pre_np_blin2 = blin2_output.cpu().detach().numpy()
            pre_np_blin2i = blin2i_output.cpu().detach().numpy()
            pre_np_add = blin2_add_output.cpu().detach().numpy()
            reconciliation_1 = reconciliation_1.cpu().detach().numpy()
            reconciliation_2 = reconciliation_2.cpu().detach().numpy()
            reconciliation_3 = reconciliation_3.cpu().detach().numpy()
            reconciliation_4 = reconciliation_4.cpu().detach().numpy()
            reconciliation_5 = reconciliation_5.cpu().detach().numpy()

        input_np = model_input.cpu().detach().numpy()
        gt_np = gt.cpu().detach().numpy()

        fig = plt.figure(1, figsize=(2 * 4, 2 * 2))
        plt.clf()
        gs = fig.add_gridspec(2, 2)
        for j in range(4):
            buff = np.concatenate((pre_np_bert[j, :], gt_np[j, :]))

            if second:
                buff = np.concatenate((buff, pre_np_blin2[j, :], pre_np_blin2i[j, :]))

            patch_bgn = np.where(mask_np[j, :] == 1.0)[0][0] - 1
            patch_end = np.where(mask_np[j, :] == 1.0)[0][-1] + 1
            x = np.arange(patch_bgn, patch_end + 1)
            y_min = np.amin(buff)
            y_max = np.amax(buff)

            ax = fig.add_subplot(gs[j % 2, j // 2])
            # draw input data
            ax.plot(np.arange(0, patch_bgn + 1), input_np[j, 0:patch_bgn + 1], 'k', linewidth=1)
            ax.plot(np.arange(patch_end, np.size(input_np, axis=1)), input_np[j, patch_end:], 'k', linewidth=1)

            if second:
                ax.plot(x, pre_np_blin2[j, patch_bgn:patch_end + 1], 'r--', linewidth=1, label='BERT-PIN_2')
                ax.plot(x, pre_np_blin2i[j, patch_bgn:patch_end + 1], 'r-.', linewidth=1, label='BERT-PIN_2i')

            # draw output data
            ax.plot(x, pre_np_bert[j, patch_bgn:patch_end + 1], 'r', linewidth=1, label='BERT-PIN')
            # draw gt
            ax.plot(x, gt_np[j, patch_bgn:patch_end + 1], 'g', linewidth=1, label='GT')
            plt.ylim(y_min, y_max)
            plt.xticks([])
            plt.yticks([])
            rect = plt.Rectangle((patch_bgn, y_min), patch_end - patch_bgn + 1, y_max - y_min,
                                 facecolor="k", alpha=0.1)
            ax.add_patch(rect)
            ax.legend()

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.pause(0.001)  # pause a bit so that plots are updated

        fn = '../plot/' + TAG + '/samples_pre' + str(i) + '.png'
        fig.savefig(fn, dpi=300)

        # calculate errs
        for idx in range(bs):
            patch_bgn = np.where(mask_np[idx, :] == 1.0)[0][0]
            patch_end = np.where(mask_np[idx, :] == 1.0)[0][-1] + 1

            rmse_bert = mean_squared_error(pre_np_bert[idx, patch_bgn:patch_end], gt_np[idx, patch_bgn:patch_end]) ** 0.5

            # percentage error: (gt - pre) / gt
            pe_bert = abs(gt_np[idx, patch_bgn:patch_end] - pre_np_bert[idx, patch_bgn:patch_end]) / gt_np[idx, patch_bgn:patch_end]

            mpe_bert = np.mean(pe_bert)

            # energy error
            egy_err_bert = abs(np.sum(pre_np_bert[idx, patch_bgn:patch_end]) - np.sum(gt_np[idx, patch_bgn:patch_end])) / np.sum(
                gt_np[idx, patch_bgn:patch_end])


            # peak error, valley error, peak-valley error
            pk_err_bert, vl_err_bert, pkvy_err_bert = evaluation.peak_valley_err(pre_np_bert[idx, patch_bgn:patch_end], gt_np[idx, patch_bgn:patch_end])

            # frequency components error
            f_pre_bert = abs(fft(pre_np_bert[idx, patch_bgn:patch_end]))

            f_gt = abs(fft(gt_np[idx, patch_bgn:patch_end]))
            fft_err_bert = np.sum(abs(f_gt - f_pre_bert)) / (patch_end - patch_bgn + 1)

            bert_mpe_buff.append(mpe_bert)
            bert_rmse_buff.append(rmse_bert)
            bert_pk_err_buff.append(pk_err_bert)
            bert_vl_err_buff.append(vl_err_bert)
            bert_egy_err_buff.append(egy_err_bert)
            bert_fce_buff.append(fft_err_bert)

            if second:
                rmse_2 = mean_squared_error(pre_np_blin2[idx, patch_bgn:patch_end],
                                              gt_np[idx, patch_bgn:patch_end]) ** 0.5
                rmse_2i = mean_squared_error(pre_np_blin2i[idx, patch_bgn:patch_end],
                                               gt_np[idx, patch_bgn:patch_end]) ** 0.5

                count_ite = 0
                count_reconciliation = [0, 0, 0, 0, 0, 0]
                for index in range(patch_bgn, patch_end):
                    base_distance = abs(pre_np_bert[idx, index] - gt_np[idx, index])
                    # distance_ite = abs(pre_np_blin2[idx, index] - gt_np[idx, index])
                    distance_ite = abs(pre_np_blin2i[idx, index] - gt_np[idx, index])
                    distance_reconciliation = [abs(pre_np_add[idx, index] - gt_np[idx, index]),
                                               abs(reconciliation_1[idx, index] - gt_np[idx, index]),
                                               abs(reconciliation_2[idx, index] - gt_np[idx, index]),
                                               abs(reconciliation_3[idx, index] - gt_np[idx, index]),
                                               abs(reconciliation_4[idx, index] - gt_np[idx, index]),
                                               abs(reconciliation_5[idx, index] - gt_np[idx, index])]
                    if distance_ite < base_distance:
                        count_ite += 1
                    for par in range(6):
                        if distance_reconciliation[par] < base_distance:
                            count_reconciliation[par] += 1

                top2_ite.append(count_ite / 0.16)
                top2_reconciliation.append([item / 0.16 for item in count_reconciliation])

                pe_2 = abs(gt_np[idx, patch_bgn:patch_end] - pre_np_blin2[idx, patch_bgn:patch_end]) / gt_np[idx,
                                                                                                       patch_bgn:patch_end]
                pe_2i = abs(gt_np[idx, patch_bgn:patch_end] - pre_np_blin2i[idx, patch_bgn:patch_end]) / gt_np[idx,
                                                                                                         patch_bgn:patch_end]
                mpe_2 = np.mean(pe_2)
                mpe_2i = np.mean(pe_2i)
                egy_err_2 = abs(
                    np.sum(pre_np_blin2[idx, patch_bgn:patch_end]) - np.sum(gt_np[idx, patch_bgn:patch_end])) / np.sum(
                    gt_np[idx, patch_bgn:patch_end])
                egy_err_2i = abs(
                    np.sum(pre_np_blin2i[idx, patch_bgn:patch_end]) - np.sum(gt_np[idx, patch_bgn:patch_end])) / np.sum(
                    gt_np[idx, patch_bgn:patch_end])
                pk_err_2, vl_err_2, pkvy_err_2 = evaluation.peak_valley_err(pre_np_blin2[idx, patch_bgn:patch_end],
                                                                                  gt_np[idx, patch_bgn:patch_end])
                pk_err_2i, vl_err_2i, pkvy_err_2i = evaluation.peak_valley_err(
                    pre_np_blin2i[idx, patch_bgn:patch_end],
                    gt_np[idx, patch_bgn:patch_end])
                f_pre_2 = abs(fft(pre_np_blin2[idx, patch_bgn:patch_end]))
                f_pre_2i = abs(fft(pre_np_blin2i[idx, patch_bgn:patch_end]))
                fft_err_2 = np.sum(abs(f_gt - f_pre_2)) / (patch_end - patch_bgn + 1)
                fft_err_2i = np.sum(abs(f_gt - f_pre_2i)) / (patch_end - patch_bgn + 1)
                blin2_mpe_buff.append(mpe_2)
                blin2_rmse_buff.append(rmse_2)
                blin2_pk_err_buff.append(pk_err_2)
                blin2_vl_err_buff.append(vl_err_2)
                blin2_egy_err_buff.append(egy_err_2)
                blin2_fce_buff.append(fft_err_2)
                blin2i_mpe_buff.append(mpe_2i)
                blin2i_rmse_buff.append(rmse_2i)
                blin2i_pk_err_buff.append(pk_err_2i)
                blin2i_vl_err_buff.append(vl_err_2i)
                blin2i_egy_err_buff.append(egy_err_2i)
                blin2i_fce_buff.append(fft_err_2i)

                rmse_add = mean_squared_error(pre_np_add[idx, patch_bgn:patch_end],
                                             gt_np[idx, patch_bgn:patch_end]) ** 0.5

                pe_add = abs(gt_np[idx, patch_bgn:patch_end] - pre_np_add[idx, patch_bgn:patch_end]) / gt_np[idx,
                                                                                                       patch_bgn:patch_end]
                mpe_add = np.mean(pe_add)
                egy_err_add = abs(
                    np.sum(pre_np_add[idx, patch_bgn:patch_end]) - np.sum(gt_np[idx, patch_bgn:patch_end])) / np.sum(
                    gt_np[idx, patch_bgn:patch_end])

                pk_err_add, vl_err_add, pkvy_err_add = evaluation.peak_valley_err(pre_np_add[idx, patch_bgn:patch_end],
                                                                            gt_np[idx, patch_bgn:patch_end])
                f_pre_add = abs(fft(pre_np_add[idx, patch_bgn:patch_end]))

                fft_err_add = np.sum(abs(f_gt - f_pre_add)) / (patch_end - patch_bgn + 1)

                blin2add_mpe_buff.append(mpe_add)
                blin2add_rmse_buff.append(rmse_add)
                blin2add_pk_err_buff.append(pk_err_add)
                blin2add_vl_err_buff.append(vl_err_add)
                blin2add_egy_err_buff.append(egy_err_add)
                blin2add_fce_buff.append(fft_err_add)

        if i > 20:
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

    print(np.mean(bert_mpe_buff), np.mean(bert_rmse_buff), np.mean(bert_pk_err_buff), np.mean(bert_vl_err_buff), np.mean(bert_egy_err_buff), np.mean(bert_fce_buff))

    if second:
        print(np.mean(blin2_mpe_buff), np.mean(blin2i_mpe_buff), '\n',
              np.mean(blin2_rmse_buff), np.mean(blin2i_rmse_buff), '\n',
              np.mean(blin2_pk_err_buff), np.mean(blin2i_pk_err_buff), '\n',
              np.mean(blin2_vl_err_buff), np.mean(blin2i_vl_err_buff), '\n',
              np.mean(blin2_egy_err_buff), np.mean(blin2i_egy_err_buff), '\n',
              np.mean(blin2_fce_buff), np.mean(blin2i_fce_buff))

        print("error of add")
        print(np.mean(blin2add_mpe_buff), '\n',
              np.mean(blin2add_rmse_buff), '\n',
              np.mean(blin2add_pk_err_buff), '\n',
              np.mean(blin2add_vl_err_buff), '\n',
              np.mean(blin2add_egy_err_buff), '\n',
              np.mean(blin2add_fce_buff))

        figure = plt.figure(3)
        top2_reconciliation = np.transpose(top2_reconciliation)
        violin_parts = plt.violinplot([top2_ite,
                                       top2_reconciliation[0],
                                       top2_reconciliation[1],
                                       top2_reconciliation[2],
                                       top2_reconciliation[3],
                                       top2_reconciliation[4],
                                       top2_reconciliation[5]], showmeans=True, showextrema=False)
        plt.xticks(range(1, 8), labels=['max',
                                        str(reconciliation_parameters[0]),
                                        str(reconciliation_parameters[1]),
                                        str(reconciliation_parameters[2]),
                                        str(reconciliation_parameters[3]),
                                        str(reconciliation_parameters[4]),
                                        str(reconciliation_parameters[5])])
        plt.xlabel("threshold")
        plt.ylabel("PoCP (%)")
        plt.show()
        fn = '../plot/' + TAG + '/reconciliation.png'
        figure.savefig(fn, dpi=300)
        print("percentage || ite: ", np.mean(top2_ite), " reconciliations: ", np.mean(top2_reconciliation, axis=1))


if __name__ == "__main__":
    test()



