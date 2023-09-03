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
import diffusion
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
    if config.USE_CENTRAL_MASK:
        transformer.load_state_dict(
            torch.load('../checkpoint/0615_BLIN_multipatch_peak_NewRiver_all_step672/encoder_50.pth'))
    else:
        transformer.load_state_dict(
            torch.load('../checkpoint/0615_BLIN_multipatch_central_NewRiver_all_step672/encoder_30.pth'))
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
    transformer_pretrain = model.Transformer(src_vocab_size=K + heads, embed_size=K + heads, heads=heads, num_layers=2,
                                             forward_expansion=heads, max_length=DIM_INPUT, device=device).to(device)
    if config.USE_CENTRAL_MASK:
        transformer.load_state_dict(
            torch.load('../checkpoint/0615_BLIN_multipatch_central_NewRiver_all_step672/encoder_30.pth'))
        transformer_pretrain.load_state_dict(torch.load('../checkpoint/0629_multipatch_pretrain_NewRiver_all_step672/encoder_10.pth'))
    else:
        transformer.load_state_dict(
            torch.load('../checkpoint/0615_BLIN_multipatch_peak_NewRiver_all_step672/encoder_50.pth'))
        transformer_pretrain.load_state_dict(
            torch.load('../checkpoint/0629_multipatch_pretrain_peak_NewRiver_all_step672/encoder_10.pth'))
    bert_mpe_buff = []
    bert_rmse_buff = []
    bert_pk_err_buff = []
    bert_vl_err_buff = []
    bert_egy_err_buff = []
    bert_fce_buff = []
    # pretrained error buff
    sae_mpe_buff = []
    sae_rmse_buff = []
    sae_pk_err_buff = []
    sae_vl_err_buff = []
    sae_egy_err_buff = []
    sae_fce_buff = []

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

        bert2_output = transformer_pretrain(bert_input, temperature, mask)
        blin2_output = bert2_output.argmax(dim=-1)
        blin2_output = blin2_output / K
        blin2_output = model_input[:, :] + blin2_output * mask

        pre_np_bert = blin_output.cpu().detach().numpy()
        pre_np_bert2 = blin2_output.cpu().detach().numpy()
        input_np = model_input.cpu().detach().numpy()
        gt_np = gt.cpu().detach().numpy()

        fig = plt.figure(1, figsize=(3 * 3, 2 * 3))
        plt.clf()
        gs = fig.add_gridspec(3, 1)
        for j in range(3):
            ax = fig.add_subplot(gs[j, 0])
            buff = np.concatenate((pre_np_bert[j, :], pre_np_bert2[j, :], gt_np[j, :]))

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
                    ax.plot(x, pre_np_bert[j, patch_bgn:patch_end + 1], 'r', linewidth=1, label='BLIN')
                    ax.plot(x, pre_np_bert2[j, patch_bgn:patch_end + 1], 'b', linewidth=1, label='BLIN_PRE')
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

                # pretrained model error
                pre_day_sae = pre_np_bert2[idx, n * dim_day: (n + 1) * dim_day]

                rmse_sae += mean_squared_error(pre_day_sae[patch_bgn:patch_end],
                                              gt_day[patch_bgn:patch_end]) ** 0.5

                pe_sae = abs(gt_day[patch_bgn:patch_end] - pre_day_sae[patch_bgn:patch_end]) / gt_day[patch_bgn:patch_end]

                mpe_sae += np.mean(pe_sae)

                egy_err_sae += abs(
                    np.sum(pre_day_sae[patch_bgn:patch_end]) - np.sum(
                        gt_day[patch_bgn:patch_end])) / np.sum(
                    gt_day[patch_bgn:patch_end])

                pk_err_sae_, vl_err_sae_, pkvy_err_sae = evaluation.peak_valley_err(
                    pre_day_sae[patch_bgn:patch_end],
                    gt_day[patch_bgn:patch_end])

                pk_err_sae += pk_err_sae_
                vl_err_sae += vl_err_sae_

                f_pre_sae = abs(fft(pre_day_sae[patch_bgn:patch_end]))

                fft_err_sae += np.sum(abs(f_gt - f_pre_sae)) / (patch_end - patch_bgn + 1)


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

    print(np.mean(bert_mpe_buff),
              np.mean(bert_rmse_buff),
              np.mean(bert_pk_err_buff),
              np.mean(bert_vl_err_buff),
              np.mean(bert_egy_err_buff),
              np.mean(bert_fce_buff))

    print(np.mean(sae_mpe_buff), np.mean(sae_rmse_buff), np.mean(sae_pk_err_buff), np.mean(sae_vl_err_buff), np.mean(sae_egy_err_buff), np.mean(sae_fce_buff))


if __name__ == "__main__":
    # train_bert()
    test()





