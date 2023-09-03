#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  26 2023

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
    mask_np = mask.cpu().detach().numpy()

    bs = gt.size(0)
    local_entropy_loss = 0
    for j in range(bs):
        if mask is not None:
            # pre = pre.masked_fill(mask == False, float(0.0))
            # gt = gt.masked_fill(mask == False, float(0.0))
            patch_bgn = np.where(mask_np[0, :] == 1.0)[0][0] - 1
            patch_end = np.where(mask_np[0, :] == 1.0)[0][-1] + 1
        loss_1 = criterion(pre[j, patch_bgn:patch_end+1], gt[j, patch_bgn:patch_end+1])
        local_entropy_loss += loss_1
    local_entropy_loss = local_entropy_loss / bs

    global_entropy_loss = 0
    for j in range(bs):
        loss_1 = criterion(pre[j], gt[j])
        global_entropy_loss += loss_1
    global_entropy_loss = global_entropy_loss / bs

    pre = pre.argmax(dim=-1)
    pre = pre / K
    gt = gt / K

    patch_pre = pre[:, patch_bgn:patch_end+1]
    patch_gt = gt[:, patch_bgn:patch_end+1]
    mse = nn.MSELoss(reduction='mean')
    mse_loss = mse(patch_pre, patch_gt)

    loss = global_entropy_loss + 5 * local_entropy_loss # + 5 * mse_loss
    return loss

def train_bert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    K = 202
    heads = 2
    transformer = model.Transformer(src_vocab_size=K+heads, embed_size=K+heads, heads=heads, num_layers=2,
                                    forward_expansion=heads, max_length=config.DIM_INPUT, device=device).to(device)
    if config.USE_CENTRAL_MASK:
        transformer.load_state_dict(
            torch.load('../checkpoint/0417_feeder2000_200_global_local_peak_mse_NewRiver_all_step96/encoder_100.pth'))
    else:
        transformer.load_state_dict(
            torch.load('../checkpoint/0419_feeder2000_200_global_local_mse_mid_NewRiver_all_step96/encoder_100.pth'))
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

            # pre, pre_coarse,_ = models[j](model_input)
            output = transformer(model_input, temperature, mask)
            # loss = 0
            # for j in range(bs):
            #     loss_1 = criterion(output[j], gt[j])
            #     loss += loss_1
            # # loss_2 = criterion(output, gt1)

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
            # print("epoch ", epoch, " step ", i, "/", len(trainloader), " ====== train loss ", loss.item(), " eval loss ", eval_loss, t)

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

def test():
    second = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = 202
    heads = 2
    transformer = model.Transformer(src_vocab_size=K + heads, embed_size=K + heads, heads=heads, num_layers=2,
                                    forward_expansion=heads, max_length=DIM_INPUT, device=device).to(device)
    transformer_pretrain = model.Transformer(src_vocab_size=K + heads, embed_size=K + heads, heads=heads, num_layers=2,
                                    forward_expansion=heads, max_length=DIM_INPUT, device=device).to(device)

    if config.USE_CENTRAL_MASK:
        transformer.load_state_dict(
            torch.load('../checkpoint/0419_feeder2000_200_global_local_mse_mid_NewRiver_all_step96/encoder_100.pth'))
        transformer_pretrain.load_state_dict(
            torch.load('../checkpoint/0629_pretrained_central_NewRiver_all_step96/encoder_20.pth'))
    else:
        transformer.load_state_dict(
            torch.load('../checkpoint/0417_feeder2000_200_global_local_peak_mse_NewRiver_all_step96/encoder_100.pth'))
        transformer_pretrain.load_state_dict(
            torch.load('../checkpoint/0629_pretrained_NewRiver_all_step96/encoder_20.pth'))

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
            blin2_output = transformer_pretrain(bert_input, temperature, mask)
            blin2_output = blin2_output.argmax(dim=-1)
            blin2_output = blin2_output / K
            blin2_output = model_input[:, :] + blin2_output * mask

        pre_np_bert = blin_output.cpu().detach().numpy()
        if second:
            pre_np_blin2 = blin2_output.cpu().detach().numpy()

        input_np = model_input.cpu().detach().numpy()
        gt_np = gt.cpu().detach().numpy()

        fig = plt.figure(1, figsize=(2 * 4, 2 * 2))
        plt.clf()
        gs = fig.add_gridspec(2, 2)
        for j in range(4):
            buff = np.concatenate((pre_np_bert[j, :], gt_np[j, :]))
            if second:
                buff = np.concatenate((buff, pre_np_blin2[j, :]))

            patch_bgn = np.where(mask_np[j, :] == 1.0)[0][0] - 1
            patch_end = np.where(mask_np[j, :] == 1.0)[0][-1] + 1
            x = np.arange(patch_bgn, patch_end + 1)
            y_min = np.amin(buff)
            y_max = np.amax(buff)

            ax = fig.add_subplot(gs[j % 2, j // 2])
            # draw input data
            ax.plot(np.arange(0, patch_bgn + 1), input_np[j, 0:patch_bgn + 1], 'k', linewidth=1)
            ax.plot(np.arange(patch_end, np.size(input_np, axis=1)), input_np[j, patch_end:], 'k', linewidth=1)

            # draw output data
            ax.plot(x, pre_np_bert[j, patch_bgn:patch_end + 1], 'r', linewidth=1, label='BLIN')
            if second:
                ax.plot(x, pre_np_blin2[j, patch_bgn:patch_end + 1], 'b', linewidth=1, label='BLIN_PRE')

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

                pe_2 = abs(gt_np[idx, patch_bgn:patch_end] - pre_np_blin2[idx, patch_bgn:patch_end]) / gt_np[idx,
                                                                                                       patch_bgn:patch_end]
                mpe_2 = np.mean(pe_2)
                egy_err_2 = abs(
                    np.sum(pre_np_blin2[idx, patch_bgn:patch_end]) - np.sum(gt_np[idx, patch_bgn:patch_end])) / np.sum(
                    gt_np[idx, patch_bgn:patch_end])
                pk_err_2, vl_err_2, pkvy_err_2 = evaluation.peak_valley_err(pre_np_blin2[idx, patch_bgn:patch_end],
                                                                                  gt_np[idx, patch_bgn:patch_end])
                f_pre_2 = abs(fft(pre_np_blin2[idx, patch_bgn:patch_end]))
                fft_err_2 = np.sum(abs(f_gt - f_pre_2)) / (patch_end - patch_bgn + 1)
                blin2_mpe_buff.append(mpe_2)
                blin2_rmse_buff.append(rmse_2)
                blin2_pk_err_buff.append(pk_err_2)
                blin2_vl_err_buff.append(vl_err_2)
                blin2_egy_err_buff.append(egy_err_2)
                blin2_fce_buff.append(fft_err_2)

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

    print(np.mean(bert_mpe_buff), np.mean(bert_rmse_buff), np.mean(bert_pk_err_buff), np.mean(bert_vl_err_buff), np.mean(bert_egy_err_buff), np.mean(bert_fce_buff))

    if second:
        print(np.mean(blin2_mpe_buff), np.mean(blin2_rmse_buff), np.mean(blin2_pk_err_buff),
              np.mean(blin2_vl_err_buff), np.mean(blin2_egy_err_buff), np.mean(blin2_fce_buff))



if __name__ == "__main__":
    train_bert()
    # test()



