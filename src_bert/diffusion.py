'''
Describe: Build defusion model and train it.
Author: Yi Hu
Email: yhu28@ncsu.edu
'''

import os
import glob
import numpy as np
import torch.nn as nn
import torch
import random
import time
import config
from config import *
from dataset import testloader, trainloader, devloader, cvrloader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import evaluation

class Denoise(nn.Module):
    def __init__(self, dim=32):
        super(Denoise, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        # input image size: [1, 96*7, 8]

        self.cnn_layers = nn.Sequential(
            # 96
            nn.Conv1d(1, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            # 96
            nn.Conv1d(dim, 2*dim, 3, 3, 0),
            nn.BatchNorm1d(2*dim),
            nn.ReLU(),
            # 32
            nn.Conv1d(2 * dim, 4 * dim, 5, 2, 2),
            nn.BatchNorm1d(4 * dim),
            nn.ReLU(),
            # 16
            nn.ConvTranspose1d(4 * dim, 2 * dim, 5, 2, 2, 1),
            nn.BatchNorm1d(2 * dim),
            nn.ReLU(),
            # 32
            nn.ConvTranspose1d(2*dim, dim, 3, 3, 0),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            # 96
            nn.ConvTranspose1d(dim, 1, 3, 1, 1),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            # 96
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(96, 96 * 2),
            nn.ReLU(),
            nn.Linear(96 * 2, 96)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        x = x.reshape((x.shape[0], x.shape[2]))
        x = self.fc_layers(x)

        return x



def train_defusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    model = Denoise().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, betas=(0.9, 0.99))
    model.train()
    criterion = torch.nn.L1Loss(reduction='sum')

    # ------------------------------------Training------------------------------------
    start_t = time.strftime("%Y/%m/%d %H:%M:%S")
    loss_train_rec = []
    loss_eval_rec = []

    print("start train Diffusion Model.")
    for epoch in range(config.N_EPOCH):
        train_loss_list = []
        for i, data in enumerate(trainloader):
            model.train()
            _, temperature, mask, gt = data

            gt = gt.reshape((gt.shape[0], gt.shape[2])).to(device)
            bs = gt.size(0)

            # noise = np.random.uniform(-0.05, 0.05, size=(bs, 96))
            noise = np.random.normal(0, 0.02, size=(bs, 96))
            noise = torch.tensor(noise, dtype=torch.float).to(device)
            model_input = gt + noise

            output = model(model_input)

            loss = criterion(output, noise)
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
            torch.save(model.state_dict(),
                       '../checkpoint/' + config.TAG + '/diffusion_' + str(epoch) + '.pth')

        train_loss = np.mean(train_loss_list)
        eval_loss = eval_set(model, epoch)
        loss_train_rec.append(train_loss)
        loss_eval_rec.append(eval_loss)

        t = time.strftime("%Y/%m/%d %H:%M:%S")
        print("epoch ", epoch, " ====== train loss ", train_loss, " eval loss ", eval_loss, t)

        evaluation.plot_loss(loss_train_rec, loss_eval_rec, epoch, 'DIFF')
        epoch += 1

    TRAINLOSS_PTH = '../eval/'+ config.TAG + '/DIFF_train_loss.npy'
    np.save(TRAINLOSS_PTH, loss_train_rec)
    EVALLOSS_PTH = '../eval/' + config.TAG + '/DIFF_eval_loss.npy'
    np.save(EVALLOSS_PTH, loss_eval_rec)

def eval_set(model, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    criterion = torch.nn.L1Loss(reduction='sum')

    train_loss_list = []
    for i, data in enumerate(devloader):
        model.train()
        _, temperature, mask, gt = data

        gt = gt.reshape((gt.shape[0], gt.shape[2])).to(device)
        bs = gt.size(0)

        # noise = np.random.uniform(-0.05, 0.05, size=(bs, 96))
        noise = np.random.normal(0, 0.02, size=(bs, 96))
        noise = torch.tensor(noise, dtype=torch.float).to(device)
        model_input = gt + noise

        output = model(model_input)
        loss = criterion(output, noise)
        train_loss_list.append(loss.item())

        est = model_input - output

    ave_loss = np.mean(train_loss_list)

    gt_np = gt.cpu().detach().numpy()
    noise_np = noise.cpu().detach().numpy()
    model_input = model_input.cpu().detach().numpy()
    output_np = output.cpu().detach().numpy()
    est_np = est.cpu().detach().numpy()
    fig = plt.figure(1, figsize=(10, 20))
    plt.clf()
    gs = fig.add_gridspec(5, 1)
    for i in range(0, 5):
        ax = fig.add_subplot(gs[i, 0])
        buff = np.concatenate((gt_np[i, :], noise_np[i, :], model_input[i, :], output_np[i, :], est_np[i, :]))
        y_min = np.amin(buff)
        y_max = np.amax(buff)
        ax.plot(gt_np[i, :], 'g', linewidth=1, label="gt")
        ax.plot(noise_np[i, :], 'b', linewidth=1, label="noise")
        ax.plot(model_input[i, :], 'y', linewidth=1, label="in")
        ax.plot(output_np[i, :], 'r', linewidth=1, label="out")
        ax.plot(est_np[i, :], 'k', linewidth=1, label="est")
        plt.ylim(y_min, y_max)
        plt.xticks([])
        plt.yticks([])
        plt.legend()

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.pause(0.001)  # pause a bit so that plots are updated

    if epoch % 10 == 0:
        fn = '../plot/' + TAG + '/diffusion_results' + str(epoch) + '.png'
        fig.savefig(fn)

    model.train()
    return ave_loss

if __name__ == "__main__":
    train_defusion()