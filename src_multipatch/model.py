#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  26 2023

@author: Yi Hu
"""

import torch
import torch.nn as nn
import config
import os
from network_module import GatedConv1dWithActivation, GatedDeConv1dWithActivation, SNConvWithActivation

class MultiheadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiheadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)    # (N, value_len, embed_size)
        keys = self.keys(keys)          # (N, key_len, embed_size)
        queries = self.queries(query)   # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        alpha = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # alpha: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            alpha = alpha.masked_fill(mask == True, float("-1e20"))

        # Normalize alpha values
        # attention shape: (N, heads, query_len, key_len)
        attention = torch.softmax(alpha / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim),
        # then reshape and flatten the last two dimensions.

        out = self.fc_out(out)  # (N, query_len, embed_size)

        return out


class TransformerLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerLayer, self).__init__()
        self.attention = MultiheadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
        device,
    ):

        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.temperature_embedding = nn.Embedding(src_vocab_size, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, embed_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, temperature, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions) + self.temperature_embedding(temperature))
        ).to(self.device)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = self.fc_out(out)
        out = self.logsoftmax(out)

        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size=4000,
        embed_size=5000,
        num_layers=4,
        device="cup",
        forward_expansion=4,
        heads=8,
        dropout=0,
        max_length=96,
    ):

        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
            device
        )

        self.device = device
        self.name = 'BERT'

    def make_src_mask(self, mask):
        src_mask = mask.unsqueeze(1).unsqueeze(2).to(self.device)
        # (N, 1, 1, src_len)
        return src_mask

    def forward(self, src, temperature, mask):
        # mask for patched load  False-True-False
        src_mask = self.make_src_mask(mask).to(self.device)
        enc_src = self.encoder(src, temperature, src_mask)

        return enc_src


class SAE(nn.Module):
    def __init__(self, name='SAE'):
        super().__init__()
        self.name = name
        self.num_fea = 128
        self.sae = nn.Sequential(
            nn.Linear(config.DIM_INPUT, self.num_fea * 4),
            nn.ReLU(),
            nn.Linear(self.num_fea * 4, self.num_fea * 2),
            nn.ReLU(),
            nn.Linear(self.num_fea * 2, self.num_fea * 1),
            nn.ReLU(),
            nn.Linear(self.num_fea * 1, self.num_fea * 2),
            nn.ReLU(),
            nn.Linear(self.num_fea * 2, self.num_fea * 4),
            nn.ReLU(),
            nn.Linear(self.num_fea * 4, config.DIM_INPUT),
        )

    def forward(self, x):
        x = self.sae(x)
        # x = x.unsqueeze(1)
        return x

    def save_checkpoint(self, epoch):
        if epoch % config.SAVE_PER_EPO == 0:
            filename = os.path.join("../checkpoint/" + config.TAG +
                                    '/' + self.name + '_epoch' + str(epoch) + '.pth')
            torch.save(self.state_dict(), filename)

    def save_best_checkpoint(self):
        filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_best.pth')
        torch.save(self.state_dict(), filename)


class LSTM(nn.Module):
    def __init__(self, in_channels=3, name='LSTM'):
        super().__init__()
        self.num_layers = 1
        self.hidden_size = 128
        self.out_dim = config.DIM_INPUT
        self.seq_len = config.DIM_INPUT
        self.in_channels = in_channels
        self.drop_rate = 0.0

        self.lstm = nn.LSTM(self.in_channels, self.hidden_size, self.num_layers,
                            batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * self.seq_len, self.out_dim),
            nn.Dropout(p=self.drop_rate)
        )

        self.name = name
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x, mask, temperature):
        bs = x.size(0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = x.unsqueeze(1)
        mask = mask.unsqueeze(1)
        temperature = temperature.unsqueeze(1)
        x = torch.concat([x, mask, temperature], dim=1)
        x = x.reshape(bs, self.out_dim, self.in_channels)
        x, _ = self.lstm(x, (h0, c0))
        x = x.reshape(x.shape[0], -1)
        out = self.fc(x)
        # out = torch.unsqueeze(x, 1)
        return out

    def save_checkpoint(self, epoch):
        if epoch % config.SAVE_PER_EPO == 0:
            filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_epoch' + str(epoch) + '.pth')
            torch.save(self.state_dict(), filename)

    def save_best_checkpoint(self):
        filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_best.pth')
        torch.save(self.state_dict(), filename)

class Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation=None):
        super(Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize,-1,width).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width) # B X C x (*W*H)
        energy = torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width)

        return out, proj_value, attention
class GenGIN(torch.nn.Module):
    """
    Generator of Generative Infilling Net with Multi-head Attention
    """

    def __init__(self, in_ch=3, n_fea=64, name='GIN'):
        super(GenGIN, self).__init__()
        self.cor_gc01 = GatedConv1dWithActivation(in_ch, n_fea, 5, 1, 2)
        self.cor_gc02 = GatedConv1dWithActivation(n_fea, 2 * n_fea, 4, 2, 1)  # //2
        self.cor_gc03 = GatedConv1dWithActivation(2 * n_fea, 2 * n_fea, 3, 1, 1)
        self.cor_gc04 = GatedConv1dWithActivation(2 * n_fea, 4 * n_fea, 4, 2, 1)  # //2
        self.cor_gc05 = GatedConv1dWithActivation(4 * n_fea, 4 * n_fea, 3, 1, 1)
        # upsample
        self.cor_gdc1 = GatedDeConv1dWithActivation(2, 4 * n_fea, 2 * n_fea, 3, 1, 1)
        self.cor_gc09 = GatedConv1dWithActivation(2 * n_fea, 2 * n_fea, 3, 1, 1)
        self.cor_gdc2 = GatedDeConv1dWithActivation(2, 2 * n_fea, n_fea, 3, 1, 1)
        self.cor_gc11 = GatedConv1dWithActivation(n_fea, 1, 3, 1, 1, activation=None)

        self.rf1_gc01 = GatedConv1dWithActivation(in_ch, n_fea, 5, 1, 2)
        self.rf1_gc03 = GatedConv1dWithActivation(n_fea, n_fea, 4, 2, 1)
        self.rf1_gc05 = GatedConv1dWithActivation(n_fea, n_fea, 3, 1, 1)
        self.rf1_gc07 = GatedConv1dWithActivation(n_fea, n_fea, 4, 2, 1)

        self.rf2_gc01 = GatedConv1dWithActivation(n_fea, 2 * n_fea, 5, 1, 2)
        self.rf2_gc02 = GatedConv1dWithActivation(2 * n_fea, 4 * n_fea, 3, 1, 1)

        self.attn_head1 = Attn(n_fea)
        self.attn_head2 = Attn(n_fea)
        self.attn_head3 = Attn(n_fea)
        self.attn_head4 = Attn(n_fea)

        self.rf_up_gc02 = GatedConv1dWithActivation(8 * n_fea, 4 * n_fea, 3, 1, 1)
        self.rf_up_gdc1 = GatedDeConv1dWithActivation(2, 4 * n_fea, 4 * n_fea, 3, 1, 1)
        self.rf_up_gc03 = GatedConv1dWithActivation(4 * n_fea, 2 * n_fea, 3, 1, 1)
        self.rf_up_gdc2 = GatedDeConv1dWithActivation(2, 2 * n_fea, 2 * n_fea, 3, 1, 1)
        self.rf_up_gc04 = GatedConv1dWithActivation(2 * n_fea, n_fea, 3, 1, 1)
        self.rf_up_gc05 = GatedConv1dWithActivation(n_fea, 1, 3, 1, 1)

        self.rec_info = True if config.EVAL_MODE else False
        self.name = name
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def rec_layer(self, nn_layer, x):
        layer_rec = {}
        layer_rec['in'] = x.cpu().detach().numpy()
        x, x_raw, score = nn_layer(x)
        layer_rec['raw'] = x_raw.cpu().detach().numpy()
        layer_rec['out'] = x.cpu().detach().numpy()
        layer_rec['score'] = score.cpu().detach().numpy()
        return x, layer_rec

    def forward(self, x):
        masked_x = x[:, 0, :].unsqueeze(1)
        mask = x[:, 1, :].unsqueeze(1)  # 1-hole
        temp = x[:, 2, :].unsqueeze(1)
        record = []

        # coarse layer
        if self.rec_info:
            x, rec = self.rec_layer(self.cor_gc01, x)
            record.append(rec)
            x, rec = self.rec_layer(self.cor_gc02, x)
            record.append(rec)
            x, rec = self.rec_layer(self.cor_gc03, x)
            record.append(rec)
            x, rec = self.rec_layer(self.cor_gc04, x)
            record.append(rec)
            x, rec = self.rec_layer(self.cor_gc05, x)
            record.append(rec)
            x, rec = self.rec_layer(self.cor_gdc1, x)
            record.append(rec)
            x, rec = self.rec_layer(self.cor_gc09, x)
            record.append(rec)
            x, rec = self.rec_layer(self.cor_gdc2, x)
            record.append(rec)
            x, rec = self.rec_layer(self.cor_gc11, x)
            record.append(rec)

            coarse_x = x

            x = torch.cat([masked_x + x * mask, mask, temp], dim=1)
            x, rec = self.rec_layer(self.rf1_gc01, x)
            record.append(rec)
            x, rec = self.rec_layer(self.rf1_gc03, x)
            record.append(rec)
            x, rec = self.rec_layer(self.rf1_gc05, x)
            record.append(rec)
            x, rec = self.rec_layer(self.rf1_gc07, x)
            record.append(rec)

            x1, rec = self.rec_layer(self.rf2_gc01, x)
            record.append(rec)
            x1, rec = self.rec_layer(self.rf2_gc02, x1)
            record.append(rec)

            x2, rec = self.rec_layer(self.attn_head1, x)
            record.append(rec)
            res = x2
            x2, rec = self.rec_layer(self.attn_head2, x2)
            record.append(rec)
            res = torch.cat([res, x2], dim=1)
            x2, rec = self.rec_layer(self.attn_head3, x2)
            record.append(rec)
            res = torch.cat([res, x2], dim=1)
            x2, rec = self.rec_layer(self.attn_head4, x2)
            record.append(rec)
            res = torch.cat([res, x2], dim=1)

            x, rec = self.rec_layer(self.rf_up_gc02, torch.cat([x1, res], dim=1))
            record.append(rec)
            x, rec = self.rec_layer(self.rf_up_gdc1, x)
            record.append(rec)
            x, rec = self.rec_layer(self.rf_up_gc03, x)
            record.append(rec)
            x, rec = self.rec_layer(self.rf_up_gdc2, x)
            record.append(rec)
            x, rec = self.rec_layer(self.rf_up_gc04, x)
            record.append(rec)
            x, rec = self.rec_layer(self.rf_up_gc05, x)
            record.append(rec)

        else:
            x, _, _ = self.cor_gc01(x)
            x, _, _ = self.cor_gc02(x)
            x, _, _ = self.cor_gc03(x)
            x, _, _ = self.cor_gc04(x)
            x, _, _ = self.cor_gc05(x)
            x, _, _ = self.cor_gdc1(x)
            x, _, _ = self.cor_gc09(x)
            x, _, _ = self.cor_gdc2(x)
            x, _, _ = self.cor_gc11(x)

            coarse_x = x

            x = torch.cat([masked_x + x * mask, mask, temp], dim=1)
            x, _, _ = self.rf1_gc01(x)
            x, _, _ = self.rf1_gc03(x)
            x, _, _ = self.rf1_gc05(x)
            x, _, _ = self.rf1_gc07(x)

            x1, _, _ = self.rf2_gc01(x)
            x1, _, _ = self.rf2_gc02(x1)

            x2, _, _ = self.attn_head1(x)
            res = x2
            x2, _, _ = self.attn_head2(x2)
            res = torch.cat([res, x2], dim=1)
            x2, _, _ = self.attn_head3(x2)
            res = torch.cat([res, x2], dim=1)
            x2, _, _ = self.attn_head4(x2)
            res = torch.cat([res, x2], dim=1)

            x, _, _ = self.rf_up_gc02(torch.cat([x1, res], dim=1))
            x, _, _ = self.rf_up_gdc1(x)
            x, _, _ = self.rf_up_gc03(x)
            x, _, _ = self.rf_up_gdc2(x)
            x, _, _ = self.rf_up_gc04(x)
            x, _, _ = self.rf_up_gc05(x)

        return x, coarse_x, record
        # return coarse_x

    def save_checkpoint(self, epoch):
        if epoch % config.SAVE_PER_EPO == 0:
            filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_epoch' + str(epoch) + '.h5')
            torch.save(self.state_dict(), filename)

    def save_best_checkpoint(self):
        filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_best.h5')
        torch.save(self.state_dict(), filename)
class DisGIN(nn.Module):
    def __init__(self, n_fea=8, name='disc'):
        super(DisGIN, self).__init__()
        if config.USE_LOCAL_GAN_LOSS:
            self.snconv1 = SNConvWithActivation(3, 2 * n_fea, 4, 2, 2)
            self.snconv2 = SNConvWithActivation(2 * n_fea, 4 * n_fea, 4, 2, 2)
            self.snconv3 = SNConvWithActivation(4 * n_fea, 8 * n_fea, 4, 2, 2)
            self.snconv4 = SNConvWithActivation(8 * n_fea, 8 * n_fea, 4, 2, 2)
            self.snconv5 = SNConvWithActivation(8 * n_fea, 8 * n_fea, 4, 2, 2)
            self.linear = nn.Linear(8 * n_fea * 2 * 2, 1)

            self.discriminator_net = nn.Sequential(
                SNConvWithActivation(3, 2 * n_fea, 4, 2, 2),
                SNConvWithActivation(2 * n_fea, 4 * n_fea, 4, 2, 2),
                SNConvWithActivation(4 * n_fea, 8 * n_fea, 4, 2, 2),
                SNConvWithActivation(8 * n_fea, 8 * n_fea, 4, 2, 2),
                SNConvWithActivation(8 * n_fea, 8 * n_fea, 4, 2, 2),
            )
        else:
            self.snconv1 = SNConvWithActivation(3, 2 * n_fea, 4, 2, 2)
            self.snconv2 = SNConvWithActivation(2 * n_fea, 4 * n_fea, 4, 2, 2)
            self.snconv3 = SNConvWithActivation(4 * n_fea, 8 * n_fea, 4, 2, 2)
            self.snconv4 = SNConvWithActivation(8 * n_fea, 8 * n_fea, 4, 2, 2)
            self.snconv5 = SNConvWithActivation(8 * n_fea, 8 * n_fea, 4, 2, 2)
            self.linear = nn.Linear(8 * n_fea * 2 * 2, 1)

            self.discriminator_net = nn.Sequential(
                SNConvWithActivation(3, 2 * n_fea, 4, 2, 2),
                SNConvWithActivation(2 * n_fea, 4 * n_fea, 4, 2, 2),
                SNConvWithActivation(4 * n_fea, 8 * n_fea, 4, 2, 2),
                SNConvWithActivation(8 * n_fea, 8 * n_fea, 4, 2, 2),
                # SNConvWithActivation(8*n_fea, 8*n_fea, 4, 2, padding=get_pad(16, 4, 2)),
                # SNConvWithActivation(8*n_fea, 8*n_fea, 4, 2, padding=get_pad(8, 4, 2)),
                # Self_Attn(8*n_fea, 'relu'),
                SNConvWithActivation(8 * n_fea, 8 * n_fea, 4, 2, 2),
            )

        self.name = name

    def forward(self, x):
        x1 = self.snconv1(x)
        x2 = self.snconv2(x1)
        x3 = self.snconv3(x2)
        x4 = self.snconv4(x3)
        x = self.snconv5(x4)
        # x = self.discriminator_net(x)
        x = x.view((x.size(0), -1))
        fea = torch.cat((x1.view((x1.size(0), -1)),
                         x2.view((x2.size(0), -1)),
                         x3.view((x3.size(0), -1)),
                         x4.view((x4.size(0), -1))), dim=1)
        # x = self.linear(x)
        return x, fea

    def save_checkpoint(self, epoch):
        if epoch % config.SAVE_PER_EPO == 0:
            filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_epoch' + str(epoch) + '.h5')
            torch.save(self.state_dict(), filename)

    def save_best_checkpoint(self):
        filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_best.h5')
        torch.save(self.state_dict(), filename)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    # trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    mask = torch.tensor([[True, True, True, True, True, False, False, False, False],
                         [True, True, True, True, True, False, False, False, False]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(device=device).to(device)
    out = model(x, mask, x)
    print(out.shape)

# todo
# makae load data into integers
# embed the integer load data
# use only encoder to inpainting missing load profile