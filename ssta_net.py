from models import *
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

seed = 0
random.seed(seed)

class SSTA_Net(nn.Module):
    def __init__(self, input_dim, h_units, act, args):
        super(SSTA_Net, self).__init__()
        # [10, 128, 128, 5]
        self.filter_size = args.filter_size
        self.padding = self.filter_size // 2
        self.frame_predictor = DeterministicConvLSTM(input_dim, h_units[-1], h_units[0], len(h_units), args)
        self.l3 = nn.Conv3d(h_units[-1], 2, kernel_size=self.filter_size, stride=1, padding=self.padding, bias=False)

        if act == "relu":
            self.act = F.relu
        elif act == "sig":
            self.act = F.sigmoid

    def __call__(self, x, m_t, m_t_others, memory):
        pred_x_tp1, message, memory\
            = self.forward(x, m_t, m_t_others, memory)
        return pred_x_tp1, message, memory

    def forward(self, x_t, m_t, m_t_others, frame_predictor_hidden):
        x = torch.cat([x_t, m_t, *m_t_others] , -1)
        x = x.permute(0, 4, 1, 2, 3)
        
        h, frame_predictor_hidden = self.frame_predictor(x, frame_predictor_hidden)
        pred_x_tp1 = self.l3(h)
        message = m_t
        message = None
        pred_x_tp1 = pred_x_tp1.permute(0, 2, 3, 4, 1)
        pred_x_tp1 = F.sigmoid(pred_x_tp1)
        return pred_x_tp1, message, frame_predictor_hidden

    def predict(self, x, m_t, m_t_others, memory):
        pred_x_tp1, message, memory\
            = self.forward(x, m_t, m_t_others, memory)
        return pred_x_tp1.data, message.data, memory