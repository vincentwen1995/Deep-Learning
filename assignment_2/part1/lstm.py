################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
################################################################################


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        self.W_gx = nn.Parameter(torch.zeros(self.num_hidden, self.input_dim))
        self.W_gh = nn.Parameter(torch.zeros(self.num_hidden, self.num_hidden))
        self.b_g = nn.Parameter(torch.zeros(self.num_hidden))
        self.W_ix = nn.Parameter(torch.zeros(self.num_hidden, self.input_dim))
        self.W_ih = nn.Parameter(torch.zeros(self.num_hidden, self.num_hidden))
        self.b_i = nn.Parameter(torch.zeros(self.num_hidden))
        self.W_fx = nn.Parameter(torch.zeros(self.num_hidden, self.input_dim))
        self.W_fh = nn.Parameter(torch.zeros(self.num_hidden, self.num_hidden))
        self.b_f = nn.Parameter(torch.zeros(self.num_hidden))
        self.W_ox = nn.Parameter(torch.zeros(self.num_hidden, self.input_dim))
        self.W_oh = nn.Parameter(torch.zeros(self.num_hidden, self.num_hidden))
        self.b_o = nn.Parameter(torch.zeros(self.num_hidden))
        self.W_ph = nn.Parameter(torch.zeros(self.num_classes, self.num_hidden))
        self.b_p = nn.Parameter(torch.zeros(self.num_classes))
        nn.init.xavier_normal_(self.W_gx)
        nn.init.xavier_normal_(self.W_gh)
        nn.init.xavier_normal_(self.W_ix)
        nn.init.xavier_normal_(self.W_ih)
        nn.init.xavier_normal_(self.W_fx)
        nn.init.xavier_normal_(self.W_fh)
        nn.init.xavier_normal_(self.W_ox)
        nn.init.xavier_normal_(self.W_oh)
        nn.init.xavier_normal_(self.W_ph)
        self.to(device)

    def forward(self, x):
        # Implementation here ...
        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()

        h_init = torch.zeros(x.size(0), self.num_hidden)
        h_init = h_init.to(self.device)
        last_h = h_init

        c_init = torch.zeros(x.size(0), self.num_hidden)
        c_init = c_init.to(self.device)
        last_c = c_init
        for t in np.arange(self.seq_length):
            g = tanh(x[:, t].reshape(-1, 1) @ self.W_gx.t() + last_h @ self.W_gh.t() + self.b_g)
            i = sigmoid(x[:, t].reshape(-1, 1) @ self.W_ix.t() + last_h @ self.W_ih.t() + self.b_i)
            f = sigmoid(x[:, t].reshape(-1, 1) @ self.W_fx.t() + last_h @ self.W_fh.t() + self.b_f)
            o = sigmoid(x[:, t].reshape(-1, 1) @ self.W_ox.t() + last_h @ self.W_oh.t() + self.b_o)
            c = g * i + last_c * f
            h = tanh(c) * o
            # p = h @ self.W_ph.t() + self.b_p
            last_h = h
            last_c = c
        p = h @ self.W_ph.t() + self.b_p
        return p
