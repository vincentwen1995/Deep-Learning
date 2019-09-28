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


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.W_hx = nn.Parameter(torch.zeros(self.num_hidden, self.input_dim))
        self.W_hh = nn.Parameter(torch.zeros(self.num_hidden, self.num_hidden))
        self.b_h = nn.Parameter(torch.zeros(self.num_hidden))
        self.W_ph = nn.Parameter(torch.zeros(self.num_classes, self.num_hidden))
        self.b_p = nn.Parameter(torch.zeros(self.num_classes))
        nn.init.xavier_normal_(self.W_hx)
        nn.init.xavier_normal_(self.W_hh)
        nn.init.xavier_normal_(self.W_ph)
        self.to(device)

    def forward(self, x):
        # Implementation here ...
        tanh = nn.Tanh()
        h_init = torch.zeros(x.size(0), self.num_hidden)
        h_init = h_init.to(self.device)
        last_h = h_init
        for t in np.arange(self.seq_length):
            # print('W_hx {}, x[:, t] {}, W_hh {}, last_h {}, b_h {}'.format(self.W_hx.size(), x[:, t].size(), self.W_hh.size(), last_h.size(), self.b_h.size()))
            h = tanh(x[:, t].reshape(-1, 1) @ self.W_hx.t() + last_h @ self.W_hh.t() + self.b_h)
            p = h @ self.W_ph.t() + self.b_p
            last_h = h

        return p
