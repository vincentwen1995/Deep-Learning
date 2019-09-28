# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden, lstm_num_layers, device, dropout_keep_prob):

        super(TextGenerationModel, self).__init__()
        # Initialization here...
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.device = device
        self.dropout_keep_prob = dropout_keep_prob
        self.lstm = nn.LSTM(self.vocabulary_size, self.lstm_num_hidden, self.lstm_num_layers, dropout=(1 - self.dropout_keep_prob))
        self.linear = nn.Linear(self.lstm_num_hidden, self.vocabulary_size)
        self.to(device)

    def forward(self, x, h, c):
        # Implementation here...
        out, (hidden, cell) = self.lstm(x, (h, c))
        out = self.linear(out)
        return out, hidden, cell
