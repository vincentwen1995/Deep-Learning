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

import os
import time
from datetime import datetime
import argparse

import numpy as np
import csv
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

################################################################################


def evaluate(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Load the dataset
    with open(config.dataset, 'rb') as dataset_file:
        dataset = pickle.load(dataset_file)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, config.lstm_num_hidden,
                                config.lstm_num_layers, device, config.dropout_keep_prob)  # fixme
    model.load_state_dict(torch.load(config.ckpt))

    # Generate some sentences by sampling from the model
    model.eval()
    # Create tensor to hold the generated samples.
    samples = torch.zeros((config.sample_batch_size, config.sample_length), dtype=torch.int, device=device, requires_grad=False)

    last_h = torch.zeros(config.lstm_num_layers, config.sample_batch_size, config.lstm_num_hidden, device=device, requires_grad=False)
    last_c = torch.zeros(config.lstm_num_layers, config.sample_batch_size, config.lstm_num_hidden, device=device, requires_grad=False)

    if config.pre_text:
        pre_input = torch.tensor([dataset._char_to_ix[ch] for ch in config.pre_text] * 10, device=device,
                                 requires_grad=False).view(config.sample_batch_size, -1).t().unsqueeze(-1)
        onehot_pre_input = torch.zeros((pre_input.shape[0], pre_input.shape[1], dataset.vocab_size), device=device, requires_grad=False)
        onehot_pre_input.scatter_(2, pre_input, 1)
        logits, last_h, last_c = model(onehot_pre_input, last_h, last_c)
        logits = nn.functional.softmax(logits[-1, :, :].unsqueeze(-1) / config.temperature, dim=1)
        start_chars = logits.squeeze().argmax(-1)
        samples[:, 0] = start_chars
        onehot_chars = torch.zeros((1, config.sample_batch_size, dataset.vocab_size), device=device, requires_grad=False)
        onehot_chars.scatter_(2, start_chars.view(1, config.sample_batch_size, 1), 1)
    else:
        # Initialize the first characters for the samples.
        start_chars = torch.randint(dataset.vocab_size, size=(1, config.sample_batch_size, 1), dtype=torch.long, device=device, requires_grad=False)
        samples[:, 0] = start_chars.squeeze()
        # Create a tensor to hold the one-hot encoding for the output characters of the LSTM network (one per each time step).
        onehot_chars = torch.zeros((1, config.sample_batch_size, dataset.vocab_size), device=device, requires_grad=False)
        onehot_chars.scatter_(2, start_chars, 1)

    for t in np.arange(config.sample_length - 1):
        logits, last_h, last_c = model(onehot_chars, last_h, last_c)
        logits = nn.functional.softmax(logits / config.temperature, dim=2)
        next_chars = logits.squeeze().argmax(-1)
        onehot_chars.zero_()
        onehot_chars.scatter_(2, next_chars.view(1, config.sample_batch_size, 1), 1)
        samples[:, t + 1] = next_chars

    samples = samples.tolist()
    samples = [dataset.convert_to_string(sample) for sample in samples]
    # Output the samples into a text file.
    with open(config.summary_path + 'samples.txt', 'a') as txt_file:
        txt_file.write('Temperature: {}\nSample length: {}\n'.format(config.temperature, config.sample_length))
        txt_file.writelines(map(lambda x: config.pre_text + x + '\n', samples))

    print('Done evaluation.')

 ################################################################################
 ################################################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5, help='Dropout keep probability')
    parser.add_argument('--lstm_num_hidden', type=int, default=256, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    parser.add_argument('--dataset', type=str, default='./model_ckpt/train.dataset', help='Dataset object to generate text')
    parser.add_argument('--ckpt', type=str, default='./model_ckpt/lstm_gen_epoch30.ckpt', help='Model checkpoint to load for generating the samples')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature parameter for generating the samples')
    parser.add_argument('--sample_batch_size', type=int, default=10, help='Number of samples to generate in a batch')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./eval_result/", help='Output path for evaluation')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--sample_length', type=int, default=30, help='Length of the sample sequence')
    parser.add_argument('--pre_text', type=str, default=None, help='Preformulated text/sentence for the generator to complete')

    config = parser.parse_args()

    # Train the model
    evaluate(config)
