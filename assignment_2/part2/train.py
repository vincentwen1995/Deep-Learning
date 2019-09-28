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


def accuracy_fn(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      targets: 2D int array of size [batch_size, n_classes]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    predictions = predictions.argmax(dim=-1)
    matches = torch.eq(predictions, targets)
    accuracy = matches.sum().item() / predictions.shape[0]
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Save the instantiated dataset.
    with open('model_ckpt/train.dataset', 'wb') as dataset_file:
        pickle.dump(dataset, dataset_file)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, config.lstm_num_hidden,
                                config.lstm_num_layers, device, config.dropout_keep_prob)  # fixme

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()  # reduction='mean'(default) - average over all timesteps and all batches as they are merged.
    optimizer = optim.RMSprop(model.parameters(), config.learning_rate)  # fixme
    # optimizer = optim.Adam(model.parameters(), config.learning_rate)

    # Create a tensor to hold the one-hot encoding for the batch inputs.
    onehot_batch_inputs = torch.FloatTensor(config.seq_length, config.batch_size, dataset.vocab_size)
    onehot_batch_inputs = onehot_batch_inputs.to(device)

    h_init = torch.zeros(config.lstm_num_layers, config.batch_size, config.lstm_num_hidden, device=device)
    c_init = torch.zeros(config.lstm_num_layers, config.batch_size, config.lstm_num_hidden, device=device)

    # Record the learning rate steps individually for learning rate decay.
    lr_step = 0
    lr = 1
    for epoch in np.arange(config.epochs):
        losses = []
        accs = []
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################
            # Add more code here ...
            #######################################################
            model.train()
            # Convert the DataLoader output from list of tensors to tensors.
            batch_inputs = torch.stack(batch_inputs)
            batch_inputs = batch_inputs.to(device)

            # If the epoch is finished and there is not enough character to extract, break the loop
            if batch_inputs.shape[0] * batch_inputs.shape[1] != onehot_batch_inputs.shape[0] * onehot_batch_inputs.shape[1]:
                break

            # Zero the one-hot encoding and encode according to batch_inputs.
            onehot_batch_inputs.zero_()
            onehot_batch_inputs.scatter_(2, batch_inputs.unsqueeze_(-1), 1)

            # Convert the DataLoader output from list of tensors to tensors.
            batch_targets = torch.stack(batch_targets)
            batch_targets = batch_targets.to(device)

            # Learning rate decay.
            if lr_step % config.learning_rate_step == 0:
                optimizer = optim.RMSprop(model.parameters(), config.learning_rate * lr)
                lr *= config.learning_rate_decay

            optimizer.zero_grad()
            logits, _, _ = model(onehot_batch_inputs, h_init, c_init)
            # The seq_length dimension and batch_size dimension of the logits and batch_targets are merged together, and the mean is computed over this new dimension.
            loss = criterion(logits.view(-1, dataset.vocab_size), batch_targets.view(-1))   # fixme
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)

            accuracy = accuracy_fn(logits.view(-1, dataset.vocab_size), batch_targets.view(-1))  # fixme
            optimizer.step()

            losses.append(loss.item())
            accs.append(accuracy)

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:
                print("[{}] Epoch {}/{}, Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                          datetime.now().strftime("%Y-%m-%d %H:%M"), epoch + 1, config.epochs, step,
                          config.train_steps, config.batch_size, examples_per_second,
                          accuracy, loss
                      ))

            if step % config.sample_every == 0:
                # Generate some sentences by sampling from the model
                model.eval()
                # Create tensor to hold the generated samples.
                samples = torch.zeros((5, config.sample_length), dtype=torch.int, device=device)
                # Initialize the first characters for the samples.
                start_chars = torch.randint(dataset.vocab_size, size=(1, 5, 1), dtype=torch.long, device=device)
                samples[:, 0] = start_chars.squeeze()
                # Create a tensor to hold the one-hot encoding for the output characters of the LSTM network (one per each time step).
                onehot_chars = torch.zeros((1, 5, dataset.vocab_size), device=device)
                onehot_chars.scatter_(2, start_chars, 1)

                last_h = torch.zeros(config.lstm_num_layers, 5, config.lstm_num_hidden, device=device)
                last_c = torch.zeros(config.lstm_num_layers, 5, config.lstm_num_hidden, device=device)
                for t in np.arange(config.sample_length - 1):
                    logits, last_h, last_c = model(onehot_chars, last_h, last_c)
                    next_chars = logits.squeeze().argmax(-1)
                    onehot_chars.zero_()
                    onehot_chars.scatter_(2, next_chars.view(1, 5, 1), 1)
                    samples[:, t + 1] = next_chars

                samples = samples.tolist()
                samples = [dataset.convert_to_string(sample) for sample in samples]
                # Output the samples into a text file.
                with open(config.summary_path + 'samples.txt', 'a') as txt_file:
                    txt_file.write('Epoch: {}\nStep: {}\n'.format(epoch + 1, step))
                    txt_file.writelines(map(lambda x: x + '\n', samples))

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

            lr_step += 1
        # After each training epoch, save the model and the training loss and accuracy.
        model.train()
        torch.save(model.state_dict(), 'model_ckpt/lstm_gen_epoch{}.ckpt'.format(epoch + 1))
        with open(config.summary_path + 'train_epoch{}.csv'.format(epoch + 1), 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(losses)
            csv_writer.writerow(accs)

    print('Done training.')

 ################################################################################
 ################################################################################


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=256, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=10000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1000000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./result/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--sample_length', type=int, default=30, help='Length of the sample sequence')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')

    config = parser.parse_args()

    # Train the model
    train(config)
