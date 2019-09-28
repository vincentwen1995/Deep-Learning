"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, bn_flag=False):
        """
        Initializes MLP object. 

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super(MLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        if not bn_flag:
            layers = []
            last_n_units = self.n_inputs
            for n_units in n_hidden:
                layers.append(nn.Linear(last_n_units, n_units))
                layers.append(nn.ReLU())
                last_n_units = n_units
        else:
            layers = []
            last_n_units = self.n_inputs
            for n_units in n_hidden:
                layers.append(nn.Linear(last_n_units, n_units))
                layers.append(nn.BatchNorm1d(n_units))
                layers.append(nn.ReLU())
                last_n_units = n_units

        layers.append(nn.Linear(last_n_units, self.n_classes))

        self.layers = nn.Sequential(*layers)
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through 
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        out = self.layers.forward(x)
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

        return out
