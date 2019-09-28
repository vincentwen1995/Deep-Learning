"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data. 
    """

    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module. 

        Args:
          in_features: size of each input sample
          out_features: size of each output sample

        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
        std = 0.0001. Initialize biases self.params['bias'] with 0. 

        Also, initialize gradients with zeros.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.params = {'weight': np.random.normal(scale=0.0001, size=(in_features, out_features)), 'bias': np.zeros(out_features)}
        self.grads = {'weight': np.zeros(shape=self.params['weight'].shape), 'bias': np.zeros(out_features)}
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module. 

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        out = x @ self.params['weight'] + self.params['bias']
        self.x = x
        # self.activation = out
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads['weight'] = self.x.T @ dout
        self.grads['bias'] = np.mean(dout)
        dx = dout @ self.params['weight'].T
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class ReLUModule(object):
    """
    ReLU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module. 

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.live_inds = x > 0
        out = x * self.live_inds
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dx = dout * self.live_inds
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = x
        x_max = x.max(axis=-1)
        exp_x = np.exp(x - x_max[:, np.newaxis])
        self.softmax = exp_x / exp_x.sum(axis=-1)[:, np.newaxis]
        out = self.softmax
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # dx = np.empty(dout.shape, float)
        # for i in np.arange(dx.shape[0]):
        # dx[i, :] = dout[i, :] @ (np.diag(self.softmax[i, :]) - np.outer(self.softmax[i, :], self.softmax[i, :]))
        tmp = np.split(self.softmax, dout.shape[0], axis=0)
        tmp = tmp * np.eye(self.softmax.shape[1])
        tmp = tmp - self.softmax[:, :, np.newaxis] * self.softmax[:, np.newaxis, :]
        dx = np.squeeze(dout[:, np.newaxis, :] @ tmp)
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module. 
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        out = -np.log(x[np.arange(y.shape[0]), np.argmax(y, axis=-1)])
        out = np.mean(out, axis=0)
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # dx = np.zeros(shape=x.shape)
        # argmax_y = np.argmax(y, axis=-1)
        # dx[argmax_y] = -1 / x[argmax_y]
        dx = - (y / x) / y.shape[0]
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx
