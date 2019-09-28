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
        self.params = {'weight': np.random.normal(scale=0.0001, size=(out_features, in_features)), 'bias': np.zeros(out_features)}
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
        out = self.params['weight']@x + self.params['bias']
        self.x = x
        self.activation = out
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
        self.grads['weight'] = np.outer(self.x, dout)
        self.grads['bias'] = dout
        dx = dout @ self.params['weight']
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
        self.dead_inds = x < 0
        x[self.dead_inds] = 0
        out = x
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
        dout[self.dead_inds] = 0
        dx = dout
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
        x_max = x.max()
        exp_x = np.exp(x - x_max)
        self.softmax = exp_x / exp_x.sum()
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
        dx = dout @ (np.diag(self.softmax) - np.outer(self.softmax, self.softmax))
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
        out = -np.log(x[np.argmax(y)])
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
        dx = np.zeros(shape=x.shape)
        argmax_y = np.argmax(y)
        dx[argmax_y] = -1 / x[argmax_y]
        # raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx
