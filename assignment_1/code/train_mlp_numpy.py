"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule, SoftMaxModule, LinearModule
import cifar10_utils
from matplotlib import pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'  # '100'
LEARNING_RATE_DEFAULT = 2e-3  # 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
                                                                                                                                      with one-hot encoding. Ground truth labels for
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
    # results = np.diag(predictions @ targets.T)
    # accuracy = results.sum() / results.shape[0]
    predictions = np.argmax(predictions, axis=1)
    targets = np.argmax(targets, axis=1)
    matches = np.equal(predictions, targets).astype(int)
    accuracy = np.sum(matches) / predictions.shape[0]
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    # DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    # Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    def reshape_cifar10_mlp(x):
        batch_size = x.shape[0]
        x = x.transpose([2, 3, 1, 0])
        x = x.reshape([-1, batch_size])
        x = x.transpose()
        return x

    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    x_train, y_train = cifar10['train'].next_batch(FLAGS.batch_size)
    x_train = reshape_cifar10_mlp(x_train)

    softmax = SoftMaxModule()
    crossent = CrossEntropyModule()
    mlp = MLP(x_train.shape[1], dnn_hidden_units, y_train.shape[1])

    train_accs = []
    train_losses = []
    eval_accs = []
    eval_losses = []
    for i in np.arange(FLAGS.max_steps):
        print('\nStep: {}\n'.format(i))
        print('Training: ')
        logits = mlp.forward(x_train)
        softmax_logits = softmax.forward(logits)
        train_loss = crossent.forward(softmax_logits, y_train)
        train_acc = accuracy(softmax_logits, y_train)
        print('loss: {:.4f}, acc: {:.4f}\n'.format(train_loss, train_acc))

        dL = crossent.backward(softmax_logits, y_train)
        dL = softmax.backward(dL)
        mlp.backward(dL)

        for layer in mlp.layers:
            if isinstance(layer, LinearModule):
                layer.params['weight'] -= FLAGS.learning_rate * layer.grads['weight']
                layer.params['bias'] -= FLAGS.learning_rate * layer.grads['bias']

        x_train, y_train = cifar10['train'].next_batch(FLAGS.batch_size)
        x_train = reshape_cifar10_mlp(x_train)
        if i % FLAGS.eval_freq == 0:
            print('Evaluation: ')
            x_eval, y_eval = cifar10['test'].images, cifar10['test'].labels
            x_eval = reshape_cifar10_mlp(x_eval)

            logits = mlp.forward(x_eval)
            softmax_logits = softmax.forward(logits)
            eval_loss = crossent.forward(softmax_logits, y_eval)
            eval_acc = accuracy(softmax_logits, y_eval)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            eval_losses.append(eval_loss)
            eval_accs.append(eval_acc)
            print('loss: {:.4f}, acc: {:.4f}'.format(eval_loss, eval_acc))

    print('Evaluation: ')
    x_eval, y_eval = cifar10['test'].images, cifar10['test'].labels
    x_eval = reshape_cifar10_mlp(x_eval)

    logits = mlp.forward(x_eval)
    softmax_logits = softmax.forward(logits)
    eval_loss = crossent.forward(softmax_logits, y_eval)
    eval_acc = accuracy(softmax_logits, y_eval)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    eval_losses.append(eval_loss)
    eval_accs.append(eval_acc)
    print('loss: {:.4f}, acc: {:.4f}'.format(eval_loss, eval_acc))

    print('Finished training.')

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(train_losses)), train_losses, label='training loss')
    plt.plot(np.arange(len(eval_losses)), eval_losses, label='evaluation loss')
    plt.legend()
    plt.xlabel('Iterations [x{}]'.format(FLAGS.eval_freq))
    plt.savefig('results/mlp_loss.png', bbox_inches='tight')

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(train_accs)), train_accs, label='training accuracy')
    plt.plot(np.arange(len(eval_accs)), eval_accs, label='evaluation accuracy')
    plt.legend()
    plt.xlabel('Iterations [x{}]'.format(FLAGS.eval_freq))
    plt.savefig('results/mlp_acc.png', bbox_inches='tight')

    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
