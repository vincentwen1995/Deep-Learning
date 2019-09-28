import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse


def running_average(x, window_size=5):
    mid = window_size // 2
    x = x.squeeze()
    y = np.copy(x)
    len_x = x.shape[0]
    for i in np.arange(len_x):
        window_sum = 0
        offset = 0
        for j in np.arange(i - mid, i + mid + 1):
            if j < 0 or j >= len_x:
                offset += 1
                continue
            window_sum += x[j]
        y[i] = window_sum / (window_size - offset)
    return y


def main(config):
    # files = ['5', '6', '7', '8', '9', '10', '15', '20', '25']
    files = ['1', '2', '3', '4', '5']
    with open('result/' + 'train_epoch' + '1' + '.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        len_acc = len(list(csv_reader)[0])

    accs = np.empty(shape=(len(files), len_acc), dtype=np.float)
    plt.figure(figsize=(10, 5))
    for i_file, file in enumerate(files):
        with open('result/' + 'train_epoch' + file + '.csv', 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for i_row, row in enumerate(csv_reader):
                if i_row == 1:
                    accs[i_file, :] = np.array(list(map(float, row)))
        plt.plot(np.arange(accs.shape[1]), running_average(accs[i_file, :], 21), linewidth=0.8, label='Epoch_' + file)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig('result/acc.png', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    config = parser.parse_args()

    main(config)
