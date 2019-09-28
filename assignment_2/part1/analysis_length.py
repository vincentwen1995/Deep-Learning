import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main(config):
    files = list(map(str, range(5, 26, 1)))
    accs = np.empty(shape=(len(files), 10001), dtype=np.float)
    plt.figure(figsize=(10, 5))
    for i_file, file in enumerate(files):
        with open('result/' + config.model_type + '/P' + file + '.csv', 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for i_row, row in enumerate(csv_reader):
                if i_row == 1:
                    accs[i_file, :] = np.array(list(map(float, row)))
        # plt.plot(np.arange(accs.shape[1]), running_average(accs[i_file, :], 21), linewidth=0.8, label='T=' + file)
    # plt.plot(np.arange(5, 26, 1, dtype=np.int8), np.max(accs, axis=-1), '-o')
    plt.plot(np.arange(5, 26, 1, dtype=np.int8), accs[:, -1], '-o')
    # plt.legend()
    plt.xlabel('Palindrome Length')
    plt.ylabel('Final Accuracy')
    plt.savefig('result/' + config.model_type + '/palindrome_length.png', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    config = parser.parse_args()

    main(config)
