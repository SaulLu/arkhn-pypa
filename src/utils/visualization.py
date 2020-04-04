import csv
from matplotlib import pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_pathname",
        type=str,
        default='test',
        help="name of the root folder associated to the model")
    model_pathname = parser.parse_args().model_pathname

    epoch = []
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    train_f1 = []
    val_f1 = []

    with open(f'data/results/{model_pathname}/metrics.csv', 'r') as f:
        reader = csv.reader(f)
        _ = next(reader)
        for row in reader:
            epoch.append(int(row[0]))
            train_loss.append(float(row[1]))
            val_loss.append(float(row[2]))
            train_accuracy.append(float(row[3]))
            val_accuracy.append(float(row[4]))
            train_f1.append(float(row[5]))
            val_f1.append(float(row[6]))


    _, axs = plt.subplots(1, 3, figsize=(15, 15))

    loss_plot = axs[0]
    loss_plot.plot(epoch, train_loss, c='red', label='Train')
    loss_plot.plot(epoch, val_loss, c='blue', label='Val')
    loss_plot.set_title('Loss')
    loss_plot.set(xlabel='Epoch', ylabel='Loss')
    loss_plot.legend()

    accu_plot = axs[1]
    accu_plot.plot(epoch, train_accuracy, c='red', label='Train')
    accu_plot.plot(epoch, val_accuracy, c='blue', label='Val')
    accu_plot.set_title('Accuracy')
    accu_plot.set(xlabel='Epoch', ylabel='Accuracy')
    accu_plot.legend()

    f1_plot = axs[2]
    f1_plot.plot(epoch, train_f1, c='red', label='Train')
    f1_plot.plot(epoch, val_f1, c='blue', label='Val')
    f1_plot.set_title('F1')
    f1_plot.set(xlabel='Epoch', ylabel='F1')
    f1_plot.legend()

    plt.savefig(f'data/results/{model_pathname}/metrics.png')

if __name__ == "__main__":
    main()
