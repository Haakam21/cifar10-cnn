from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from cifar10 import default_model_path, default_log_path, load_eval_data, create_data_gen, load_model


parser = argparse.ArgumentParser()
parser.add_argument('--model-paths', nargs='+', default=[default_model_path])
parser.add_argument('--log-path', nargs='+', default=[default_log_path])
parser.add_argument('--no-train', default=False, action='store_true')
parser.add_argument('--no-loss', default=False, action='store_true')
args = parser.parse_args()


images, labels, data_size = load_eval_data()

data_gen = create_data_gen(images)
data_gen_input = data_gen.flow(images, labels, batch_size=data_size)
images, labels = next(data_gen_input)


for path in args.model_paths:
    model = load_model(path, False)

    loss, accuracy = model.evaluate(images, labels, verbose=0)
    print(path + ' validation accuracy: {0:.4f}'.format(accuracy))


logs = [pd.read_csv(path) for path in args.log_paths]

max_epochs = min([log['epoch'].tolist()[-1] for log in logs])
epochs = range(1, max_epochs + 1)


if not args.no_loss:
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for i in range(len(logs)):
        if not args.no_train: plt.plot(epochs, logs[i]['train loss'].tolist()[:max_epochs], label=args.log_paths[i] + ' train')
        plt.plot(epochs, logs[i]['validation loss'].tolist()[:max_epochs], label=args.log_paths[i] + ' val')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.subplot(1, 2, 2)
else:
    plt.figure(figsize=(6, 6))

for i in range(len(logs)):
    if not args.no_train: plt.plot(epochs, logs[i]['train accuracy'].tolist()[:max_epochs], label=args.log_paths[i] + ' train')
    plt.plot(epochs, logs[i]['validation accuracy'].tolist()[:max_epochs], label=args.log_paths[i] + ' val')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.15, hspace=0.3)
plt.show()
