from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import pandas as pd

from tensorflow.keras import callbacks, models

from cifar10 import default_model_path, default_log_path, load_train_and_eval_data, create_data_gen, create_train_data_gen, load_model


parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default=default_model_path)
parser.add_argument('--log-path', type=str, default=default_log_path)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=64)
args = parser.parse_args()


(train_images, train_labels, train_data_size), (eval_images, eval_labels, eval_data_size) = load_train_and_eval_data()

train_data_gen = create_train_data_gen(train_images)
train_data_gen_input = train_data_gen.flow(train_images, train_labels, batch_size=args.batch_size)

eval_data_gen = create_data_gen(eval_images)
eval_data_gen_input = eval_data_gen.flow(eval_images, eval_labels, batch_size=eval_data_size)
eval_images, eval_labels = next(eval_data_gen_input)


model = load_model(args.model_path)


class SaveModelAndLog(callbacks.Callback):
    def __init__(self, model_path, log_path):
        super(SaveModelAndLog, self).__init__()

        self.model_path = model_path
        self.log_path = log_path

    def on_train_begin(self, logs=None):
        try:
            self.log = pd.read_csv(self.log_path, index_col='epoch')
            self.best_val_acc = max(self.log['validation accuracy'].tolist())
        except:
            self.log = None
            self.best_val_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') > self.best_val_acc:
            self.model.save(self.model_path)
            self.best_val_acc = logs.get('val_accuracy')

        try:
            self.log = self.log.append(pd.DataFrame({'train loss': [logs.get('loss')], 'train accuracy': [logs.get('accuracy')], 'validation loss': [logs.get('val_loss')], 'validation accuracy': [logs.get('val_accuracy')]}), ignore_index=True)
        except:
            self.log = pd.DataFrame({'train loss': [logs.get('loss')], 'train accuracy': [logs.get('accuracy')], 'validation loss': [logs.get('val_loss')], 'validation accuracy': [logs.get('val_accuracy')]})
        self.log.index += 1
        self.log.index.name = 'epoch'
        self.log.to_csv(self.log_path)

train = model.fit_generator(train_data_gen_input, epochs=args.epochs, validation_data=(eval_images, eval_labels), callbacks=[SaveModelAndLog(args.model_path, args.log_path)])
