from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import random
import matplotlib.pyplot as plt

from cifar10 import default_model_path, load_eval_data, load_test_data, create_data_gen, load_model


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default=default_model_path)
parser.add_argument('--test-images', default=False, action='store_true')
parser.add_argument('--incorrect', default=False, action='store_true')
parser.add_argument('--shuffle', type=bool, default=True)
args = parser.parse_args()


images, labels, data_size = load_test_data() if args.test_images else load_eval_data()

data_gen = create_data_gen(images)
data_gen_input = data_gen.flow(images, labels, batch_size=data_size, shuffle=False)
prediction_images = next(data_gen_input)[0]


model = load_model(args.model_path)


probabilities_list = model.predict(prediction_images).tolist()

if args.shuffle:
    shuffled_indices = list(range(len(images)))
    random.shuffle(shuffled_indices)
    images = [images[i] for i in shuffled_indices]
    labels = [labels[i] for i in shuffled_indices]
    probabilities_list = [probabilities_list[i] for i in shuffled_indices]

predictions = [probabilities.index(max(probabilities)) for probabilities in probabilities_list]


if args.incorrect:
    incorrect_indices = [i for i in range(len(predictions)) if predictions[i] != labels[i]]
    probabilities_list = [probabilities_list[i] for i in incorrect_indices]
    predictions = [predictions[i] for i in incorrect_indices]
    images = [images[i] for i in incorrect_indices]
    labels = [labels[i] for i in incorrect_indices]


def plot_image(max_probability, prediction, correct_label, image):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap=plt.cm.binary)

    color = 'blue' if prediction == correct_label else 'red'
    plt.xlabel('{} ({:2.0f}%) ({})'.format(class_names[prediction], 100 * max_probability, class_names[correct_label]), color=color)

def plot_value_array(probabilities, prediction, correct_label):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.ylim([0, 1])

    thisplot = plt.bar(range(10), probabilities, color='#777777')
    thisplot[prediction].set_color('red')
    thisplot[correct_label].set_color('blue')


plot_index = 0
next = True
while next:
    num_rows = 5
    num_cols = 5
    num_subplots = num_rows * num_cols

    print('Showing predictions {}-{}'.format(plot_index * num_subplots + 1, min((plot_index + 1) * num_subplots, len(predictions))))

    plt.figure(figsize=(12, 6))
    for i in range(num_subplots):
        subplot_index = plot_index * num_subplots + i
        if subplot_index < len(predictions):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            plot_image(max(probabilities_list[subplot_index]), predictions[subplot_index], labels[subplot_index], images[subplot_index])

            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            plot_value_array(probabilities_list[subplot_index], predictions[subplot_index], labels[subplot_index])
    plt.subplots_adjust(left=0.02, bottom=0.04, right=0.98, top=0.98, wspace=0.02, hspace=0.3)
    plt.show()

    plot_index += 1

    if plot_index * num_subplots < len(predictions):
        next = input('Show more predictions? (Y)es or (N)o: ').lower() == 'y'
    else:
        next = False
