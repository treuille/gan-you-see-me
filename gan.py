import notebook
import numpy as np

from keras.datasets import mnist

import wandb
from wandb.wandb_keras import WandbKerasCallback

with notebook.Notebook() as print:
    # init wandb
    run = wandb.init()

    # load the real training data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    print('x_train', x_train.shape, x_train.dtype)
    # print(x_train[0])

    # create some garbage data
    x_garbage = np.random.uniform(0.0, 1.0, size=x_train.shape)
    print('x_garbage', x_garbage.shape, x_garbage.dtype)
    print(x_garbage[0])

    # combine them together
    x_combined = np.concatenate([x_train, x_garbage])
    labels = np.zeros(x_train.shape[0] * 2)
    labels[:x_train.shape[0]] = 1
    indices = np.arange(x_train.shape[0] * 2)
    np.random.shuffle(indices)
    print(indices[:10])
    x_combined = x_combined[indices]
    labels = labels[indices]

    for i in range(10):
        print(i, f'label={labels[i]}', x_combined[i])
    # print('combined shape', x_combined.shape)
    # print('labels', labels.shape)
