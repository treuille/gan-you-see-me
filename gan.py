import notebook
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.datasets import mnist

import wandb
from wandb.wandb_keras import WandbKerasCallback

def generate_garbage(data):
    garbage = np.random.uniform(0.0, 1.0, size=data.shape)
    print('garbage', garbage.shape, garbage.dtype)
    print(garbage[0])

    # combine them together
    combined = np.concatenate([data, garbage])
    labels = np.zeros(data.shape[0] * 2)
    labels[:data.shape[0]] = 1
    indices = np.arange(combined.shape[0])
    np.random.shuffle(indices)
    combined = combined[indices]
    labels = labels[indices]
    combined.shape = combined.shape + (1,)

    return (combined, labels)

with notebook.Notebook() as print:
    # init wandb
    model = Sequential()
    run = wandb.init()

    # load the real training data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    print('x_train', x_train.shape, x_train.dtype)

    train, train_labels = generate_garbage(x_train)
    test, test_labels = generate_garbage(x_test)

    model.add(Conv2D(16, (3,3), input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='sgd', loss='binary_crossentropy',
        metrics=['accuracy'])

    print('the_model', model)
    model.summary(print_fn=print)

    model.fit(train, train_labels, validation_data=(test, test_labels),
        epochs=1,callbacks=[WandbKerasCallback()])
