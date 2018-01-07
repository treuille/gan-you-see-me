import notebook
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Conv2DTranspose, Reshape
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD

from keras import metrics
import keras

import wandb
from wandb.wandb_keras import WandbKerasCallback

def mix_data(data, generator):
    num_examples=data.shape[0]
    seeds = [np.random.uniform(-100.0, 100.0, size=num_examples)]
    fake_train = generator.predict(seeds)[:,:,:,0]
    print(fake_train.shape)
    combined  = np.concatenate([fake_train, data])

    #print('garbage', garbage.shape, garbage.dtype)
    #print(garbage[0])

    # combine them together
    labels = np.zeros(data.shape[0] * 2)
    labels[:data.shape[0]] = 1
    indices = np.arange(combined.shape[0])
    np.random.shuffle(indices)
    combined = combined[indices]
    labels = labels[indices]
    combined.shape += (1,)

    labels = np_utils.to_categorical(labels)

    return (combined, labels)

def train_discriminator(generator, discriminator, x_train, x_test):

    train, train_labels = mix_data(x_train, generator)
    test, test_labels = mix_data(x_test, generator)

    discriminator.trainable = True
    discriminator.compile(optimizer=sgd, loss='categorical_crossentropy',
            metrics=['acc'])
    discriminator.fit(train, train_labels, epochs=1,batch_size=256,
                    validation_data=(test, test_labels))
    print("Done Training discriminator")


def train_generator(generator, discriminator):
    print("Training Generator")
    num_examples = 10000
    train = [np.random.uniform(-100.0, 100.0, size=num_examples)]
    labels = np_utils.to_categorical(np.zeros(num_examples))
    joint_model = Sequential()
    joint_model.add(generator)
    joint_model.add(discriminator)

    discriminator.trainable = False

    joint_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])
    joint_model.fit(train, labels, epochs=1,batch_size=256)

with notebook.Notebook() as print:
    # init wandb
    run = wandb.init()

    # load the real training data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.
    x_test = x_test / 255.
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #print('x_train', x_train.shape, x_train.dtype)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    #print(train_labels[:10])
    discriminator = Sequential()

    discriminator.add(Conv2D(16, (3,3), input_shape=(28,28,1), activation='relu'))
    discriminator.add(MaxPooling2D(pool_size=(2,2)))
    discriminator.add(Conv2D(32, (3,3), activation='relu'))
    discriminator.add(MaxPooling2D(pool_size=(2,2)))
    discriminator.add(Flatten(input_shape=(28,28,1)))
    discriminator.add(Dense(16, activation='relu'))
    discriminator.add(Dense(2, activation='softmax'))

    generator = Sequential()
    generator.add(Dense(49, input_shape=(1,)))
    generator.add(Reshape((7, 7, 1), input_shape=(1,)))
    generator.add(Conv2DTranspose(32, (3,3), input_shape=(100,), strides=2, padding='same'))
    generator.add(Conv2DTranspose(1, (3,3), input_shape=(100,), strides=2, padding='same'))
    generator.summary()

    for i in range(100):
        train_discriminator(generator, discriminator, x_train, x_test)
        train_generator(generator, discriminator)
