import notebook
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Conv2DTranspose, Reshape
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import LambdaCallback
from keras import metrics
import keras
from os import path

import wandb
from wandb.wandb_keras import WandbKerasCallback

run = wandb.init()
config = run.config

config.discriminator_epochs = 10
config.discriminator_examples = 1000
config.generator_epochs = 10
config.generator_examples = 5000
print(run.dir)

def mix_data(data, generator, length=1000):
    num_examples=int(length/2)

    data= data[:num_examples, :, :]
    seeds = [np.random.uniform(-100.0, 100.0, size=num_examples)]
    fake_train = generator.predict(seeds)[:,:,:,0]
    combined  = np.concatenate([ data, fake_train])

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

def log_discriminator(epoch, logs):
    run.history.add({'discriminator_loss': logs['loss'],
                 'discriminator_acc': logs['acc']})
    run.summary['discriminator_loss'] = logs['loss']
    run.summary['discriminator_acc'] = logs['acc']

def train_discriminator(generator, discriminator, x_train, x_test):

    train, train_labels = mix_data(x_train, generator, config.discriminator_examples)
    test, test_labels = mix_data(x_test, generator, config.discriminator_examples)
    print("Training Discriminator", fmt='header')
    for i in range(10):
        print((train[i,:,:,0] + 1.0) / 2.0, fmt="img")
        print(train_labels[i,0], train[i,:,:,0].flatten().max(), train[i,:,:,0].flatten().min())

    discriminator.trainable = True
    discriminator.summary()
    discriminator.compile(optimizer='sgd', loss='categorical_crossentropy',
            metrics=['acc'])

    wandb_logging_callback = LambdaCallback(on_epoch_end=log_discriminator)

    history = discriminator.fit(train, train_labels, epochs=config.discriminator_epochs,
        batch_size=config.batch_size, validation_data=(test, test_labels), callbacks = [wandb_logging_callback])

    discriminator.save(path.join(run.dir, "discriminator.h5"))

def log_generator(epoch, logs):
    run.history.add({'generator_loss': logs['loss'],
                     'generator_acc': logs['acc']})
    run.summary['generator_loss'] = logs['loss']
    run.summary['generator_acc'] = logs['acc']

def train_generator(generator, discriminator):
    print("Training Generator")
    num_examples = config.generator_examples
    train = [np.random.uniform(-100.0, 100.0, size=num_examples)]
    labels = np_utils.to_categorical(np.ones(num_examples))
    joint_model = Sequential()
    joint_model.add(generator)
    joint_model.add(discriminator)

    discriminator.trainable = False

    wandb_logging_callback = LambdaCallback(on_epoch_end=log_generator)

    joint_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
    joint_model.fit(train, labels, epochs=config.generator_epochs,
            batch_size=config.batch_size,
            callbacks=[wandb_logging_callback])

    generator.save(path.join(run.dir, "generator.h5"))

with notebook.Notebook() as print:
    # init wandb

    # load the real training data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0 * 2.0 - 1.0
    x_test = x_test / 255.0 * 2.0 - 1.0
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
    generator.add(Dense(49, input_shape=(1,), activation='relu'))
    generator.add(Reshape((7, 7, 1), input_shape=(1,)))
    generator.add(Conv2DTranspose(32, (3,3), input_shape=(100,), strides=2, padding='same', activation='relu'))
    generator.add(Conv2DTranspose(1, (3,3), input_shape=(100,), strides=2, padding='same', activation='tanh'))
    generator.summary()

    for i in range(100):
        train_discriminator(generator, discriminator, x_train, x_test)
        train_generator(generator, discriminator)
