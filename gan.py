import notebook
import numpy as np
import scipy.misc

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, \
    Conv2DTranspose, Reshape, AveragePooling2D, UpSampling2D
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import LambdaCallback
from keras.layers.advanced_activations import LeakyReLU
from keras import metrics
import keras
from os import path
import scipy.misc

import wandb
from wandb.wandb_keras import WandbKerasCallback

run = wandb.init()
config = run.config

config.discriminator_epochs = 3
config.discriminator_examples = 10000
config.generator_epochs = 3
config.generator_examples = 5000
config.generator_seed_dim = 10
config.generator_conv_size = 64
print(run.dir)

# previous_fake_train = [np.zeros((0,28,28))]
def mix_data(data, generator, length=1000):
    num_examples=int(length/2)

    data= data[:num_examples, :, :]
    seeds = [np.random.uniform(-100.0, 100.0,
        size=(num_examples, config.generator_seed_dim))]
    fake_train = generator.predict(seeds)[:,:,:,0]
    # print('data.shape', data.shape)
    # print('fake_train.shape', fake_train.shape)
    # print('previous_fake_train.shape', previous_fake_train[0].shape)
    # combined  = np.concatenate([ data, fake_train, previous_fake_train[0]])
    combined  = np.concatenate([ data, fake_train ])

    # previous_fake_train[0] = np.concatenate([fake_train, previous_fake_train[0]])
    # np.random.shuffle(previous_fake_train[0])
    # previous_fake_train[0] = previous_fake_train[0][:num_examples*2]
    # print('There are %s previous examples.' % len(previous_fake_train[0]))

    #print('garbage', garbage.shape, garbage.dtype)
    #print(garbage[0])

    # combine them together
    labels = np.zeros(combined.shape[0])
    labels[:data.shape[0]] = 1
    indices = np.arange(combined.shape[0])
    np.random.shuffle(indices)
    combined = combined[indices]
    labels = labels[indices]
    combined.shape += (1,)

    labels = np_utils.to_categorical(labels)

    return (combined, labels)

def log_discriminator(epoch, logs):
    run.history.add({
            'generator_loss': 0.0,
            'generator_acc': (1.0-logs['acc'])*2.0,
            'discriminator_loss': logs['loss'],
            'discriminator_acc': logs['acc']})
    run.summary['discriminator_loss'] = logs['loss']
    run.summary['discriminator_acc'] = logs['acc']

def create_discriminator():
    discriminator = Sequential()
    discriminator.add(Conv2D(16, (3,3), input_shape=(28,28,1)))
    discriminator.add(LeakyReLU(alpha=0.3))
    discriminator.add(AveragePooling2D(pool_size=(2,2)))
    discriminator.add(Dropout(0.5))
    discriminator.add(Conv2D(32, (3,3)))
    discriminator.add(LeakyReLU(alpha=0.3))

    discriminator.add(AveragePooling2D(pool_size=(2,2)))
    discriminator.add(Dropout(0.5))
    discriminator.add(Flatten(input_shape=(28,28,1)))
    discriminator.add(Dropout(0.5))
    discriminator.add(Dense(16))
    discriminator.add(LeakyReLU(alpha=0.3))

    discriminator.add(Dropout(0.5))
    discriminator.add(Dense(2, activation='softmax'))
    discriminator.compile(optimizer='sgd', loss='categorical_crossentropy',
        metrics=['acc'])
    return discriminator

def create_generator():
    # # sanity check braindead generator
    # generator = Sequential()
    # generator.add(Dense(28*28, input_shape=(1,), activation='tanh'))
    # generator.add(Reshape((28, 28, 1), input_shape=(1,)))
    # # generator.add(UpSampling2D())
    # # generator.add(Conv2DTranspose(64, (3,3), padding='same', activation='relu'))
    # # generator.add(UpSampling2D())
    # # generator.add(Conv2DTranspose(32, (3,3), padding='same', activation='relu'))
    # # generator.add(Conv2DTranspose(1, (3,3), padding='same', activation='tanh'))
    # generator.summary()
    # return generator

    generator = Sequential()
    generator.add(Dense(7*7*128, input_shape=(config.generator_seed_dim,)))
    generator.add(LeakyReLU(alpha=0.3))

    generator.add(Reshape((7, 7, 128), input_shape=(1,)))
    generator.add(Dropout(0.5))
    generator.add(UpSampling2D())
    generator.add(Conv2DTranspose(config.generator_conv_size, (5,5), padding='same'))
    generator.add(LeakyReLU(alpha=0.3))

    generator.add(Dropout(0.5))
    generator.add(UpSampling2D())
    # generator.add(Conv2DTranspose(4, (3,3), padding='same', activation='relu'))
    # generator.add(Dropout(0.5))
    generator.add(Conv2DTranspose(1, (3,3), padding='same', activation='tanh'))
    generator.summary()
    return generator

def create_joint_model(generator, discriminator):
    joint_model = Sequential()
    joint_model.add(generator)
    joint_model.add(discriminator)

    discriminator.trainable = False

    joint_model.compile(optimizer='adam', loss='categorical_crossentropy',
        metrics=['acc'])

    return joint_model


def train_discriminator(generator, discriminator, x_train, x_test, iter):

    train, train_labels = mix_data(x_train, generator, config.discriminator_examples)
    test, test_labels = mix_data(x_test, generator, config.discriminator_examples)
    print("Training Discriminator", fmt='header')
    for i in range(10):
        print((train[i,:,:,0] + 1.0) / 2.0, fmt="img")
        if train_labels[i,0] == 1.0:
            scipy.misc.imsave(f'image-{iter}.jpg', train[i,:,:,0])

    discriminator.trainable = True
    discriminator.summary()

    wandb_logging_callback = LambdaCallback(on_epoch_end=log_discriminator)
    notebook_callback = notebook.KerasCallback(print.add_block(), len(train))
    
    history = discriminator.fit(train, train_labels,
        epochs=config.discriminator_epochs,
        batch_size=config.batch_size, validation_data=(test, test_labels),
        callbacks = [wandb_logging_callback, notebook_callback])

    discriminator.save(path.join(run.dir, "discriminator.h5"))

def log_generator(epoch, logs):
    run.history.add({'generator_loss': logs['loss'],
                     'generator_acc': logs['acc'],
                     'discriminator_loss': 0.0,
                     'discriminator_acc': (1-logs['acc'])/2.0+0.5})
    run.summary['generator_loss'] = logs['loss']
    run.summary['generator_acc'] = logs['acc']

def train_generator(discriminator, joint_model):
    print("Training Generator", fmt='header')
    num_examples = config.generator_examples
    train = [np.random.uniform(-100.0, 100.0,
        size=(num_examples, config.generator_seed_dim)) ]
    labels = np_utils.to_categorical(np.ones(num_examples))

    wandb_logging_callback = LambdaCallback(on_epoch_end=log_generator)
    notebook_callback = notebook.KerasCallback(print.add_block(), len(train))
    
    discriminator.trainable = False
    
    joint_model.fit(train, labels, epochs=config.generator_epochs,
            batch_size=config.batch_size,
            callbacks=[wandb_logging_callback, notebook_callback])

    generator.save(path.join(run.dir, "generator.h5"))

with notebook.Notebook() as print:
    # init wandb

    # load the real training data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0 * 2.0 - 1.0
    x_test = x_test / 255.0 * 2.0 - 1.0
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # only try to create 8s
    # x_train = x_train[y_train == 8]
    # x_test = x_test[y_test == 8]

    discriminator = create_discriminator()
    generator = create_generator()
    joint_model = create_joint_model(generator, discriminator)

    # # just try training to output 8s
    # x_train.shape += (1,)
    # generator.compile(loss='mse', metrics=['mse'], optimizer='adam')
    # def show_image(epoch, logs):
    #     # if epoch % 100 != 0:
    #     #     return
    #     print(f'epoch: {epoch}')
    #     for k, v in logs.items():
    #         print(f'{k}: {v}')
    #     n_imgs = 10
    #     img = generator.predict(np.random.uniform(0.0, 1.0, 10))
    #     print('img', img.shape)
    #     for i in range(n_imgs):
    #         print(f'min={img[i,:,:,0].flatten().min()} max={img[i,:,:,0].flatten().max()}')
    #         print((img[i,:,:,0] + 1.0) / 2.0, fmt='img')
    # show_image_callback = LambdaCallback(on_epoch_end=show_image)
    # input = np.random.uniform(0.0, 1.0, x_train.shape[0])
    # generator.fit(input, x_train, batch_size=16, epochs=10000,
    #     callbacks=[show_image_callback])

    iter = 0
    while True:
        iter += 1
        train_discriminator(generator, discriminator, x_train, x_test, iter)
        train_generator(discriminator, joint_model)
