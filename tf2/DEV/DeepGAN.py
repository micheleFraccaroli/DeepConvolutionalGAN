import os

from tqdm import tqdm
import glob
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import time

BUFFER_SIZE = 60000
BATCH_SIZE = 256


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')

    plt.savefig('res/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def load_data():
    x_trl, y_trl, x_tel, y_tel = [], [], [], []
    for dd in ["Dataset/train/", "Dataset/test/"]:
        for img in os.listdir(dd):
            try:
                original_data = cv2.imread(dd + img)
                if original_data is None:
                    continue
                data = cv2.resize(original_data, (100, 100))
                if "train" in dd:
                    x_trl.append(data)
                    y_trl.append("bike")
                else:
                    x_tel.append(data)
                    y_tel.append("bike")
            except EOFError:
                pass
    x_train = np.asarray(x_trl)
    x_train = x_train.reshape(362, 30000)
    y_train = np.asarray(y_trl)
    x_test = np.asarray(x_tel)
    y_test = np.asarray(y_tel)
    return x_train, y_train, x_test, y_test


(train_images, train_labels, _, _) = load_data()
train_images = train_images.reshape(
    train_images.shape[0], 100, 100, 3).astype('float32')
# Normalize the images to [-1, 1]
train_images = (train_images - 127.5) / 127.5

train_dataset = tf.data.Dataset.from_tensor_slices(
    train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def artist():
    model = tf.keras.Sequential()
    model.add(layers.Dense(25*25*256, use_bias=False, input_shape=(255,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((25, 25, 256)))
    # Note: None is the batch size
    assert model.output_shape == (None, 25, 25, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 25, 25, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 50, 50, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2),
                                     padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 100, 100, 3)

    return model


generator = artist()

noise = tf.random.normal([3, 255])
generated_image = generator(noise, training=False)
# plt.imshow(generated_image[0,:,:,0])
# plt.show()


def critic():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[100, 100, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


discriminator = critic()
decision = discriminator(generated_image)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def critic_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def artist_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 200
noise_dim = 255
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])


def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = artist_loss(fake_output)
        disc_loss = critic_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    path_ckpt = "./training_checkpoints/*"
    for epoch in range(epochs):
        for _ in tqdm(range(BATCH_SIZE)):
            start = time.time()

            for image_batch in dataset:
                train_step(image_batch)

            # Produce images for the GIF as we go
            #generate_and_save_images(generator,
            #                         epoch + 1,
            #                         seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 25 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
                # Generate after the final epoch
                generate_and_save_images(generator,
                                         epochs,
                                         seed)

            print('Time for epoch {} is {} sec'.format(
                epoch + 1, time.time()-start))

# def display_image(epoch_no):
#   return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
#
# display_image(EPOCHS)


train(train_dataset, EPOCHS)
