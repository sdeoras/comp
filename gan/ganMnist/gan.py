# this code has been modified from the original source (see reference to original code below).
# changes:
#    * Variables initialize using placeholders
#    * Weights and biases are saved using proto buf serialization
#    * A reshape tf-op has been added
#
# The main purpose of this code is to obtain a trained model as a checkpoint file along with
# the graph that can execute the computation. Both graph and checkpoint files are in protobuf
# format and are imported by the go code.
#
# Dependency:
# you will need to copy python files from github.com/sdeoras/api/pb
# ========= instructions for python =============
# "copy python file to virtualenv site package for it to be imported
# for instance here: ~/.venv/lib/python3.6/site-packages
# or ~/.conda/envs/comp/lib/python2.7/site-packages
# where <comp> is the name of conda environment
# or copy it to the folder where other python file is trying to import it
# or copy where ever site packages are stored

""" Generative Adversarial Networks (GAN).
Using generative adversarial networks (GAN) to generate digit images from a
noise distribution.
References:
    - Generative adversarial nets. I Goodfellow, J Pouget-Abadie, M Mirza,
    B Xu, D Warde-Farley, S Ozair, Y. Bengio. Advances in neural information
    processing systems, 2672-2680.
    - Understanding the difficulty of training deep feedforward neural networks.
    X Glorot, Y Bengio. Aistats 9, 249-256
Links:
    - [GAN Paper](https://arxiv.org/pdf/1406.2661.pdf).
    - [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
    - [Xavier Glorot Init](www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.../AISTATS2010_Glorot.pdf).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import comp_pb2 as proto

# we will use proto to serialize weights and biases and store them as checkpoint
cp = proto.Checkpoint()
weights_cp = cp.weights
biases_cp = cp.biases

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Params
num_steps = 100000
batch_size = 128
learning_rate = 0.0002

# Network Params
image_dim = 784 # 28*28 pixels
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100 # Noise data points

# labels
labels = ['gen_hidden1', 'gen_out', 'disc_hidden1', 'disc_out']

buff = tf.placeholder(dtype=tf.float32, name="buff")
shape = tf.placeholder(dtype=tf.int64, name="shape")
mat_reshape = tf.reshape(buff, shape=shape)
tf.identity(mat_reshape, name="reshapeOp")

# initial weights and biases
weights_init = {
    'gen_hidden1': tf.random_normal(shape=[noise_dim, gen_hidden_dim], stddev=1. / tf.sqrt(noise_dim / 2.)),
    'gen_out': tf.random_normal(shape=[gen_hidden_dim, image_dim], stddev=1. / tf.sqrt(gen_hidden_dim / 2.)),
    'disc_hidden1': tf.random_normal(shape=[image_dim, disc_hidden_dim], stddev=1. / tf.sqrt(image_dim / 2.)),
    'disc_out': tf.random_normal(shape=[disc_hidden_dim, 1], stddev=1. / tf.sqrt(disc_hidden_dim / 2.)),
}
biases_init = {
    'gen_hidden1': tf.zeros(shape=[gen_hidden_dim]),
    'gen_out': tf.zeros(shape=[image_dim]),
    'disc_hidden1': tf.zeros(shape=[disc_hidden_dim]),
    'disc_out': tf.zeros(shape=[1]),
}

# placeholders (you can feed initial weights and biases or checkpoint values later)
weights_ph = {
    'gen_hidden1': tf.placeholder(dtype=tf.float32, shape=[None, None], name="gen_hidden1_w_ph"),
    'gen_out': tf.placeholder(dtype=tf.float32, shape=[None, None], name="gen_out_w_ph"),
    'disc_hidden1': tf.placeholder(dtype=tf.float32, shape=[None, None], name="disc_hidden1_w_ph"),
    'disc_out': tf.placeholder(dtype=tf.float32, shape=[None, None], name="disc_out_w_ph"),
}
biases_ph = {
    'gen_hidden1': tf.placeholder(dtype=tf.float32, shape=[None], name="gen_hidden1_b_ph"),
    'gen_out': tf.placeholder(dtype=tf.float32, shape=[None], name="gen_out_b_ph"),
    'disc_hidden1': tf.placeholder(dtype=tf.float32, shape=[None], name="disc_hidden1_b_ph"),
    'disc_out': tf.placeholder(dtype=tf.float32, shape=[None], name="disc_out_b_ph"),
}

# variables defined by placeholders
weights = {
    'gen_hidden1': tf.Variable(weights_ph['gen_hidden1'], validate_shape=False, name="gen_hidden1_weights"),
    'gen_out': tf.Variable(weights_ph['gen_out'], validate_shape=False, name="gen_out_weights"),
    'disc_hidden1': tf.Variable(weights_ph['disc_hidden1'], validate_shape=False, name="disc_hidden1_weights"),
    'disc_out': tf.Variable(weights_ph['disc_out'], validate_shape=False, name="disc_out_weights"),
}
biases = {
    'gen_hidden1': tf.Variable(biases_ph['gen_hidden1'], validate_shape=False, name="gen_hidden1_biases"),
    'gen_out': tf.Variable(biases_ph['gen_out'], validate_shape=False, name="gen_out_biases"),
    'disc_hidden1': tf.Variable(biases_ph['disc_hidden1'], validate_shape=False, name="disc_hidden1_biases"),
    'disc_out': tf.Variable(biases_ph['disc_out'], validate_shape=False, name="disc_out_biases"),
}


# Generator
def generator(x):
    hidden_layer = tf.matmul(x, weights['gen_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer, name="generatorOutLayer")
    return out_layer


# Discriminator
def discriminator(x, tag):
    hidden_layer = tf.matmul(x, weights['disc_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
    out_layer = tf.nn.sigmoid(out_layer, name=tag+"DiscriminatorOutLayer")
    return out_layer


# Build Networks
# Network Inputs
gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')

# Build Generator Network
gen_sample = generator(gen_input)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(disc_input, "real")
disc_fake = discriminator(gen_sample, "fake")

# Build Loss
gen_loss = -tf.reduce_mean(tf.log(disc_fake))
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = [weights['gen_hidden1'], weights['gen_out'],
            biases['gen_hidden1'], biases['gen_out']]
# Discriminator Network Variables
disc_vars = [weights['disc_hidden1'], weights['disc_out'],
            biases['disc_hidden1'], biases['disc_out']]

# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    feed_dict = {}

    # run init value ops for weights and biases and collect in feed-dict
    for label in labels:
        x = sess.run(weights_init[label])
        feed_dict[weights_ph[label]] = x

        x = sess.run(biases_init[label])
        feed_dict[biases_ph[label]] = x

    # Run the initializer, passing feed_dict that we built in above step
    sess.run(init, feed_dict=feed_dict)

    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)
        # Generate noise to feed to the generator
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

        # Train
        feed_dict = {disc_input: batch_x, gen_input: z}
        _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                feed_dict=feed_dict)
        if i % 1000 == 0 or i == 1:
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

    # collect weights and biases and save them as checkpoints
    for label in labels:
        w = sess.run([weights[label]])
        weights_cp[label].data.extend(w[0].flatten())
        weights_cp[label].size.extend(w[0].shape)
        w = sess.run([biases[label]])
        biases_cp[label].data.extend(w[0].flatten())
        biases_cp[label].size.extend(w[0].shape)

    with open('./model/cp.pb', 'wb') as f:
        f.write(cp.SerializeToString())

    print("saved checkpoint to ./model/cp.pb")

    # Generate images from noise, using the generator network.
    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[4, noise_dim])
        g = sess.run([gen_sample], feed_dict={gen_input: z})
        g = np.reshape(g, newshape=(4, 28, 28, 1))
        # Reverse colours for better display
        g = -1 * (g - 1)
        for j in range(4):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                             newshape=(28, 28, 3))
            a[j][i].imshow(img)

    f.show()
    plt.draw()
    plt.waitforbuttonpress()

    # finally save the graph to be used in Go code
    graph = tf.Session().graph_def
    tf.io.write_graph(graph, "./model", "graph.pb", as_text=False)