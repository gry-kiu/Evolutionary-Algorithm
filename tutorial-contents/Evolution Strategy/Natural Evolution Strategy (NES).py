# Compatibility update for TensorFlow 2

"""
The basic idea about Nature Evolution Strategy with visualation.
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
Dependencies:
Tensorflow >= r1.2
numpy
matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

tf.compat.v1.disable_eager_execution()

DNA_SIZE = 2  # parameter (solution) number
N_POP = 20  # population size
N_GENERATION = 100  # training step
LR = 0.02  # learning rate


# fitness function
def get_fitness(pred):
    return -((pred[:, 0]) ** 2 + pred[:, 1] ** 2)


# build multivariate distribution
mean = tf.Variable(tf.random.normal([2,], 13.0, 1.0), dtype=tf.float32)
cov = tf.Variable(5.0 * tf.eye(DNA_SIZE), dtype=tf.float32)
mvn = tfp.distributions.MultivariateNormalFullCovariance(
    loc=mean, covariance_matrix=cov
)
# sampling operation
make_kid = mvn.sample(N_POP)

# compute gradient and update mean and covariance matrix from sample and fitness
tfkids_fit = tf.placeholder(tf.float32, [N_POP,])
tfkids = tf.placeholder(tf.float32, [N_POP, DNA_SIZE])
# log prob * fitness
loss = -tf.reduce_mean(mvn.log_prob(tfkids) * tfkids_fit)
train_op = tf.train.GradientDescentOptimizer(LR).minimize(
    loss
)  # compute and apply gradients for mean and cov

sess = tf.Session()
# initialize tf variables
sess.run(tf.global_variables_initializer())

# something about plotting (can be ignored)
n = 300
x = np.linspace(-20, 20, n)
X, Y = np.meshgrid(x, x)
Z = np.zeros_like(X)
for i in range(n):
    for j in range(n):
        Z[i, j] = get_fitness(np.array([[x[i], x[j]]]))
plt.contourf(X, Y, -Z, 100, cmap=plt.cm.rainbow)
plt.ylim(-20, 20)
plt.xlim(-20, 20)
plt.ion()

# training
for g in range(N_GENERATION):
    kids = sess.run(make_kid)
    kids_fit = get_fitness(kids)
    # update distribution parameters
    sess.run(train_op, {tfkids_fit: kids_fit, tfkids: kids})

    # plotting update
    if "sca" in globals():
        sca.remove()
    sca = plt.scatter(kids[:, 0], kids[:, 1], s=30, c="k")
    plt.pause(0.01)

print("Finished")
plt.ioff()
plt.show()
