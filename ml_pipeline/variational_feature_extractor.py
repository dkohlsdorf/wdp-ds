# Variational Deep Unsupervised Feature Learning
#  
# Variational Model Shamelessly stolen from: https://www.tensorflow.org/tutorials/generative/cvae
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import * 
from tensorflow.keras.losses import * 

from feature_extractor import * 
import numpy as np
import tensorflow as tf


class FeatureVAE(Model):

  def __init__(self, input_shape, latent_dim):
    super(FeatureVAE, self).__init__()
    self.latent_dim = latent_dim
    self.latent  = encoder(input_shape, latent_dim)
    inp          = Input(input_shape)
    x            = self.latent(inp)
    x            = Dense(2 * latent_dim)(x)
    self.encoder = Model(inputs = [inp], outputs = [x])
    self.decoder = decoder(input_shape[0], latent_dim, input_shape[1])  

  @tf.function
  def sample(self, batch, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(batch, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z):
    return self.decoder(z)

  def predict(self, x):
    return self.encode(x)[0]    

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss