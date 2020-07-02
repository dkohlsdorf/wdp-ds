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

  def save(self, output_folder, epoch=None):
    if epoch is not None:
      self.latent.save('{}/encoder_{}.h5'.format(output_folder, epoch))
      self.encoder.save('{}/va_encoder_{}.h5'.format(output_folder, epoch))
      self.decoder.save('{}/decoder_{}.h5'.format(output_folder, epoch))
    else:
      self.latent.save('{}/encoder.h5'.format(output_folder))
      self.encoder.save('{}/va_encoder.h5'.format(output_folder))
      self.decoder.save('{}/decoder.h5'.format(output_folder))

  @classmethod
  def from_files(cls, output_folder, epoch=None):    
    if epoch is None:
      latent  = load_model('{}/encoder.h5'.format(output_folder))
      encoder = load_model('{}/va_encoder.h5'.format(output_folder))
      decoder = load_model('{}/decoder.h5'.format(output_folder))
    else:
      latent  = load_model('{}/encoder_{}.h5'.format(output_folder, epoch))
      encoder = load_model('{}/va_encoder_{}.h5'.format(output_folder, epoch))
      decoder = load_model('{}/decoder_{}.h5'.format(output_folder, epoch))

    input_shape = (latent.layers[0].input.shape[1], latent.layers[0].input.shape[2], latent.layers[0].input.shape[3])
    latent_dim  = m.layers[-1].output.shape[1]
    vae = cls(input_shape, latent_dim)
    vae.latent  = latent
    vae.encoder = encoder
    vae.decoder = decoder 
    return vae


  def reconstruct(self, x):
    mean, logvar = self.encode(x)
    z            = self.reparameterize(mean, logvar)
    r            = self.decode(z)
    return r

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z):
    return self.decoder(z)

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
  logpx_z   = -tf.reduce_mean(tf.math.square(x_logit - x), axis=[1, 2, 3])
  logpz     = log_normal_pdf(z, 0., 0.)
  logqz_x   = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss