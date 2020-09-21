# Classification on top of embedder
#  
# REFERENCES: 
# [KOH4] Daniel Kohlsdorf, Denise Herzing, Thad Starner: "An Auto Encoder For Audio Dolphin Communication", IJCNN, 2020
# [JEN20] Jensen et. al: "Coincidence, Categorization, and Consolidation: Learning to Recognize Sounds with Minimal Supervision", ICASSP 2020.

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.activations import * 
from tensorflow.keras.regularizers import *
from tensorflow.keras.losses import *


class ClusteringLoss(Loss):
  '''
  Entropy loss [JEN20] 
  '''
  def __init__(self, gamma=1.0):
    super().__init__()
    self.gamma = gamma
 
  def call(self, true, pred):
    y  = pred + 1e-6
    h1 = -1.0 * tf.reduce_sum(
         tf.math.log(y) * y, axis=-1
    )
    H1 = tf.math.reduce_mean(h1)
    total_y = tf.math.reduce_mean(y, axis=0)
    H2 = -1 * tf.reduce_sum(
       tf.math.log(total_y) * total_y
    )
    return H1 - self.gamma * H2


def classifier(encoder, n_labels=1, freeze=True):
    """
    A classifier stacked on top of an encoder.
    All layers except the last are frozen.

    See Figure [KOH4] Figure 2. left part as encoder

    :param encoder: an encoder that ouputs a vector
    :param n_labels: number of labels for the classification

    :returns: a keras model
    """
    if freeze:
        for layer in encoder.layers:
            layer.trainable = False        
    shape = encoder.layers[0].input_shape[0][1:]
    inp = Input(shape)
    x   = encoder(inp)
    if n_labels == 0:
        x = Activation(softmax)(x)
        model = Model(inputs = [inp], outputs = [x])
        model.compile(optimizer='adam', loss=ClusteringLoss(), metrics=['accuracy'])        
    else:
        x   = BatchNormalization()(x)    
        # Multi layer neural network for classification on top of the encoder
        x   = Dense(64, activation='relu')(x)
        x   = Dense(32, activation='relu')(x)
        x   = Dropout(0.5)(x)
        if n_labels == 1:
            x = Dense(1, activation='sigmoid')(x)
            model = Model(inputs = [inp], outputs = [x])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])        
        else:
            x = Dense(n_labels, activation='softmax')(x)
            model = Model(inputs = [inp], outputs = [x])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])        
    return model
