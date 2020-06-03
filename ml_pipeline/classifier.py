# Classification on top of embedder
#  
# REFERENCES: 
# [KOH4] Daniel Kohlsdorf, Denise Herzing, Thad Starner: "An Auto Encoder For Audio Dolphin Communication", IJCNN, 2020


from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import l2

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
