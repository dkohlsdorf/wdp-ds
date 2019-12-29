from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import l2

def classifier(encoder, n_labels=1):
    """
    A classifier stacked on top of an encoder.
    All layers except the last are frozen.

    :param encoder: an encoder that ouputs a vector
    :param n_labels: number of labels for the classification

    :returns: a keras model
    """
    shape = encoder.layers[0].input_shape[0][1:]
    inp = Input(shape)
    x   = encoder(inp)
    x   = BatchNormalization()(x)    
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
