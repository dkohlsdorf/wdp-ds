from tensorflow.keras.layers import *
from tensorflow.keras.models import *

def classifier(encoder, n_labels=1):
    '''
    A classifier stacked on top of an encoder. 
    All layers except the last are frozen.
    
    encoder: an encoder that ouputs a vector 
    n_labels: number of labels for the classification

    returns: a keras model
    '''
    for layer in encoder.layers[:-1]:
        layer.trainable = False
    shape = encoder.layers[0].input_shape[0][1:]
    inp = Input(shape)
    x   = encoder(inp)
    x   = BatchNormalization()(x)    
    x   = Dense(64)(x)
    x   = Dense(32)(x)
    x   = Dropout(0.5)(x) 
    if n_labels == 1:
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs = [inp], outputs = [x])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        x = Dense(n_labels, activation='softmax')(x)
        model = Model(inputs = [inp], outputs = [x])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])        
    return model
