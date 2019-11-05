# Models and Their Analysis

+ `convnet_v1`: multi layer convolutional auto encoder
+ `lstm_v2`: lstm stack on single conv layer (8 x 8), predicting the next frame
+ `lstm_v3`: lstm stack on concatenated conv layers (8 x 8 and 8 x 256), predicting the next frame.
+ `lstm_v4`: encoder is lstm stacked on top of convnet (only local features) and the output is one single vector.
            the decoder is a lstm with one vector as an input and then predicting through the same 
            lstm stack anf conv layer as the encoder but reversed itself.
         