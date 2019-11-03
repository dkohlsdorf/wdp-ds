# Models and Their Analysis

+ `convnet_v1`: multi layer convolutional auto encoder
+ `lstm_v2`: lstm stack on single conv layer (8 x 8), predicting the next frame
+ `lstm_v3`: lstm stack on concatenated conv layers (8 x 8 and 8 x 256), predicting the next frame.
        - `v3.1`: only catalogue
        - `v3.2`: catalogue + clusters from `v3.1`
        - `v3.3`: larger model / dont freeze all layers form encoder 
        - `v3.4`: larger model all data from before 
                  batch size increased to 100 since we have more data.
                  Predict 10 frames instead of one.
        - `v3.5`: same model as `v3.4` but use silence results not clustering results
        - `v3.6`: even more parameters but only data from thesis
