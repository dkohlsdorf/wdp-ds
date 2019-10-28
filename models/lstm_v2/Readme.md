# LSTM V2

Idea: Build an LSTM stack that predicts the next element in the sequence. Build a silence detector on top. 

## Data:
The data is the sound type catalog from my thesis. We slice each
audio file using a sliding window of 32 frames. The spectorgram is
computed using a window of 512 samples with a 256 sample skip.

## Model
The encoder is shown below:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 32, 256, 1)]      0         
_________________________________________________________________
conv2d (Conv2D)              (None, 32, 1, 32)         24608     
_________________________________________________________________
reshape (Reshape)            (None, 32, 32)            0         
_________________________________________________________________
bidirectional (Bidirectional (None, 32, 16)            2624      
_________________________________________________________________
lstm_1 (LSTM)                (None, 64)                20736     
=================================================================
Total params: 47,968
Trainable params: 47,968
Non-trainable params: 0
_________________________________________________________________
```

# Offline Eval

The silence detector's confusion matrix is:

|truth/prediction|not silence|silence|
|:---|:---|:---|
|not silence|120|28|
|silence|26|715|


The embedding ... 

![embedding](images/embedding.png)

... zoomed into the whistle part (top right corner)

![embedding](images/embedding_zoomed.png)


# Conclusion

Results: `agglomerative_dtw_lstm`

+ Silence detector finds way more (~ 3500)
