# Machine Learning Scripts

+ `audio.py`: methods to read audio files and create spectrograms
+ `audio_collection.py`: write audio snippets into one file        
+ `feature_extractor.py`: contains an auto encoder to embed spectrograms based on lstm encoder / decoder 
+ `classifier.py`: build a classifier on top of an embedding
+ `plots.py`: methods to inspect the results of the encoders
+ `colors.txt`: the xkcd color scheme
+ `pipeline.py`: the training pipeline for silence detector and embedder including evaluation
+ `sequence_embedder.py`: apply the model to larger files
+ `dtw.pyx`: cython implementation of dynamic programming
+ `structured.py`: structuring dolphin communication 
+ `sequence_hashing.pyx`: using symbolic aggregate approximation for bucketing
+ `baum_welch.pyx`: estimation code for hidden Markov models
+ `distributions.py`: distributions for hidden Markov models like Gaussians
+ `fwd_bwd.pyx`: exact inference
+ `hidden_markov_model.py`: implements the actual hmm
+ `logprob.py`: probability calculations in log space
+ `markov_chain.py`: implements a markov chain
+ `viterbi.pyx`: viterbi algorithm for hidden Markov model decoding
+ `integration_test_audio.py`: testing audio reading and writing
+ `variational_feature_extractor.py`: variational auto encoder