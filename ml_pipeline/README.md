# Machine Learning Scripts

+ 1. `audio.py`: methods to read audio files and create spectrograms
+ 2. `audio_collection.py`: write audio snippets into one file        
+ 3. `feature_extractor.py`: contains an auto encoder to embed spectrograms based on lstm encoder / decoder 
+ 4. `classifier.py`: build a classifier on top of an embedding
+ 5. `plots.py`: methods to inspect the results of the encoders
+ 6. `colors.txt`: the xkcd color scheme
+ 7. `pipeline.py`: the training pipeline for silence detector and embedder including evaluation
+ 8. `sequence_embedder.py`: apply the model to larger files 
+ 9. `dtw.pyx`: cython implementation of dynamic time warping
+ 10. `sequence_clustering.py`: cluster sequences using dynamic time warping
+ 11. `sequence_hashing.pyx`: symbolic aggregate approximation