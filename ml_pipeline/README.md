# Machine Learning Scripts

+ `audio.py`: methods to read audio files and create spectrograms
+ `audio_collection.py`: write audio snippets into one file        
+ `feature_extractor.py`: contains an auto encoder to embed spectrograms based on lstm encoder / decoder 
+ `classifier.py`: build a classifier on top of an embedding
+ `plots.py`: methods to inspect the results of the encoders
+ `dtw.pyx`: dynamic time warping distance implemented in cython
+ `msa.py`: (multiple) sequence alignment functions
+ `colors.txt`: the xkcd color scheme
+ `pipeline.py`: the training pipeline for silence detector and embedder including evaluation
+ `sequence_embedder.py`: apply the model to larger files
+ `reporting_template.md`: template for reporting
+ `generate_report.py`: report generation tool
