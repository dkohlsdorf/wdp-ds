# wdp-ds
Data Science For The Wild Dolphin Project

This is a collection of scripts to build and analyze dolphin communication models
and communicate the results with domain experts. 

## Contents

+ 1) `ml_pipeline`: Training and Evaluation of Machine Learning Models
+ 2) `config`: Config files to run pipeline

## Dependencies
+ sklearn
+ tensorflow 2.0
+ numpy / scipy
+ pydub
+ librosa
+ cython

# Setup
You need to run the cython compiler for all cython parts of the library
```
cythonize -if ml_pipeline/*.pyx
```

## Usage:
The machine learning pipeline can be run as often as necessary. 
Training will continue from the last encoder:
```
    usage for training:      python ml_pipeline/pipeline.py train config/default_config.yaml 2>debug.log
```
Evaluation can be run using the same script:
```
    evaluate model:          python ml_pipeline/pipeline.py evaluate config/default_config.yaml
```
A demo training run can be found in the ```training_script.sh```.
