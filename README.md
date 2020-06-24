# wdp-ds
Data Science For The Wild Dolphin Project

This is a collection of scripts to build and analyze dolphin communication models
and communicate the results with domain experts. 

## Contents

+ 1) `ml_pipeline`: Training and Evaluation of Machine Learning Models
+ 2) `config`: Config files to run pipeline


## Setup
You need to run the cython compiler for all cython parts of the library
```
    cythonize -if ml_pipeline/*.pyx
```


## Dependencies
+ sklearn
+ tensorflow 2.0
+ numpy / scipy
+ pydub
+ cython
+ kneed

## Usage:
The machine learning pipeline can be run in 4 modes:
```
    usage for training:      python ml_pipeline/pipeline.py train config/default_config.yaml
    usage for induction:     python ml_pipeline/pipeline.py induction config/induction_config.yaml
    usage for annotation:    python ml_pipeline/pipeline.py annotate config/annotation_config.yaml
    usage for word spotting: python ml_pipeline/pipeline.py simplified config/word_spotting.yaml
```

+ training:   learn the auto encoder as well as the silence detector and type classifier 
+ induction:  learn clustering using quantization, hierarchical clustering and hidden Markov models
+ annotation: given an annotation file assigning names to clusters, get new audio files for each snippet
+ word spotting: run training and induction without the silence detector and type classifier

## Running Tests
```
    python -m unittest discover -s ml_pipeline/ -v
```