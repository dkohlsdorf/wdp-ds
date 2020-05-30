# wdp-ds
Data Science For The Wild Dolphin Project

This is a collection of scripts to build and analyze dolphin communication models
and communicate the results with domain experts. 

## Contents

+ 1) `ml_pipeline`: Training and Evaluation of Machine Learning Models
+ 2) `config`: Config files to run pipeline

## Usage:
```
    usage for training:      python ml_pipeline/pipeline.py train config/default_config.yaml
    usage for induction:     python ml_pipeline/pipeline.py induction config/induction_config.yaml
    usage for annotation:    python ml_pipeline/pipeline.py annotate config/annotation_config.yaml
    usage for word spotting: python ml_pipeline/pipeline.py simplified config/word_spotting.yaml
```

## Running Tests
```
    python -m unittest discover -s ml_pipeline/tests -v
```