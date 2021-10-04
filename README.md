# wdp-ds
Cancel changesData Science For The Wild Dolphin Project

This is a collection of scripts to build and analyze dolphin communication models
and communicate the results with domain experts. 

## History

+ The models from [IJCNN20](https://arxiv.org/abs/2005.07623), December 2019 are [here](https://github.com/dkohlsdorf/wdp-ds/tree/v4.0/). 
+ The version with DTW + HMMs is [here](https://github.com/dkohlsdorf/wdp-ds/tree/denise_semi_happy), June 2020
+ The HTK version of the Conv LSTM HMM Hybrid is [here](https://github.com/dkohlsdorf/wdp-ds/releases/tag/v15), Aug 2021
+ The version that also includes a lot of evaluastion and training [here](https://github.com/dkohlsdorf/wdp-ds/releases/tag/v16) Sep 2021
## Contents

+ 1) `lib_dolphin`: Training and Evaluation of Machine Learning Models
+ 2) `pipeline.py`: Run this to build a model

## Dependencies
+ sklearn
+ tensorflow 2.0
+ numpy / scipy
+ numba
+ matplotlib
+ htk [Optional]
