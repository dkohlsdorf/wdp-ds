# wdp-ds
Data Science For The Wild Dolphin Project

This is a collection of scripts to build and analyze dolphin communication models
and communicate the results with domain experts. 

<img width="1116" alt="Screenshot 2021-04-13 at 23 49 49" src="https://user-images.githubusercontent.com/1425435/114625700-f3016a00-9cb2-11eb-86d6-a27c8db584ef.png">

## IJCNN

+ The models from [IJCNN20](https://arxiv.org/abs/2005.07623) are [here](https://github.com/dkohlsdorf/wdp-ds/tree/v4.0/).
+ The version with DTW + HMMs is [here](https://github.com/dkohlsdorf/wdp-ds/tree/denise_semi_happy)

## Contents

+ 1) `lib_dolphin`: Training and Evaluation of Machine Learning Models
+ 2) `pipeline.py`: Run this to build a model

## Dependencies
+ sklearn
+ tensorflow 2.0
+ numpy / scipy
+ nmslib
+ numba
