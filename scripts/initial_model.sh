python ml/train_convnet_features.py 32 128 data/catalogue/whistle_snippets/ data/catalogue/burst_snippet/
python ml/train_silence_detector.py 32 data/catalogue/whistle_snippets/ data/catalogue/burst_snippet/ data/catalogue/noise_snippets/
python ml/model_analysis.py 32 data/demo data/catalogue/whistle_snippets/ data/catalogue/burst_snippet/ data/catalogue/noise_snippets/