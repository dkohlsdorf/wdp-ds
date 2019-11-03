# ConvNet V1
python ml/train_convnet_features.py 32 128 data/catalogue/whistle_snippets/ data/catalogue/burst_snippet/
python ml/train_silence_detector.py 32 data/catalogue/whistle_snippets/ data/catalogue/burst_snippet/ data/catalogue/noise_snippets/
python ml/model_analysis.py 32 data/demo/ self models/convnet_v1/ data/catalogue/whistle_snippets/ data/catalogue/burst_snippet/ data/catalogue/noise_snippets/

# LSTM V2
python ml/train_lstm_features.py 32 128 data/catalogue/whistle_snippets/ data/catalogue/burst_snippet/
python ml/train_silence_detector.py 32 data/catalogue/whistle_snippets/ data/catalogue/burst_snippet/ data/catalogue/noise_snippets/
python ml/model_analysis.py 32 data/demo/ predict_next models/lstm_v2/ data/catalogue/whistle_snippets/ data/catalogue/burst_snippet/ data/catalogue/noise_snippets/

# LSTM V3
python ml/train_lstm_combined.py 128 128 data/catalogue/whistle_snippets/ data/catalogue/burst_snippet/ data/2012/
python ml/train_silence_detector.py 128 data/catalogue/whistle_snippets/ data/catalogue/burst_snippet/ data/catalogue/noise_snippets/ data/catalogue2011/noise/ data/catalogue2011/not_noise/
python ml/model_analysis.py 128 data/demo/ predict_next models/lstm_v3/v3.6/ data/catalogue/whistle_snippets/ data/catalogue/burst_snippet/ data/catalogue/noise_snippets/


