echo "Train"
python ml_pipeline/pipeline.py train config/default_config.yaml 2>debug.log
echo "Eval"
python ml_pipeline/pipeline.py evaluate config/default_config.yaml