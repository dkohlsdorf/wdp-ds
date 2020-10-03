echo "Train 1"
python ml_pipeline/pipeline.py train config/default_config.yaml 2>debug.log
echo "Train 2"
python ml_pipeline/pipeline.py train config/default_config.yaml 2>>debug.log
echo "Train 3"
python ml_pipeline/pipeline.py train config/default_config.yaml 2>>debug.log
echo "Train 4"
python ml_pipeline/pipeline.py train config/default_config.yaml 2>>debug.log
echo "Eval"
python ml_pipeline/pipeline.py evaluate config/default_config.yaml