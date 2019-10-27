# Collections of Scripts 

+ `ingest_annotations.py`: collect PVL and ENCODING files and join them into csv files for cloud sql import. It will also normalize the words for easier search, later.
+ `enc_schema.sql`: mysql, encoding schema for the wild dolphin data
+ `pvl_schema.sql`: mysql, schema for video log annotations
+ `audiofile_schema.sql`: mysql, schema for audio files to encoding
+ `stopwords.txt`: a collections of stopwords for keyword generation
+ `drive2cloudByColab`: move audio files from google drive to cloud storage using colab
+ `initial_model.sh`: train auto encoder and silence model
+ `scripts/SilenceDetector.ipynb`: apply silence detector in gcloud using notebook. writes results to database
+ `scripts/silence_schema.sql`: schema for non noise regions
+ `AgglomerativeClusteringDTW.ipynb`: apply encoding and then agglomerative clustering using dtw distance
+ `clustering_schema.sql`: the schema definition of the output of the clustering algorithm