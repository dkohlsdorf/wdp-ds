# Collections of Scripts TO Construct Our Database

A collection of sql schemas and scripts to clean and ingest the original data

+ `ingest_annotations.py`: collect PVL and ENCODING files and join them into csv files for cloud sql import. It will also normalize the words for easier search, later.
+ `enc_schema.sql`: mysql, encoding schema for the wild dolphin data
+ `pvl_schema.sql`: mysql, schema for video log annotations
+ `audiofile_schema.sql`: mysql, schema for audio files to encoding
+ `stopwords.txt`: a collections of stopwords for keyword generation
+ `scripts/silence_schema.sql`: schema for non noise regions
+ `clustering_schema.sql`: the schema definition of the output of the clustering algorithm
