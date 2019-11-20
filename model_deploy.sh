MODEL=$1
echo "MODEL: $MODEL"

python db_scripts/ingest_clusters.py models/$MODEL/clusters.csv data/files.csv
gsutil cp models/$MODEL/embeddings_test.png gs://wdp-data/$MODEL/
gsutil cp models/$MODEL/*.h5 gs://wdp-data/$MODEL/
gsutil cp models/$MODEL/filters*.png gs://wdp-data/$MODEL/
gsutil cp models/$MODEL/km.p gs://wdp-data/$MODEL/
gsutil cp clusters.csv gs://wdp-data/$MODEL/

gcloud sql import csv wdp-data-science gs://wdp-data/$MODEL/clusters.csv \
       --database=wdp_ds --table=clustering_results
