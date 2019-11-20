import sys
import pandas as pd
import time
import datetime

NAME = "v2_lstm_v4"

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print('python ingest_clusters.py PATH_CLUSTERS PATH_FILES')
    path_cluster = sys.argv[1]
    path_files   = sys.argv[2]
    clusters = pd.read_csv(path_cluster, header=None)

    clusters['filename'] = clusters[1].apply(
        lambda name: name.replace("regions_", "").replace(".p", "")
    )
    clusters['algorithm'] = clusters[0].apply(lambda x: NAME)

    encodings = {}
    for line in open(path_files):
        cmp = line.strip().split(",")
        encodings[cmp[1]] = cmp[0]
    clusters['encoding'] = clusters['filename'].apply(lambda name: encodings.get(name))
    clusters = clusters.dropna(how='any')
    clusters['cluster_id'] = clusters[0]
    clusters['start'] = clusters[2]
    clusters['stop'] = clusters[3]
    clusters['created_at'] = clusters['filename'].apply(
        lambda x: datetime.datetime.fromtimestamp(time.time()).strftime(
            '%Y-%m-%d %H:%M:%S'))
    clusters = clusters[['encoding', 'filename', 'start', 'stop', 'algorithm',
                         'cluster_id', 'created_at']]
    clusters.to_csv('clusters.csv', header=None)
