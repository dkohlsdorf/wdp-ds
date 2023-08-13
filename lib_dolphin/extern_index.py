import numpy as np
import sys
import grpc

sys.path.append('lib_dolphin/gen')

import indexing_pb2
import indexing_pb2_grpc

from collections import defaultdict

SIL = True

def timeseries(ts):
    if len(ts.shape) == 1:
        length = ts.shape[0]
        dim = 1
    else:
        length, dim = ts.shape
    ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
    repeated = list(ts.flatten())
    if not SIL:
        print(repeated[0:100], dim, length)
    return indexing_pb2.TimeSeries(ts=repeated, dim=dim, length=length)


def insert_all(ts, addr):
    ids = []
    with grpc.insecure_channel(addr) as channel:
        stub = indexing_pb2_grpc.TimeSeriesServiceStub(channel)
        for i in range(0, len(ts)):
            x = timeseries(ts[i])
            response = stub.insert(x)
            print(f"Insert Response: status = {response.status} id = {response.ts_id}")
            ids.append(response.ts_id)
    return ids


def reindex(addr, name):
    with grpc.insecure_channel(addr) as channel:
        stub = indexing_pb2_grpc.TimeSeriesServiceStub(channel)
        request = indexing_pb2.ReindexingRequest(n_samples = 1024)
        response = stub.reindex(request)
        print("reindexing done")
        response = stub.save(indexing_pb2.SaveIndexRequest(name = "name"))
        print("saving done")


def load(addr, name):
    with grpc.insecure_channel(addr) as channel:
        stub = indexing_pb2_grpc.TimeSeriesServiceStub(channel)        
        response = stub.load(indexing_pb2.LoadIndexRequest(name = name))
        

def find_relaxed(addr, name, sequences, inverted_idx, k = 10):
    with grpc.insecure_channel(addr) as channel:
        stub = indexing_pb2_grpc.TimeSeriesServiceStub(channel)        
        not_there = 0
        there = 0
        found = defaultdict(int)
        not_found = set()
        is_found = set()
        for sequence in sequences:
            query = timeseries(sequence)
            response = stub.query(query)
            for neighbor in response.ids:
                if neighbor not in inverted_idx:
                    not_there += 1
                    not_found.add(neighbor)
                else:
                    i = inverted_idx[neighbor]
                    found[i] += 1
                    there += 1
                    is_found.add(neighbor)
            neighbors = sorted(found.items(), key=lambda x: -x[1])[:k]

    print(f"{there} / {not_there}: {len(not_found)} / {len(is_found)}")
    return [(1.0 / n, k) for k, n in neighbors]
