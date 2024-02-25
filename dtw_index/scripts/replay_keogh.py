import sys
sys.path.append('gen')

import random
import numpy as np
import grpc
import indexing_pb2
import indexing_pb2_grpc

DATA = "../data/gunpoint.tsv"

def keogh():
    x = np.genfromtxt(DATA)    
    vectors = x[:, 1:]
    labels = x[:, 0]
    return labels, vectors

def timeseries(ts):
    if len(ts.shape) == 1:
        length = ts.shape[0]
        dim = 1
    else:
        length, dim = ts.shape
    repeated = list(ts.flatten())
    return indexing_pb2.TimeSeries(ts=repeated, dim=dim, length=length)

def run():
    y, X = keogh()
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = indexing_pb2_grpc.TimeSeriesServiceStub(channel)
        for i in range(0, len(X)):
            ts = timeseries(X[i])
            response = stub.insert(ts)
            print(f"Insert Response: status = {response.status} id = {response.ts_id}")
        request = indexing_pb2.ReindexingRequest(n_samples = 150)
        response = stub.reindex(request)
        response = stub.save(indexing_pb2.SaveIndexRequest(name = "test_save"))
        response = stub.load(indexing_pb2.LoadIndexRequest(name = "test_save"))
        print(f"Reindexing Response: status = {response.response}")
        n_correct = 0
        for i in range(0, 100):
            qid = random.randint(0, len(y) - 1)
            query = timeseries(X[qid])
            response = stub.query(query)
            n = len(response.ids)
            c = 0
            for neighbor in response.ids:
                c += 1.0 if y[neighbor] == y[qid] else 0
            p = c / n
            print(f" ... p(correct) = {p}")
            if p > 0.5:
                n_correct += 1
        print(f"n_correct / 100 = {n_correct}")

        
if __name__ == "__main__":
    run()
