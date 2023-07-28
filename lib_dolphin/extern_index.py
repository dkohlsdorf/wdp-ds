import numpy as np
import sys
import grpc

sys.path.append('lib_dolphin/gen')

import indexing_pb2
import indexing_pb2_grpc

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
            
def reindex(addr, name):
    with grpc.insecure_channel(addr) as channel:
        request = indexing_pb2.ReindexingRequest(n_samples = 1024)
        response = stub.reindex(request)
        stub = indexing_pb2_grpc.TimeSeriesServiceStub(channel)
        response = stub.save(indexing_pb2.SaveIndexRequest(name = "name"))
    
