# Time Series Indexing

Builds a time series index by recursively partitioning the dataset
by picking two far away candidates and grouping all sequences around those.
The distance is measured using the dynamic time warping.

## Structure

+ data/: some test data
+ protos/: server interface defined by google remote procedure calls
+ scripts/: some scripts that help with the repository
+ test/: implements unit tests as well as integration tests
+ libindexing/: this is where the algorithm and main code lives
+ server/: implements the actual C++ database as a grpc server

## Server
First we need to run the script that generating the remote procedure call interface. We can simply run:

```
scripts/generate_proto.sh
```

which generates the interfaces for python and c++ as defined in the protobuf definition in protos/.

## Testing

### Server Testing
If you want to test the server implementation go into the scripts folder and run:

```
$ cd scripts
$ python replay_keogh.py
```

### Unitttests
I tested the base functions in IndexingUtil with unit tests
this includes mainly the tree construction and dtw code. Some
of the tests rely on randomness so those are a little shaky.
Actually most of them are not real unit tests but more a small test run.

### Performance testing
Mainly the actual index. I am testing the speedup of parallel computation
as well as the thread safty. The performance tests can be executed
using the ./benchmark tool. I suggest to disable logging for actual measurments.
I am also using this to test for memory leaks or parallelisation issues. Great bianry to run valgrind

```
valgrind -v --leak-check=full --show-leak-kinds=all --log-file=vg_benchmark_20230601.log ./benchmark
``` 

```
valgrind --tool=helgrind  --log-file=hg_benchmark_20230601_threads.log ./benchmark
```

```
cmake .. -D CMAKE_BUILD_TYPE=Debug
```

```
GLOG_minloglevel=1 GLOG_log_dir=. ./benchmark
```

## Compile

```
$ mkdir build; cd build
$ cmake ..
$ make
$ make test
```

## Requirements
+ [1] gtest
+ [2] glog
+ [3] gflags
+ [4] grpc