#include "TimeSeriesIndex.h"
#include "IndexingUtil.h"
#include "Serialization.h"

#include <glog/logging.h>
#include <cmath>
#include <set>

#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include <fstream>

namespace tsidx {

template <typename T>
void ThreadsafeCollection<T>::insert(T x) {
  std::unique_lock lock(mutex);
  collection.push_back(x);
}
  
template <typename T>
std::vector<T> ThreadsafeCollection<T>::get() {
  std::shared_lock lock(mutex);
  return std::vector<T>(collection);
}

template <typename T>
int ThreadsafeCollection<T>::size() {
  std::shared_lock lock(mutex);
  return collection.size();
}


const char* IDX_BIN = "/idx.bin";
const char* IDX_TS = "/ts_bucket_idx.bin";

TimeSeriesIndex::TimeSeriesIndex(int n_buckets,
				 int bucket_size,
				 float band_percentage) {
  for(int i = 0; i < n_buckets; i++) {
    timeseries.push_back(new tsidx::ThreadsafeCollection<TimeSeries>());
  }
  buckets.push_back(new tsidx::ThreadsafeCollection<int>());

  this -> n_buckets = n_buckets;
  this -> band_percentage = band_percentage;
  this -> bucket_size = bucket_size;  
  status = IDX_READY;
  n = 0;
  root = NULL;
}

void TimeSeriesIndex::load(std::string name) {
  int i = 0;
  while(true) {
    std::stringstream ss_name;
    ss_name << "./" << name << "/ts_bucket" << i << ".bin";    
    LOG(INFO) << "reading timeseries: " << ss_name.str();
    std::ifstream file_ts;
    file_ts.open(ss_name.str(), std::fstream::binary);
    
    if(!file_ts.good()) break;
    ThreadsafeCollection<TimeSeries> *collection = new ThreadsafeCollection<TimeSeries>();
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file_ts), {});
    int len = buffer.size();
    int start = 0;
    while(start < len) {
      TimeSeries ts;
      start = deserialize_ts(start, ts, buffer.data());
      collection -> insert(ts);
    }
    timeseries.push_back(collection);
    i++;
  }
  
  std::stringstream ss_name_idx;
  ss_name_idx << "./" << name << IDX_BIN;
  std::ifstream file_idx;
  file_idx.open(ss_name_idx.str(), std::fstream::binary);
  std::vector<unsigned char> buffer_idx(std::istreambuf_iterator<char>(file_idx), {});
  
  std::vector<std::vector<int>> bb;
  root = new Node;
  deserialize_tree(root, leaf_map, bb, buffer_idx.data());

  LOG(INFO) << "CONVERTING Buckets";
  for(const auto& b : bb) {
    ThreadsafeCollection<int> *collection = new ThreadsafeCollection<int>();
    for(const auto id : b) {
      collection -> insert(id);
    }
    buckets.push_back(collection);
  }


  LOG(INFO) << "reading timeseries";
  std::stringstream ss_name;
  ss_name << "./" << name << IDX_TS;    
  std::ifstream file_ts;
  file_ts.open(ss_name.str(), std::fstream::binary);
  std::vector<unsigned char> buffer_ts(std::istreambuf_iterator<char>(file_ts), {});
  int len = buffer_ts.size();
  int start = 0;
  while(start < len) {
    TimeSeries ts;
    start = deserialize_ts(start, ts, buffer_ts.data());
    LOG(INFO) << "read ts from: " << start << "/" << len;
    indexing_batch.push_back(ts);
  }
  
  LOG(INFO) << "Done reading";
  status = IDX_READY;
  n = i;
  LOG(INFO) << "Done";
}
  
TimeSeriesIndex::~TimeSeriesIndex() {
  std::unique_lock lock(mutex);
  LOG(INFO) << "Deconstruct";
  delete_tree(root);  
  for(int i = 0; i < timeseries.size(); i++) {
    delete timeseries[i];      
  }
  for(int i = 0; i < buckets.size(); i++) {
    delete buckets[i];
  }
}
  
int TimeSeriesIndex::insert(TimeSeries& ts) {
  if(status == IDX_RUN) {
    LOG(INFO) << "Currently Indexing, No insertion";
    return IDX_RUN;
  }
  
  std::shared_lock lock(mutex);
  if(ts.id < 0) {
    ts.id = n;
  }
  timeseries[n % n_buckets] -> insert(ts);
  if(root == NULL) {
    buckets[0] -> insert(ts.id);
  } else {
    int node = search(ts, indexing_batch, root, band_percentage);      
    int bucket = leaf_map[node];
    buckets[bucket] -> insert(ts.id);
  }
  n++;
  return IDX_READY;  
}

int TimeSeriesIndex::search_idx(const TimeSeries& ts, std::vector<int>& nearest) {
  if(status == IDX_RUN || root == NULL) {
    LOG(INFO) << "Currently Indexing, No insertion";
    return IDX_RUN;
  }
  std::shared_lock lock(mutex);
  int node = search(ts, indexing_batch, root, band_percentage);
  int bucket = leaf_map[node];
  for(const auto& i : buckets[bucket] -> get()) {
    nearest.push_back(i);
  }
  return status;
}
 
int TimeSeriesIndex::reindex(int n_samples) {  
  std::unique_lock lock(mutex);                                                               
  LOG(INFO) << "Reindex";
  status = IDX_RUN;  

  LOG(INFO) << "Delete all";
  if(root != NULL) {
    LOG(INFO) << "root not null";
    delete_tree(root);
    leaf_map.clear();
    indexing_batch.clear();   
  }
  LOG(INFO) << "Delete buckets";
  for(int i = 0; i < buckets.size(); i++) {
    delete buckets[i];
  }
  buckets.clear();

  LOG(INFO) << "#ts_buckets: " << timeseries.size();  
  LOG(INFO) << "Find Samples: " << n_samples << " / " << n;
  std::set<int> closed;
  for(int i = 0; i < n_samples; i++) {
    int bucket_id = rand() % n_buckets;
    int ts_i = rand() % timeseries[bucket_id] -> size();
    int id = timeseries[bucket_id] -> get()[ts_i].id; 
    while(closed.find(id) != closed.end()) {
      bucket_id = rand() % n_buckets;
      ts_i = rand() % timeseries[bucket_id] -> size();
      id = timeseries[bucket_id] -> get()[ts_i].id; 
    }
    closed.insert(id);
    indexing_batch.push_back(timeseries[bucket_id] -> get()[ts_i]);
  }

  LOG(INFO) << "Build Tree: bucket = " << bucket_size << " batch = " << indexing_batch.size();
  root = new Node;
  build_tree(indexing_batch, bucket_size, band_percentage, root);  
  
  std::vector<int> leafs;
  leaf_nodes(root, "", leafs);
  LOG(INFO) << "n_buckets: " << leafs.size();
  for(int i = 0; i < leafs.size(); i++) {
    buckets.push_back(new tsidx::ThreadsafeCollection<int>());
    leaf_map[leafs[i]] = i;
  }

  // insert all
  for(const auto& bucket : timeseries) {
    TimeSeriesBatch sequences = bucket -> get();
    for(const auto& ts : sequences) {
      int node = search(ts, indexing_batch, root, band_percentage);      
      int bucket = leaf_map[node];
      buckets[bucket] -> insert(ts.id);
      LOG(INFO) << "insert: " << ts.id << " / " << n;      
    }
  }
  LOG(INFO) << "DONE INDEXING";
  status = IDX_READY;
  return IDX_READY;
}

int TimeSeriesIndex::save(std::string name) {
  std::unique_lock lock(mutex);

  std::stringstream ss;
  ss << "./" << name;
  int status;
  status = mkdir(ss.str().c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  LOG(INFO) << "Save index to folder: " << name << ": " << status;

  status = IDX_RUN;  
  // write out all time series
  for(int i = 0; i < timeseries.size(); i++) {
    int batch_size = 0;
    std::vector<TimeSeries> batch = timeseries[i] -> get();
    for(const auto& ts : batch) {
      batch_size += n_bytes(ts);
    }
    unsigned char *buf = new unsigned char[batch_size];
    int start = 0;
    for(const auto& ts : batch) {
      start = serialize_ts(start, ts, buf);
    }
    std::stringstream ss_name;
    ss_name << "./" << name << "/ts_bucket" << i << ".bin";    
    LOG(INFO) << ss_name.str();
    std::fstream file_ts;
    file_ts.open(ss_name.str(), std::fstream::out | std::fstream::binary);
    file_ts.write(reinterpret_cast<char const*>(buf), batch_size);
    file_ts.close();
    delete buf;
  }
  
  // write out indexing batch 
  int batch_size = 0;
  for(const auto& ts : indexing_batch) {
    batch_size += n_bytes(ts);
  }
  unsigned char *buf = new unsigned char[batch_size];
  int start = 0;
  for(const auto& ts : indexing_batch) {
    start = serialize_ts(start, ts, buf);
  }
  std::stringstream ss_name;
  ss_name << "./" << name << IDX_TS;    
  LOG(INFO) << ss_name.str();
  std::fstream file_ts;
  file_ts.open(ss_name.str(), std::fstream::out | std::fstream::binary);
  file_ts.write(reinterpret_cast<char const*>(buf), batch_size);
  file_ts.close();
  delete buf;

  // write out tree
  int n_nodes = tsidx::n_nodes(root);
  batch_size = n_bytes(n_nodes, buckets.size(), n);
  buf = new unsigned char[batch_size];
  std::vector<std::vector<int>> bucks;
  for(int i = 0; i < buckets.size(); i++) {
    bucks.push_back(buckets[i] -> get());
  }
  std::stringstream ss_name_idx;
  ss_name_idx << "./" << name << IDX_BIN;
  
  LOG(INFO) << ss_name_idx.str();
  serialize_tree(root, leaf_map, bucks, buf);
  std::fstream file_idx;
  file_idx.open(ss_name_idx.str(),  std::fstream::out | std::fstream::binary);
  file_idx.write(reinterpret_cast<char const*>(buf), batch_size);
  file_idx.close();
  delete buf;
  status = IDX_READY;
  return 0;
}
  
template class ThreadsafeCollection<int>;
template class ThreadsafeCollection<TimeSeries>;
  
}
