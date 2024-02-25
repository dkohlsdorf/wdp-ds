#pragma once
#include "IndexingTypes.h"
#include <shared_mutex>
#include <mutex>
#include <vector>
#include <map>
#include <atomic>

namespace tsidx {
  /**
   * Locking is hard so here are some hints:
   * Hints:
   *  [1] https://en.cppreference.com/w/cpp/thread/shared_mutex
   *  [2] https://en.cppreference.com/w/cpp/atomic/atomic
   */

  using Bucket = std::vector<int>;

  const int IDX_READY = 0;
  const int IDX_RUN = 1;

  template<typename T>
  class ThreadsafeCollection {
  public:
    ThreadsafeCollection() = default;
    void insert(T x);
    std::vector<T> get();
    int size();
  private:
    std::vector<T> collection;
    mutable std::shared_mutex mutex;
  };

  class TimeSeriesIndex {
  public:   
    TimeSeriesIndex(int n_buckets, int bucket_size,
		    float band_percentage);
    ~TimeSeriesIndex();

    void load(std::string name);
    int insert(TimeSeries& ts);
    int search_idx(const TimeSeries& ts, std::vector<int> &nearest);
    int reindex(int n_samples);
    int save(std::string name);    

    int n_buckets;
    int bucket_size;
    float band_percentage;
  private:    
    Node *root;
    TimeSeriesBatch indexing_batch;
    std::map<int, int> leaf_map;
    
    std::atomic_int status;
    mutable std::shared_mutex mutex;

    std::atomic_int n;
    std::vector<ThreadsafeCollection<TimeSeries>*> timeseries;   
    std::vector<ThreadsafeCollection<int>*> buckets;    
  };
}
