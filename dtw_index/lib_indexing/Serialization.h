#pragma once

#include "IndexingTypes.h"
#include <map>

namespace tsidx {

  int n_bytes(const TimeSeries& ts);

  int n_bytes(int n_nodes, int n_buckets, int n_ids);
  
  int serialize_ts(int start, const TimeSeries& ts, unsigned char* buffer);

  int deserialize_ts(int start, TimeSeries& ts, unsigned char* buffer);

  void serialize_tree(const Node *root,
		      const std::map<int, int>& leaf_map,
		      const std::vector<std::vector<int>>& buckets,
		      unsigned char* buffer);

  void deserialize_tree(Node *root,
			std::map<int, int>& leaf_map,
			std::vector<std::vector<int>>& buckets,
			unsigned char* buffer);

}
