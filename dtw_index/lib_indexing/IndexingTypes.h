#pragma once
#include<vector>

namespace tsidx {
  struct TimeSeries {    
    float *x;
    int length;
    int dim;
    int id;
  };

  struct BatchRange {
    int start;
    int stop;
  };

  struct Node {
    int node_id;
    int left_ts;
    int right_ts;  
    Node *left;
    Node *right;
    int span;
  };
  
  struct TreeBuilderState {
    Node* current_node;
    BatchRange range;  
    int depth;
  };
  
  using BatchIndex = std::vector<int>;
  using TimeSeriesBatch = std::vector<TimeSeries>;
}
