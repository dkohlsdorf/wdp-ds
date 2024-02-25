#include "IndexingUtil.h"
#include <vector>
#include <cmath>
#include <random>
#include <glog/logging.h>
#include <cassert>

namespace tsidx {

int idx(int i, int j, int cols) {
  return i * cols + j;
}

float euclidean_at(const TimeSeries &x, const TimeSeries &y, int i, int j) {
  assert(x.dim == y.dim);
  int dim = x.dim;
  float distance = 0.0;
  for(int d = 0; d < dim; d++) {
    distance += pow(x.x[idx(i, d, dim)] - y.x[idx(j, d, dim)], 2);    
  }
  return distance;
}

float min3(float x, float y, float z) {
  float min = x;
  if(y < min) {
    min = y;
  }
  if(z < min) {
    min = z;
  }
  return min;
}
  
float dtw(const TimeSeries &x, const TimeSeries &y, float band_percentage) {
  int N = x.length;
  int M = y.length;
  int w = band_percentage * fmax(N, M);
  w = fmax(w, abs(N - M)) + 2;
  float *dp = new float[(N + 1) * (M + 1)];
  dp[idx(0, 0, M + 1)] = 0.0;
  for(int i = 1; i < N + 1; i++) {
    dp[idx(i, 0, M + 1)] = INFINITY;
  }
  for(int j = 1; j < M + 1; j++) {
    dp[idx(0, j, M + 1)] = INFINITY;
  }
  for(int i = 1; i < N + 1; i++) {
    for(int j = 1; j < M + 1; j++) {
      dp[idx(i, j, M + 1)] = euclidean_at(x, y, i - 1, j - 1);
      dp[idx(i, j, M + 1)] += min3(
	   dp[idx(i - 1, j - 1, M + 1)],
	   dp[idx(i - 1, j, M + 1)],
	   dp[idx(i, j - 1, M + 1)]);
    }
  }
  float result = dp[idx(N, M, M + 1)];
  delete[] dp;
  return result;
}

int select_rand(const TimeSeriesBatch& batch, const BatchRange& range, const BatchIndex& idx) {
  int N = range.stop - range.start;
  int i = rand() % N + range.start;
  return idx[i];
}

int select_far(const TimeSeriesBatch& batch, const BatchRange& range, const BatchIndex& idx, int comparison_sequence, float band_percentage) {
  int N = range.stop - range.start;
  float *cum_sum_distances = new float[N];

  float total = 0.0;
  for(int i = 0; i < N; i++) {    
    if(idx[i + range.start] != comparison_sequence) {
      cum_sum_distances[i] = pow(dtw(batch[comparison_sequence], batch[idx[i + range.start]], band_percentage), 2);
    } else {
      cum_sum_distances[i] = 0;
    }
    total += cum_sum_distances[i];
  }
  for(int i = 0; i < N; i++) {
    cum_sum_distances[i] = cum_sum_distances[i] / total;
    if(i > 0) {
      cum_sum_distances[i] += cum_sum_distances[i - 1];
    }
  }
  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());
  std::uniform_real_distribution<float>  distr(0.0, 1.0); 
  float p = distr(generator);
  int pos = 0;
  for(pos = 0; pos < N; pos++) {
    if(idx[pos + range.start] != comparison_sequence) {
      if(p <= cum_sum_distances[pos]) break;
    }
  }
  delete[] cum_sum_distances;
  return idx[pos + range.start];
}

int partition(const TimeSeriesBatch& batch,
	      const BatchRange& range, BatchIndex& idx,
	      int left, int right, float band_percentage) {  
  int l = range.start-1;
  for(int r = range.start; r < range.stop; r++) {
    float dleft = dtw(batch[left], batch[idx[r]], band_percentage);
    float dright = dtw(batch[right], batch[idx[r]], band_percentage);
    if(dleft <= dright) {
      l += 1;
      int tmp = idx[l];
      idx[l] = idx[r];
      idx[r] = tmp;
    }
  }
  return l;
}

void build_tree(const TimeSeriesBatch& batch, int bucket_size, float band_percentage, Node *root) {
  BatchIndex idx;
  for(int i = 0; i < batch.size(); i++) {
    idx.push_back(i);
  }
  BatchRange range;
  range.start = 0;
  range.stop = batch.size();

  root -> node_id = 0;

  TreeBuilderState cur;
  cur.current_node = root;
  cur.current_node -> span = batch.size();
  cur.range = range;
  cur.depth = 0;
  
  std::vector<TreeBuilderState> work = {cur};
  int id = 1;
  while(work.size() > 0) {       
    TreeBuilderState state = work.back();
    work.pop_back();
    int length = state.range.stop - state.range.start;
    if(length > bucket_size) {          
      LOG(INFO) << "#work: " << work.size() << " "
		<< "node_id: " << state.current_node -> node_id << " "
		<< "length: " << length << " "
		<< "depth: " << state.depth;       
      
      TreeBuilderState right_state, left_state;      
      int left = select_rand(batch, state.range, idx);      
      int right = select_far(batch, state.range, idx, left, band_percentage);
      int mid = partition(batch, state.range, idx, left, right, band_percentage);
      
      state.current_node -> left_ts = left;
      state.current_node -> right_ts = right;

      state.current_node -> left = new tsidx::Node();
      state.current_node -> right = new tsidx::Node();
      state.current_node -> left -> left   = NULL;
      state.current_node -> left -> right  = NULL;
      state.current_node -> right -> left  = NULL;
      state.current_node -> right -> right = NULL;

      
      right_state.current_node = state.current_node -> right;
      left_state.current_node = state.current_node -> left;

      left_state.current_node -> node_id = id;
      right_state.current_node -> node_id = id + 1;
      id += 2;

      left_state.range.start = state.range.start;
      left_state.range.stop = mid + 1;
      left_state.depth = state.depth + 1;
      left_state.current_node -> span = left_state.range.stop - left_state.range.start;
      
      right_state.range.start = mid + 1;
      right_state.range.stop = state.range.stop;
      right_state.depth = state.depth + 1;
      right_state.current_node -> span = right_state.range.stop - right_state.range.start;
      LOG(INFO) << "\t ... partitions: [" << state.range.start << " | "
		<< mid + 1 << " | "
		<< state.range.stop << "]\n"; 
      
      work.push_back(left_state);
      work.push_back(right_state);
    }
  }
}

int search(const TimeSeries &ts, const TimeSeriesBatch& batch, Node *node, float band_percentage) {
  LOG(INFO) << "\t ... " << node -> node_id << " " << ts.id; 
  if(node -> left == NULL && node -> right == NULL) {
    return node -> node_id;
  }
  if(node -> left == NULL) {
    return search(ts, batch, node -> right, band_percentage);
  }
  if(node -> right == NULL) {
    return search(ts, batch, node -> left, band_percentage);
  }

  float dleft = dtw(batch[node -> left_ts], ts, band_percentage);
  float dright = dtw(batch[node -> right_ts], ts, band_percentage);

  if(dleft < dright) {
    return search(ts, batch, node -> left, band_percentage);
  }
  return search(ts, batch, node -> right, band_percentage);   
}

int delete_tree(Node *node) {
  int l = 0;
  int r = 0;
  if(node -> left != NULL) {
    l = delete_tree(node -> left);
  }
  if(node -> right != NULL) {
    r = delete_tree(node -> right);
  }
  delete node;
  return l + r + 1;
}

void leaf_nodes(Node *node, std::string offset, std::vector<int> &leafs) { 
  if(node -> left == NULL && node -> right == NULL) {
    LOG(INFO) << offset << "Node:  " << node -> node_id << " |||" << node -> span << "|||";
    leafs.push_back(node -> node_id);
  } else {
    LOG(INFO) << offset << "Node:  " << node -> node_id << " |" << node -> span << "|";
  }
  if(node -> left != NULL) {
    leaf_nodes(node -> left, offset + "\t", leafs);
  }
  if(node -> right != NULL) {
    leaf_nodes(node -> right, offset + "\t", leafs);
  }  
}

int n_nodes(const Node *node) {
  if(node == NULL) {
    return 0;
  }
  return 1 + n_nodes(node -> left) + n_nodes(node -> right);
}
  
}
