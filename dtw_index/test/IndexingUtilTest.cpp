#include "IndexingUtil.h"
#include "gtest/gtest.h"

#include <glog/logging.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

const std::string GUNPOINT_DATA = "../../data/gunpoint.tsv";
const int GUNPOINT_LEN = 150;

void gunpoint(tsidx::TimeSeriesBatch& data) {
  std::fstream data_file;
  data_file.open(GUNPOINT_DATA, std::ios::in); 
  std::string line;
  std::string cell;
  while (std::getline(data_file, line)) { 
    std::stringstream token_stream(line);
    int i = 0;
    tsidx::TimeSeries ts;
    ts.length = GUNPOINT_LEN;
    ts.dim = 1;
    ts.x = new float[ts.length];
    while(std::getline(token_stream, cell, '\t')) {
      if(i > 0) {
	float val = atof(cell.c_str());
	ts.x[i - 1] = val; 
      }
      i++;
    }
    data.push_back(ts);
  }       
  data_file.close(); 
}

void randomData(tsidx::TimeSeriesBatch& data, int n) {
  for(int i = 0; i < n; i++) {
    tsidx::TimeSeries x;
    x.length = 3;
    x.dim = 2;
    x.x = new float[2 * 3];
    for(int j = 0; j < 2 * 3; j += 1) {
      x.x[j] = rand() % 100;
    }
    data.push_back(x);
  }
}

void testData(tsidx::TimeSeriesBatch& data) {
  tsidx::TimeSeries v, w, x, y;
  x.length = 3;
  y.length = 4;
  v.length = 5;
  w.length = 6;

  x.dim = 2;
  y.dim = 2;
  v.dim = 2;
  w.dim = 2;
  
  x.x = new float[2 * 3];
  y.x = new float[2 * 4];
  v.x = new float[2 * 5];
  w.x = new float[2 * 6];

  for(int i = 0; i < 2 * 3; i+=2) {
    x.x[i] = i;
    x.x[i + 1] = i;
  }
  for(int i = 0; i < 2 * 4; i+=2) {
    y.x[i] = i + 3;
    y.x[i + 1] = i + 3;
  }
  for(int i = 0; i < 2 * 5; i+=2) {
    v.x[i] = i + 6;
    v.x[i + 1] = i + 6;
  }
  for(int i = 0; i < 2 * 6; i+=2) {
    w.x[i] = i + 9;
    w.x[i + 1] = i + 9;
  }
  
  data.push_back(x);
  data.push_back(y);
  data.push_back(v);
  data.push_back(w);
}

int max_depth(tsidx::Node* node, int depth, std::string offset) {
  LOG(INFO) << offset << "Node:  " << node -> node_id << " " << depth;
  if(node -> left == NULL and node -> right == NULL) {
    return depth;
  }   

  int l, r = 0;
  if(node -> left != NULL) {
    l = max_depth(node -> left, depth + 1, offset + " ");
  }
  if(node -> right != NULL) {
    r = max_depth(node -> right, depth + 1, offset + " ");
  }
  return fmax(l,r);
}

TEST(IndexingUtilsTest, indexing) {
  int numberCols = 2;
  ASSERT_EQ(tsidx::idx(0, 0, numberCols), 0);
  ASSERT_EQ(tsidx::idx(0, 1, numberCols), 1);
  ASSERT_EQ(tsidx::idx(1, 0, numberCols), 2); // i * 2 + 0
  ASSERT_EQ(tsidx::idx(1, 1, numberCols), 3); // i * 2 + 1
}

TEST(IndexingUtilsTest, min3) {
  ASSERT_EQ(tsidx::min3(1,2,3), 1);
  ASSERT_EQ(tsidx::min3(3,3,3), 3);
  ASSERT_EQ(tsidx::min3(-1,-2,-3), -3); 
}

TEST(IndexingUtilsTest, euclidean) {
  tsidx::TimeSeriesBatch batch;
  testData(batch);
  int N = batch[0].length > batch[1].length ? batch[1].length : batch[0].length;
  for(int i = 0; i < N; i++) {
    float dist = tsidx::euclidean_at(batch[0], batch[1], i, i);
    ASSERT_EQ(dist, 18);
  }
  for(int i = 0; i < batch.size(); i++) {
    delete[] batch[i].x;
  }
}

TEST(IndexingUtilsTest, dtw) {
  tsidx::TimeSeriesBatch batch;
  testData(batch);
  ASSERT_EQ(dtw(batch[0], batch[1], 1.0), 90);
  ASSERT_EQ(dtw(batch[0], batch[1], 0.5), 90);
  ASSERT_EQ(dtw(batch[0], batch[1], 0.1), 90);
  for(int i = 0; i < batch.size(); i++) {
    delete[] batch[i].x;
  }
}


TEST(IndexingUtilsTest, select_rand) {
  tsidx::TimeSeriesBatch batch;
  testData(batch);
  
  tsidx::BatchRange rng;
  rng.start = 1;
  rng.stop = 3;

  tsidx::BatchIndex idx = {3,2,0,1};
  for(int i = 0; i < 100; i++) {
    ASSERT_NE(select_rand(batch, rng, idx), 3);
    ASSERT_NE(select_rand(batch, rng, idx), 1);
  }
  for(int i = 0; i < batch.size(); i++) {
    delete[] batch[i].x;
  }
}

TEST(IndexingUtilsTest, select_far) {
  tsidx::TimeSeriesBatch batch;
  testData(batch);
  
  tsidx::BatchRange rng;
  rng.start = 1;
  rng.stop = 3;

  tsidx::BatchIndex idx = {2,3,1,0};
  int n1 = 0;
  int n3 = 0;
  for(int i = 0; i < 512; i++) {
    int k = select_far(batch, rng, idx, 0, 1.0);
    if(k == 1) n1++;
    if(k == 3) n3++;
  }

  ASSERT_GT(n3, n1);
  ASSERT_GT(n3, 0);
  ASSERT_GT(n1, 0);
  for(int i = 0; i < batch.size(); i++) {
    delete[] batch[i].x;
  }
}


TEST(IndexingUtilsTest, partition) {
  tsidx::TimeSeriesBatch batch;
  testData(batch);

  tsidx::BatchRange rng;
  rng.start = 0;
  rng.stop = 4;

  tsidx::BatchIndex idx = {0, 3, 1, 2};
  tsidx::BatchIndex expected_idx = {0, 1, 3, 2};
  // sequences [[1,2,3...], [4, 5, 6 ...], [10, 11, 12 ... ], [13, 14, 15, ...]]
  // left sequnces [4,5,6 ... ], right: [5,6,7],
  // idx [0, 1] sequences closest to [4, 5, 6]
  // idx [2, 3] sequences closest to [5,6,7]
  int mid = partition(batch, rng, idx, 1, 2, 1.0);
  ASSERT_EQ(mid, 1);  
  for(int i = 0; i < 4; i++) {    
    ASSERT_EQ(idx[i], expected_idx[i]);
  }
  for(int i = 0; i < batch.size(); i++) {
    delete[] batch[i].x;
  }
}

TEST(IndexingUtilsTest, buildTree) {
  tsidx::TimeSeriesBatch batch;
  randomData(batch, 100);
  tsidx::Node* root = new tsidx::Node();
  build_tree(batch, 10, 1.0, root);  
  int depth = max_depth(root, 0, " ");  
  ASSERT_TRUE(depth <= 10);  
  for(int i = 0; i < batch.size(); i++) {
    delete[] batch[i].x;
  }
  delete_tree(root);
}

TEST(IndexingUtilsTest, searchTree) {  
  tsidx::TimeSeriesBatch batch;
  gunpoint(batch);
  tsidx::Node* root = new tsidx::Node();
  build_tree(batch, 25, 0.1, root);    
  int n = 0;
  int correct = 0;

  std::vector<int> leafs;
  leaf_nodes(root, "", leafs);
  std::map<int, int> counts;
  for(int query = 0; query < batch.size(); query++) {
    tsidx::TimeSeries ts = batch[query];
    int min_dist_ts = 0;
    float min_dist = INFINITY;
    for(int i = 0; i < batch.size(); i++) {      
      if(i != query) {
	float dist = dtw(ts, batch[i], 0.75);
	if(dist < min_dist) {
	  min_dist = dist;
	  min_dist_ts = i;
	}
      }
    }

    int idx_query = search(ts, batch, root, 0.75);
    int idx_nn = search(batch[min_dist_ts], batch, root, 0.75);
    counts[idx_query] += 1;
    n += 1;
    if(idx_query == idx_nn) correct += 1;    
  }
  float acc = float(correct) / n;
  for(const auto& [k, v] : counts) {
    LOG(INFO) << "#found in bucket: " << k << " = " << v;
  }
  LOG(INFO) << "Accuracy: " << acc << " = " << correct << "/" << n;
  ASSERT_GT(acc, 0.8);
  for(int i = 0; i < batch.size(); i++) {
    delete[] batch[i].x;
  }
  delete_tree(root);
}

TEST(IndexingUtilsTest, deleteTree) {
  tsidx::TimeSeriesBatch batch;
  randomData(batch, 100);
  tsidx::Node* root = new tsidx::Node();
  build_tree(batch, 10, 1.0, root);  
  int n = delete_tree(root);
  // TODO could be uninitialized
  ASSERT_GE(n, 10);
  for(int i = 0; i < batch.size(); i++) {
    delete[] batch[i].x;
  }
}

TEST(IndexingUtilsTest, leafNodes) {
  tsidx::TimeSeriesBatch batch;
  randomData(batch, 100);
  tsidx::Node* root = new tsidx::Node();
  build_tree(batch, 10, 1.0, root);  

  std::vector<int> leafs;
  leaf_nodes(root, "", leafs);
  int n = leafs.size();
  ASSERT_GE(leafs.size(), 10);
  for(int i = 0; i < batch.size(); i++) {
    delete[] batch[i].x;
  }
  delete_tree(root);
}

TEST(IndexingUtilsTest, nNodes) {
  tsidx::TimeSeriesBatch batch;
  randomData(batch, 100);
  tsidx::Node* root = new tsidx::Node();
  build_tree(batch, 10, 1.0, root);  

  ASSERT_GE(n_nodes(root), 25);
}
