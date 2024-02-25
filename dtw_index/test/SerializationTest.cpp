#include "Serialization.h"
#include "IndexingUtil.h"
#include "gtest/gtest.h"
#include <glog/logging.h>

void timeseries(tsidx::TimeSeries &ts) {
  ts.id = 0;
  ts.length = 5;
  ts.dim = 2;
  ts.x = new float[ts.dim * ts.length];  
  for(int i = 0; i < ts.length; i++) {
    for(int j = 0; j < ts.dim; j++) {
      ts.x[i * ts.dim + j] = i * ts.dim + j;
    }
  }
}

void tree(tsidx::Node *node) {
  node -> node_id = 0;
  node -> left_ts = 0;
  node -> right_ts = 1;
  node -> span = 2;

  node -> left = new tsidx::Node;
  node -> left -> node_id = 1;
  node -> left -> left_ts  = 1;
  node -> left -> right_ts = 2;
  node -> left -> span     = 3;
  node -> left -> left  = NULL;
  node -> left -> right = NULL;

  node -> right = new tsidx::Node;
  node -> right -> node_id = 2;
  node -> right -> left_ts  = 4;
  node -> right -> right_ts = 5;
  node -> right -> span     = 6;
  node -> right -> left  = NULL;
  node -> right -> right = NULL;

  node -> right -> right = new tsidx::Node;
  node -> right -> right -> node_id = 3;
  node -> right -> right -> left_ts  = 7;
  node -> right -> right -> right_ts = 8;
  node -> right -> right -> span     = 9;
  node -> right -> right -> left  = NULL;
  node -> right -> right -> right = NULL; 
}

bool compare_tree(tsidx::Node *a, tsidx::Node *b) {  
  if(a == NULL && b == NULL) return true;  
  if(a != NULL && b == NULL) {
    LOG(INFO) << "a is not null b is";
    return false;
  }
  if(a == NULL && b != NULL) {
    LOG(INFO) << "a null b is not";
    return true;
  }

  LOG(INFO) << "A: "    
	    << a -> node_id << " "
	    << a -> left_ts << " "
	    << a -> right_ts << " "
	    << a -> span << " ";
  LOG(INFO) << "B: "
	    << b -> node_id << " "
	    << b -> left_ts << " "
	    << b -> right_ts << " "
	    << b -> span << " ";


  if(a -> node_id  != b -> node_id) return false;
  if(a -> left_ts  != b -> left_ts) return false;
  if(a -> right_ts != b -> right_ts) return false;
  if(a -> span     != b -> span) return false; 

  return compare_tree(a -> left, b -> left)
    && compare_tree(a -> right, b -> right);
}

TEST(SerializationTest, n_bytes) {
  tsidx::TimeSeries ts;
  timeseries(ts);
  // header: [4 byte id, 4 byte length, 4 byte size]
  int header = 12;
  // dim * len * float = 10 * 4
  int size = 40;
  int expected = header + size;
  int actual = n_bytes(ts);
  ASSERT_EQ(actual, expected);
  delete[] ts.x;
}

TEST(SerializationTest, de_serialize) {
  tsidx::TimeSeries ts;
  tsidx::TimeSeries ts2;
  timeseries(ts);

  int n = n_bytes(ts);
  unsigned char* ts_ser = new unsigned char[n];
  int n_ser = serialize_ts(0, ts, ts_ser);
  int n_des = deserialize_ts(0, ts2, ts_ser);
  ASSERT_EQ(n_ser, n);
  ASSERT_EQ(n_des, n);

  ASSERT_EQ(ts.id, ts2.id);
  ASSERT_EQ(ts.length, ts2.length);
  ASSERT_EQ(ts.dim, ts2.dim);
  for(int i = 0; i < ts.dim * ts.length; i++) {
    ASSERT_EQ(ts.x[i], ts2.x[i]);
  }    
  delete[] ts2.x;
  delete[] ts.x;
  delete[] ts_ser;
}

TEST(SerializationTest, de_serialize_tree) {
  tsidx::Node *node = new tsidx::Node;
  tsidx::Node *node2 = new tsidx::Node; 

  // TODO: Test this
  std::map<int, int> leaf_map {
    {1, 0},
    {3, 1}
  };
  std::vector<std::vector<int>> buckets;
  buckets.push_back({1,2,3});
  buckets.push_back({4,5,6,7});

  std::map<int, int> leaf_map2;
  std::vector<std::vector<int>> buckets2;
  
  tree(node);
  ASSERT_FALSE(node == NULL);

  int n = tsidx::n_bytes(4, 2, 7);
  LOG(INFO) << "n_bytes: " << n; 
  unsigned char* buffer = new unsigned char[n];
  
  serialize_tree(node, leaf_map, buckets, buffer);
  deserialize_tree(node2, leaf_map2, buckets2, buffer);
  ASSERT_TRUE(compare_tree(node, node2));

  for(const auto &[leaf, i] : leaf_map) {
    ASSERT_EQ(leaf_map2[leaf], i);
  }

  for(int i = 0; i < buckets.size(); i++) {
    std::vector<int> b1 = buckets[i];
    std::vector<int> b2 = buckets2[i];
    for(int j = 0; j < buckets[i].size(); j++) {
      int x = b1[j];
      int y = b2[j];      
      LOG(INFO) << x << " " << y;
      ASSERT_EQ(x, y);
    }
  }
  
  tsidx::delete_tree(node);
  tsidx::delete_tree(node2);
}
