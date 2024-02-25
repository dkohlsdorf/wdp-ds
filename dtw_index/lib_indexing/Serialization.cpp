#include "Serialization.h"
#include <cstdlib>
#include <glog/logging.h>

namespace tsidx {

  template <typename T>
  int write(int start, T value, unsigned char *buf) {
    unsigned char *ptr = (unsigned char *) &value;
    unsigned int i;
    for (i = 0; i < sizeof(T); i++) {
      buf[start + i] = ptr[i];
    }
    return start + i;
  }

  template <typename T>
  int read(int start, unsigned char *buf, T &val) {
    unsigned char *ptr = (unsigned char *)&val;
    unsigned int i;
    for (i = 0; i < sizeof(T); i++) {
      ptr[i] = buf[start + i];
    }
    return start + i;
  }
  
  
  // Header: [id|len|dim]
  const int HEADER = sizeof(int) * 3;
  
  int n_bytes(const TimeSeries& ts) {    
    int data_sze = ts.length * ts.dim * sizeof(float);
    return HEADER + data_sze;
  }

  int n_bytes(int n_nodes, int n_buckets, int n_ids) {
    // id, left_ts_id, right_ts_id, left_node_id, right_node_id, span
    int node_sze = sizeof(int) * 6;
    // nodes * nodesize + n_nodes
    int nodes_sze = (node_sze * n_nodes) + sizeof(size_t);

    // leaf bucket n_ids
    int header = sizeof(int) + sizeof(int) + sizeof(size_t);    

    // n_buckets + [header1 ... headern] + [id1 ... idn]
    int leaf_sze = sizeof(size_t)
      + header * n_buckets
      + n_ids * sizeof(int);
    return nodes_sze + leaf_sze;
  }
  
  int serialize_ts(int start,
		   const TimeSeries& ts, unsigned char* buffer) {
    start = write(start, ts.id, buffer);
    start = write(start, ts.length, buffer);
    start = write(start, ts.dim, buffer);
    for(int i = 0; i < ts.length * ts.dim; i++) {
      start = write(start, ts.x[i], buffer);
    }
    return start;
  }
  
  int deserialize_ts(int start,
		      TimeSeries& ts, unsigned char* buffer) {      

    start = read(start, buffer, ts.id);
    start = read(start, buffer, ts.length);
    start = read(start, buffer, ts.dim);
    ts.x = new float[ts.length * ts.dim];
    for(int i = 0; i < ts.length * ts.dim; i++) {
      start = read(start, buffer, ts.x[i]);
    }
    return start;
  }

  int serialize_node(int start, const Node *node, unsigned char *buffer) {
    // id, left_ts_id, right_ts_id, left_node_id, right_node_id, span
    start = write(start, node -> node_id, buffer);
    start = write(start, node -> left_ts, buffer);
    start = write(start, node -> right_ts, buffer);

    int left  = node -> left  == NULL ? -1 : node -> left -> node_id;
    int right = node -> right == NULL ? -1 : node -> right -> node_id;
    start = write(start, left, buffer);
    start = write(start, right, buffer);
    start = write(start, node -> span, buffer);
    return start;
  }

  int deserialize_node(int start,
		       Node *node,
		       int &left, int &right,
		       unsigned char *buffer) {
    // id, left_ts_id, right_ts_id, left_node_id, right_node_id, span
    start = read(start, buffer, node -> node_id);
    start = read(start, buffer, node -> left_ts);
    start = read(start, buffer, node -> right_ts);
    start = read(start, buffer, left);
    start = read(start, buffer, right);
    start = read(start, buffer, node -> span);
    return start;
  }
  
  void serialize_tree(const Node *root,
		      const std::map<int, int>& leaf_map,
		      const std::vector<std::vector<int>>& buckets,
		      unsigned char* buffer) {
    
    std::vector<const Node*> nodes;
    std::vector<const Node*> work;
    work.push_back(root);

    while(work.size() > 0) {
      const Node *node = work.back();

      int left  = node -> left  == NULL ? -1 : node -> left -> node_id;
      int right = node -> right == NULL ? -1 : node -> right -> node_id;
      LOG(INFO) << " ... writing: "
		<< node -> node_id << " "
		<< node -> left_ts << " "
		<< node -> right_ts << " "
		<< node -> span << " "
		<< left << " "
		<< right;
	    
      work.pop_back();      
      nodes.push_back(node);
      if(node -> left != NULL) {
	work.push_back(node -> left);
      }
      if(node -> right != NULL) {
	work.push_back(node -> right);
      }      
    }

    int start = 0;
    start = write(start, nodes.size(), buffer);
    for(const Node *node : nodes) {
      start = serialize_node(start, node, buffer);
    }

    start = write(start, buckets.size(), buffer);
    for(const auto &[leaf, i] : leaf_map) {
      size_t sze = buckets[i].size();
      start = write(start, leaf, buffer);
      start = write(start, i, buffer);
      start = write(start, sze, buffer);
      for(const auto &ts_id : buckets[i]) {
	start = write(start, ts_id, buffer);
      } 
    }
  }

  void deserialize_tree(Node *root,
			std::map<int, int>& leaf_map,
			std::vector<std::vector<int>>& buckets,
			unsigned char* buffer) {
    int start = 0;
    size_t n_nodes;
    start = read(start, buffer, n_nodes);
    LOG(INFO) << "reading: " << n_nodes << " nodes";
    std::vector<int> lefts(n_nodes);
    std::vector<int> rights(n_nodes);
    std::vector<Node*> nodes(n_nodes);
    for(int i = 0; i < n_nodes; i++) {
      Node *node = new Node;
      int left, right;
      start = deserialize_node(start, node, left, right, buffer);      
      LOG(INFO) << " ... reading: "
		<< node -> node_id << " "
		<< node -> left_ts << " "
		<< node -> right_ts << " "
		<< node -> span << " "
		<< left << " "
		<< right;
      lefts[node -> node_id] = left;
      rights[node -> node_id] = right;
      nodes[node -> node_id] = node;
    }
    for(int i = 0; i < n_nodes; i++) {
      if(lefts[i] == -1)
	nodes[i] -> left = NULL;
      else 
	nodes[i] -> left = nodes[lefts[i]];
      if(rights[i] == -1) 
	nodes[i] -> right = NULL;
      else
	nodes[i] -> right = nodes[rights[i]];     
    }
    
    root -> node_id  = nodes[0] -> node_id;
    root -> left_ts  = nodes[0] -> left_ts;
    root -> right_ts = nodes[0] -> right_ts;
    root -> span     = nodes[0] -> span;
    root -> left     = nodes[0] -> left;
    root -> right    = nodes[0] -> right;

    size_t n_buckets;
    start = read(start, buffer, n_buckets);
    LOG(INFO) << "... reading n_buckets " << n_buckets;
    for(int i = 0; i < n_buckets; i++) {
      std::vector<int> bucket;
      buckets.push_back(bucket);
    }
    for(int i = 0; i < n_buckets; i++) {
      int leaf, bucket;
      size_t sze;
      start = read(start, buffer, leaf);
      start = read(start, buffer, bucket);
      start = read(start, buffer, sze);
      LOG(INFO) << "... reading bucket "
		<< leaf << " : " << bucket << " "
		<< sze << "<<";
      leaf_map[leaf] = bucket;
      for(int j = 0; j < sze; j++) {
	int id;
	start = read(start, buffer, id);
	buckets[bucket].push_back(id);
      }
    }
    LOG(INFO) << "DONE";
  }

}
