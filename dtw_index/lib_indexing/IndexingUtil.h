#pragma once
#include "IndexingTypes.h"
#include <string>

namespace tsidx {

  /**
   * Index into a flattened 2D Matrix.
   *
   * For a N x M matrix the array is of size N*M and the
   * index [i,j] is i * M + j in the flattened matrix
   */
  int idx(int i, int j, int cols);

  /**
   * Squared euclidean distance between two frames in two time series.
   * For a series x of length N and a series y of length M with equal dimensions
   * we calculate ||x_i - y_j||^2
   */
  float euclidean_at(const TimeSeries &x, const TimeSeries &y, int i, int j);

  /**
   * Minimum of three floats
   */
  float min3(float x, float y, float z);

  /**
   * Dynamic Time Warping distance between two time series
   * the sakoe chiba band is defined in terms of max(|x|, |y|) * band_percentage
   */
  float dtw(const TimeSeries &x, const TimeSeries &y, float band_percentage);

  /**
   * Select a random sequence id for the selected range 
   */
  int select_rand(const TimeSeriesBatch& batch,
		  const BatchRange& range, const BatchIndex& idx);

  /**
   * Select a random sequence far away from the comparison sequnce.
   * The farness is determined by dtw(xi, x_comparsion)^2.
   * returns another sequence
   */
  int select_far(const TimeSeriesBatch& batch,
		 const BatchRange& range,
		 const BatchIndex& idx,
		 int comparison_sequence,
		 float band_percentage);

  /**
   * Partition time series in a range (think quicksort)
   * Given the returned partition point mid or | the data will look like
   * [left, left, left | right, right]. the left sequences will be closer to
   * left sequence and right closer to the right sequence
   */     
  int partition(const TimeSeriesBatch& batch,
		const BatchRange& range, BatchIndex& idx, int left, int right, float band_percentage);

  /**
   * Build a binary tree. At each node we index two time series who decide to follow the left node
   * or right node. Each leaf index should refer to similar sequences.
   */
  void build_tree(const TimeSeriesBatch& batch, int bucket_size, float band_percentage, Node *root);

  /**
   * Return the node id of the leaf node we find when following the indexing tree
   */
  int search(const TimeSeries &ts, const TimeSeriesBatch& batch, Node *node, float band_percentage);

  /**
   * Deconstruct a tree
   */
  int delete_tree(Node *node); 

  /**
   * All leaf node ids in a tree
   */ 
  void leaf_nodes(Node *node, std::string offset, std::vector<int> &leafs);

  /**
   * Number of nodes
   */
  int n_nodes(const Node *node);
}
