#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include <glog/logging.h>

#include "IndexingTypes.h"
#include "TimeSeriesIndex.h"
#include "Serialization.h"

#include "gen/indexing.grpc.pb.h"


using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;


void copy(const TimeSeries *from, tsidx::TimeSeries &to) {
    int dim = from -> dim();
    int len = from -> length();
    to.id = -1;
    to.x = new float[dim * len];
    for(int i = 0; i < dim * len; i++) {
      to.x[i] = from -> ts(i);
    }
    to.dim = dim;
    to.length = len;    
}


class TimeSeriesServer final : public TimeSeriesService::Service {

public:
  TimeSeriesServer(int n_buckets, int bucket_size, float band_percentage) {
    idx_ = std::make_shared<tsidx::TimeSeriesIndex>(n_buckets, bucket_size, band_percentage);
  }
  
  ~TimeSeriesServer() {}
  
  Status insert(ServerContext *context,
		const TimeSeries *ts,
		ReindexingResponse* response) override {
    LOG(INFO) << "INSERT SERVER sequence of length=" << ts->length()
	      << " and dim=" << ts->dim()
	      << " preview=" << ts->ts(0) << ","
      	      << " preview=" << ts->ts(1) << ","
      	      << " preview=" << ts->ts(2) << ","
      	      << " preview=" << ts->ts(3); 
    tsidx::TimeSeries timeseries;   
    copy(ts, timeseries);
    int status = idx_ -> insert(timeseries);    
    response -> set_status(status);
    response -> set_ts_id(timeseries.id);
    return Status::OK;
  }
  
  Status query(ServerContext *context,
	       const TimeSeries *ts,
	       TimeSeriesIdx *idx) override {
    tsidx::TimeSeries timeseries;   
    copy(ts, timeseries);
  
    std::vector<int> nearest;
    int status = idx_ -> search_idx(timeseries, nearest);
    idx -> set_status(status);
    for(const auto& nn : nearest) {
      idx -> add_ids(nn);
    }
    return Status::OK;
  }
  
  Status reindex(ServerContext *context,
		 const ReindexingRequest* request,
		 ReindexingResponse* response) override {
    // TODO make this asynch
    int status = idx_ -> reindex(request -> n_samples());
    response -> set_status(status);
    response -> set_ts_id(-1);
    return Status::OK;
  }

  Status save(ServerContext *context,
	      const SaveIndexRequest *request,
	      SaveIndexResponse *response) override {
    idx_ -> save(request -> name());
    response -> set_response(0);
    return Status::OK;
  }

  Status load(ServerContext *context,
	      const LoadIndexRequest *request,
	      LoadIndexResponse *response) override {
    int n_buckets = idx_ -> n_buckets;
    int bucket_size = idx_ -> bucket_size;
    float band_percentage = idx_ -> band_percentage;
    idx_ -> load(request -> name());
    LOG(INFO) << "DONE LOADING";
    response -> set_response(0);
    LOG(INFO) << "DONE LOADING 4 Real";    
    return Status::OK;
  }
  
  
  
private:
  std::shared_ptr<tsidx::TimeSeriesIndex> idx_;
};


void run_server(int n_buckets, int bucket_size, float band_percentage) {
  std::string server_address("0.0.0.0:50051");
  TimeSeriesServer service(n_buckets, bucket_size, band_percentage);

  ServerBuilder builder;
  builder.AddListeningPort(server_address,
			   grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}


int main(int argc, char **argv) {
  run_server(100, 10, 0.75);
  return 0;
}
