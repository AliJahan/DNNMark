#include <iostream>
#include "common.h"
#include "dnnmark.h"
#include "usage.h"

using namespace dnnmark;

int main(int argc, char **argv) {
  //INIT_FLAGS(argc, argv);
  //INIT_LOG(argv);
  //LOG(INFO) << "DNNMark suites: Start...";
  std::cout << "DNNMark suites: Start...\n";
  DNNMark<TestType> dnnmark;
  dnnmark.ParseGeneralConfig(argv[1]);
  dnnmark.ParseLayerConfig(argv[1]);
  dnnmark.Initialize();
  dnnmark.Forward();
  dnnmark.GetTimer()->SumRecords();
  dnnmark.TearDown();
  //LOG(INFO) << "Total running time(ms): " << dnnmark.GetTimer()->GetTotalTime();
  //LOG(INFO) << "DNNMark suites: Tear down...";
  std::cout << "Total running time(ms): " << dnnmark.GetTimer()->GetTotalTime() << std::endl;
  std::cout << "DNNMark suites: Tear down...\n";
  return 0;
}
