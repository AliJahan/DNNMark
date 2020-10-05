#include <iostream>
#include "common.h"
#include "dnnmark.h"
//#include "usage.h"

using namespace dnnmark;

int main(int argc, char **argv) {
  //INIT_FLAGS(argc, argv);
  //INIT_LOG(argv);
  std::cout << "@@!@@DNNMark suites: Start...\n";
  DNNMark<TestType> dnnmark(58);
  dnnmark.ParseAllConfig(argv[1]);

  dnnmark.Initialize();
  std::cout << "@@!@@DNNMark suites: Initialized...\n";
  // Warm up
  /*if (1) {
    for (int i = 0; i < 5; i++) {
      dnnmark.Forward();
      dnnmark.Backward();
    }
  }*/
  dnnmark.GetTimer()->Clear();
  std::cout << "@@!@@DNNMark suites: Forward.start...\n";
  // Real benchmark
  //for (int i = 0; i < 10; i++) {
    //std::cout << "Iteration " << i<<std::endl;
    dnnmark.Forward();
    //dnnmark.Backward();
  //}
  std::cout << "@@!@@DNNMark suites: Forward.End...\n";
  dnnmark.GetTimer()->SumRecords();

  dnnmark.TearDown();

  std::cout << "@@!@@Total running time(ms): " << dnnmark.GetTimer()->GetTotalTime()<<std::endl;
  std::cout << "@@!@@DNNMark suites: Tear down...\n";
  printf("@@!@@Total running time(ms): %f\n", dnnmark.GetTimer()->GetTotalTime());
  return 0;
}
