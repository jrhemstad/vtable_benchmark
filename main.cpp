#include <vector>
#include <iostream>
#include <typeinfo>
#include <chrono>

#include "include/switch_test.h"
#include "include/virtual_test.cuh"

const int INPUT_SIZE{100};
const int ITERATIONS{1};

int main()
{
  std::cout << "Running VTable vs Switch benchmark. Input Size: "
            << INPUT_SIZE << " Iterations: " << ITERATIONS << "\n";

  switch_test the_switch_test{};
  const auto switch_duration = the_switch_test.run_cpu_test<int>(INPUT_SIZE, ITERATIONS);

  std::cout << "CPU Switch duration(s): " << switch_duration << "\n";

  double virtual_duration = run_cpu_virtual_test<int>(INPUT_SIZE, ITERATIONS);
  std::cout << "CPU Virtual duration(s): " << virtual_duration << "\n";


  double virtual_gpu_duration = run_gpu_virtual_test<int>(INPUT_SIZE, ITERATIONS);
  std::cout << "GPU Virtual duration(s): " << virtual_gpu_duration << "\n";
}
