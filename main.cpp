#include <vector>
#include <iostream>
#include <typeinfo>
#include <chrono>

#include "include/switch_test.cuh"
#include "include/virtual_test.cuh"

const int INPUT_SIZE{100000000};
const int ITERATIONS{1};

int main()
{
  std::cout << "Running VTable vs Switch benchmark. Input Size: "
            << INPUT_SIZE << " Iterations: " << ITERATIONS << "\n";

  double switch_duration = run_cpu_switch_test<int>(INPUT_SIZE, ITERATIONS);
  std::cout << "CPU Switch duration(s): " << switch_duration << "\n";

  double virtual_duration = run_cpu_virtual_test<int>(INPUT_SIZE, ITERATIONS);
  std::cout << "CPU Virtual duration(s): " << virtual_duration << "\n";

  double virtual_gpu_duration = run_gpu_virtual_test<int>(INPUT_SIZE, ITERATIONS);
  std::cout << "GPU Virtual duration(s): " << virtual_gpu_duration << "\n";

  double switch_gpu_duration = run_gpu_switch_test<int>(INPUT_SIZE, ITERATIONS);
  std::cout << "GPU switch duration(s): " << switch_gpu_duration << "\n";
}
