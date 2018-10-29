#include <vector>
#include <iostream>
#include <typeinfo>
#include <chrono>
#include <algorithm>
#include <random>
#include <numeric>

#include "include/switch_test.cuh"
#include "include/virtual_test.cuh"

const int INPUT_SIZE{100};
const int ITERATIONS{1};

template <typename T>
std::vector<T> generate_random_vector(const int size){
  std::vector<T> vec(size);
  std::iota(vec.begin(), vec.end(), T(0));
  std::shuffle(vec.begin(), vec.end(), std::mt19937{std::random_device{}()});
  return vec;
}

int main()
{
  std::cout << "Running VTable vs Switch benchmark. Input Size: "
            << INPUT_SIZE << " Iterations: " << ITERATIONS << "\n";

  using input_t = int;

  auto left = generate_random_vector<input_t>(INPUT_SIZE);
  auto right = generate_random_vector<input_t>(INPUT_SIZE);

  auto cpu_switch_result = run_cpu_switch_test(left, right, ITERATIONS);

  auto cpu_virtual_result = run_cpu_virtual_test(left, right, ITERATIONS);

  if(false == std::equal(cpu_switch_result.begin(), cpu_switch_result.end(), 
                         cpu_virtual_result.begin()))
  {
    std::cout << "ERROR: result mismatch between CPU Switch and Virtual test!\n";

    for(int i = 0; i < left.size(); ++i)
    {
      if(cpu_switch_result[i] != cpu_virtual_result[i]){
        std::cout << i << ": switch: " << cpu_switch_result[i] 
                       << " virtual: " << cpu_virtual_result[i] << "\n";
      }
    }
  }

  double virtual_gpu_duration = run_gpu_virtual_test<input_t>(INPUT_SIZE, ITERATIONS);
  std::cout << "GPU Virtual duration(s): " << virtual_gpu_duration << "\n";

  double switch_gpu_duration = run_gpu_switch_test<input_t>(INPUT_SIZE, ITERATIONS);
  std::cout << "GPU switch duration(s): " << switch_gpu_duration << "\n";
}
