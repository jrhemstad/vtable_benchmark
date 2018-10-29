#include <vector>
#include <iostream>
#include <typeinfo>
#include <chrono>
#include <algorithm>
#include <random>
#include <numeric>

#include "include/switch_test.cuh"
#include "include/virtual_test.cuh"

const int INPUT_SIZE{100000000};
const int ITERATIONS{10};

template <typename T>
std::vector<T> generate_random_vector(const int size){
  std::vector<T> vec(size);
  std::iota(vec.begin(), vec.end(), T(0));
  std::shuffle(vec.begin(), vec.end(), std::mt19937{std::random_device{}()});
  return vec;
}

template <typename T>
bool is_equal(std::vector<T> const& lhs, std::vector<T> const& rhs)
{

  bool equal = std::equal(lhs.begin(), lhs.end(), rhs.begin());

  if(false == equal)
  {
    for(int i = 0; i < lhs.size() && i < rhs.size(); ++i)
    {
      if(lhs[i] != rhs[i]){
        std::cout << i << ": lhs: " << lhs[i] 
                       << " rhs: " << rhs[i] << "\n";
      }
    }
  }
  return equal;
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

  auto gpu_switch_result = run_gpu_switch_test(left, right, ITERATIONS);

  auto gpu_virtual_result = run_gpu_virtual_test(left, right, ITERATIONS);

  if( false == is_equal(cpu_switch_result, cpu_virtual_result) )
    std::cout << "ERROR: result mismatch between CPU Switch and CPU Virtual test!\n";

  if( false == is_equal(cpu_switch_result, gpu_virtual_result) )
    std::cout << "ERROR: result mismatch between CPU Switch and GPU Virtual test!\n";

  if( false == is_equal(gpu_virtual_result, gpu_switch_result) )
    std::cout << "ERROR: result mismatch between GPU Switch and GPU Virtual test!\n";

}
