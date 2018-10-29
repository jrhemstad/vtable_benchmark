#include <thrust/device_vector.h>
#include <chrono>
#include <new> // overload new operator

#include "../include/virtual_test.cuh"
#include "../include/column.cuh"

struct BaseColumn
{
  __host__ __device__
  virtual void add_element(BaseColumn const& other_column, const int my_index, const int other_index ) = 0;

  void * base_data;
protected:
  __host__ __device__
  BaseColumn(column & the_column) : base_data{the_column.data}, size{the_column.size}
  {}
  size_t size;
};

template <typename T>
struct TypedColumn : BaseColumn
{
  __host__ __device__
  TypedColumn(column & the_column) : BaseColumn{the_column}, data{static_cast<T*>(base_data)}
  { }

  __host__ __device__
  virtual void add_element(BaseColumn const& other_column, const int my_index, const int other_index ) override
  {
    // DANGER: This assumes other_column is a TypedColumn<T> with the same T...
    // Is there some way to guarantee that they are the same?
    // Solution 1: Check that the enum types are equal
    // Solution 2: Use dynamic cast and check for nullptr/thrown exception
    data[my_index] += static_cast<T*>(other_column.base_data)[other_index];
  }

private:
  T * data;
};

template <typename input_type>
std::vector<input_type> run_cpu_virtual_test(std::vector<input_type> left,
                                             std::vector<input_type> right,
                                             const int num_iterations)
{
  std::vector<input_type> result{left};

  column left_col(result);
  column right_col(right);

  BaseColumn * left_base{new TypedColumn<input_type>(left_col)};
  BaseColumn * right_base{new TypedColumn<input_type>(right_col)};

  auto start = std::chrono::high_resolution_clock::now();
  for(int iter = 0; iter < num_iterations; ++iter)
  {
    for(int i = 0; i < left.size(); ++i)
    {
      left_base->add_element(*right_base, i, i);
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = stop - start;

  std::cout << "CPU Virtual test elpased time (s): " << elapsed.count() << "\n";

  return result;
}
template std::vector<int> run_cpu_virtual_test<int>(std::vector<int> left,
                                                   std::vector<int> right,
                                                   const int num_iterations);
template std::vector<double> run_cpu_virtual_test<double>(std::vector<double> left,
                                                         std::vector<double> right,
                                                         const int num_iterations);

__global__
void test_kernel(BaseColumn ** left, BaseColumn ** right, size_t size)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  while(idx < size)
  {
    (*left)->add_element(**right, idx, idx);
    idx += gridDim.x * blockDim.x;
  }
}

template <typename input_type>
__global__ 
void derived_allocator(BaseColumn ** p, column the_column)
{
  if(0 == threadIdx.x)
    *p = static_cast<BaseColumn*>(new TypedColumn<input_type>(the_column));
}

template <typename input_type>
double run_gpu_virtual_test(const int input_size, const int num_iterations)
{
  // Generate random input vector
  std::vector<input_type> left(input_size);
  std::iota(left.begin(), left.end(), input_type(0));
  std::shuffle(left.begin(), left.end(), std::mt19937{std::random_device{}()});

  std::vector<input_type> right(input_size);
  std::iota(right.begin(), right.end(), input_type(0));
  std::shuffle(right.begin(), right.end(), std::mt19937{std::random_device{}()});

  bool device_column = true;
  column left_col(left, device_column);
  column right_col(right, device_column);

  BaseColumn ** left_base{nullptr};
  cudaMalloc(&left_base, sizeof(BaseColumn*));
  derived_allocator<input_type><<<1,1>>>(left_base, left_col);

  BaseColumn ** right_base{nullptr};
  cudaMalloc(&right_base, sizeof(BaseColumn*));
  derived_allocator<input_type><<<1,1>>>(right_base, right_col);

  constexpr int block_size = 256;
  const int grid_size = (input_size + block_size - 1)/block_size;

  auto start = std::chrono::high_resolution_clock::now();

  for(int i = 0; i < num_iterations; ++i)
    test_kernel<<<grid_size, block_size>>>(left_base, right_base, left.size());

  if(cudaSuccess != cudaDeviceSynchronize()){
    std::cout << "GPU Virtual Test failed!\n";
  }

  auto stop = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = stop - start;

  return elapsed.count();
}
template double run_gpu_virtual_test<int>(int input_size, int num_iterations);
template double run_gpu_virtual_test<double>(int input_size, int num_iterations);
