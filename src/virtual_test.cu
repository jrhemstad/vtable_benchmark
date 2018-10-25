#include <thrust/device_vector.h>
#include <chrono>

#include "../include/virtual_test.cuh"
#include "../include/column.cuh"

struct BaseColumn
{
  __host__ __device__
  virtual void add_element(BaseColumn const& other_column, const int my_index, const int other_index ) = 0;

  void * base_data;
protected:
  BaseColumn(column the_column) : base_data{the_column.data}, size{the_column.size}
  {}
  int size;
};

template <typename T>
struct TypedColumn : BaseColumn
{
  TypedColumn(column the_column) : BaseColumn{the_column}, data{static_cast<T*>(base_data)}
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
double run_cpu_virtual_test(const int input_size, const int num_iterations)
{
  // Generate random input vector
  std::vector<input_type> left(input_size);
  std::iota(left.begin(), left.end(), input_type(0));
  std::shuffle(left.begin(), left.end(), std::mt19937{std::random_device{}()});

  std::vector<input_type> right(input_size);
  std::iota(right.begin(), right.end(), input_type(0));
  std::shuffle(right.begin(), right.end(), std::mt19937{std::random_device{}()});

  column left_col(left);
  column right_col(right);

  BaseColumn * left_base{new TypedColumn<input_type>(left_col)};
  BaseColumn * right_base{new TypedColumn<input_type>(right_col)};

  auto start = std::chrono::high_resolution_clock::now();
  for(int iter = 0; iter < num_iterations; ++iter)
  {
    for(int i = 0; i < input_size; ++i)
    {
      left_base->add_element(*right_base, i, i);
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = stop - start;

  return elapsed.count();
}
template double run_cpu_virtual_test<int>(int input_size, int num_iterations);
template double run_cpu_virtual_test<double>(int input_size, int num_iterations);

__global__
void test_kernel(BaseColumn * left, BaseColumn * right, size_t size)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  while(idx < size)
  {
    left->add_element(*right, idx, idx);
    idx += gridDim.x * blockDim.x;
  }
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

  column left_col(left);
  column right_col(right);

  cudaMalloc(&left_col.data, input_size * sizeof(input_type));
  cudaMemcpy(left_col.data, left.data(), left.size() * sizeof(input_type), cudaMemcpyHostToDevice);

  cudaMalloc(&right_col.data, input_size * sizeof(input_type));
  cudaMemcpy(right_col.data, right.data(), right.size() * sizeof(input_type), cudaMemcpyHostToDevice);

  BaseColumn * left_base{new TypedColumn<input_type>(left_col)};
  BaseColumn * right_base{new TypedColumn<input_type>(right_col)};

  constexpr int block_size = 256;
  const int grid_size = (input_size + block_size - 1)/block_size;

  auto start = std::chrono::high_resolution_clock::now();

  for(int i = 0; i < num_iterations; ++i)
    test_kernel<<<grid_size, block_size>>>(left_base, right_base, left.size());

  cudaDeviceSynchronize();

  auto stop = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = stop - start;

  cudaFree(left_col.data);
  cudaFree(right_col.data);

  return elapsed.count();
}
template double run_gpu_virtual_test<int>(int input_size, int num_iterations);
template double run_gpu_virtual_test<double>(int input_size, int num_iterations);



