#include <thrust/device_vector.h>
#include <chrono>
#include <new> // overload new operator

#include "../include/virtual_test.cuh"
#include "../include/column.cuh"

const int NUM_BURN_IN = 2;

constexpr int VAL_SIZE{8};
__device__ int value[VAL_SIZE] = {0, 1, 2, 3, 4, 5, 6, 7};

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
    T r0;
    T r1;
    T r2;
    T r3;
    T r4;
    T r5;
    T r6;
    T r7;

      for(int j = 0; j < VAL_SIZE; ++j )
      {
        r0 = value[j] * j;
        r1 = value[j] * j;
        r2 = value[j] * j;
        r3 = value[j] * j;
        r4 = value[j] * j;
        r5 = value[j] * j;
        r6 = value[j] * j;
        r7 = value[j] * j;
      }

      for(int j = 0; j < VAL_SIZE; ++j )
      {
        data[my_index] += value[j] * r0;
        data[my_index] += value[j] * r1;
        data[my_index] += value[j] * r2;
        data[my_index] += value[j] * r3;
        data[my_index] += value[j] * r4;
        data[my_index] += value[j] * r5;
        data[my_index] += value[j] * r6;
        data[my_index] += value[j] * r7;
      }
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
  for(int iter = 0; iter < num_iterations + NUM_BURN_IN; ++iter)
  {
    if(NUM_BURN_IN == iter)
      start = std::chrono::high_resolution_clock::now();

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
std::vector<input_type> run_gpu_virtual_test(std::vector<input_type> left,
                                             std::vector<input_type> right,
                                             const int num_iterations)
{
  std::vector<input_type> result{left};

  bool device_column = true;
  column left_col(result, device_column);
  column right_col(right, device_column);

  BaseColumn ** left_base{nullptr};
  cudaMalloc(&left_base, sizeof(BaseColumn*));
  derived_allocator<input_type><<<1,1>>>(left_base, left_col);

  BaseColumn ** right_base{nullptr};
  cudaMalloc(&right_base, sizeof(BaseColumn*));
  derived_allocator<input_type><<<1,1>>>(right_base, right_col);

  constexpr int block_size = 256;
  const int grid_size = (left.size() + block_size - 1)/block_size;

  auto start = std::chrono::high_resolution_clock::now();

  for(int i = 0; i < num_iterations + NUM_BURN_IN; ++i)
  {
    if(NUM_BURN_IN == i)
    {
      cudaDeviceSynchronize();
      start = std::chrono::high_resolution_clock::now();
    }

    test_kernel<<<grid_size, block_size>>>(left_base, right_base, left.size());
  }

  if(cudaSuccess != cudaDeviceSynchronize()){
    std::cout << "GPU Virtual Test failed!\n";
  }

  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = stop - start;


  std::cout << "GPU Virtual test elpased time (s): " << elapsed.count() << "\n";

  cudaMemcpy(result.data(), left_col.data, result.size() * sizeof(input_type), cudaMemcpyDeviceToHost );

  cudaFree(left_col.data);
  cudaFree(right_col.data);

  return result;
}

template std::vector<int> run_gpu_virtual_test<int>(std::vector<int> left,
                                                   std::vector<int> right,
                                                   const int num_iterations);
template std::vector<double> run_gpu_virtual_test<double>(std::vector<double> left,
                                                         std::vector<double> right,
                                                         const int num_iterations);
