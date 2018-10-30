#include <chrono>
#include <iostream>
#include <thrust/device_vector.h>
#include "../include/baseline.cuh"

const int NUM_BURN_IN = 2;

template <typename input_type>
std::vector<input_type> run_cpu_baseline(std::vector<input_type> left,
                                         std::vector<input_type> right,
                                         const int num_iterations)
{
  std::vector<input_type> result{left};

  auto start = std::chrono::high_resolution_clock::now();

  for(int iter = 0; iter < num_iterations + NUM_BURN_IN; ++iter)
  {

    if(NUM_BURN_IN == iter)
      start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < left.size(); ++i)
      result[i] += right[i];
  }

  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = stop - start;

  std::cout << "CPU baseline elapsed time(s): " << elapsed.count() << "\n";

  return result;
}

template std::vector<int> run_cpu_baseline<int>(std::vector<int> left,
                                                std::vector<int> right,
                                                const int num_iterations);

template std::vector<double> run_cpu_baseline<double>(std::vector<double>  left,
                                                      std::vector<double>  right,
                                                      const int num_iterations); 

template <typename T>
__global__
void baseline_kernel(T * left, T * right, int64_t size){

  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  while(idx < size)
  {
    left[idx] += right[idx];
    idx += blockDim.x * gridDim.x;
  }
}

template <typename input_type>
std::vector<input_type> run_gpu_baseline(std::vector<input_type>  left,
                                         std::vector<input_type>  right,
                                         const int num_iterations)
{

  thrust::device_vector<input_type> d_left{left};
  thrust::device_vector<input_type> d_right{right};

  constexpr int block_size = 256;
  const int grid_size = (left.size() + block_size - 1)/block_size;

  auto start = std::chrono::high_resolution_clock::now();

  for(int iter = 0; iter < num_iterations + NUM_BURN_IN; ++iter)
  {
    if(NUM_BURN_IN == iter)
    {
      cudaDeviceSynchronize();
      start = std::chrono::high_resolution_clock::now();
    }

    baseline_kernel<<<grid_size, block_size>>>(d_left.data().get(), 
                                               d_right.data().get(),
                                               d_left.size());
  }

  if(cudaSuccess != cudaDeviceSynchronize()){
    std::cout << "GPU baseline failed!\n";
  }

  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = stop - start;

  std::cout << "gpu baseline elapsed time(s): " << elapsed.count() << "\n";

  std::vector<input_type> result(left.size());
  cudaMemcpy(result.data(), d_left.data().get(), d_left.size() * sizeof(input_type), cudaMemcpyDeviceToHost);

  return result;

}

template std::vector<int> run_gpu_baseline<int>(std::vector<int> left,
                                                std::vector<int> right,
                                                const int num_iterations);
template std::vector<double> run_gpu_baseline<double>(std::vector<double> left,
                                                      std::vector<double> right,
                                                      const int num_iterations);

