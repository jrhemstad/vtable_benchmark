#include <chrono>

#include "../include/switch_test.cuh"
#include "../include/column.cuh"

const int NUM_BURN_IN = 2;

__host__ __device__
void add_column_elements(column const& l_column, const int l_index,
                         column const& r_column, const int r_index)
{
  switch(l_column.t)
  {
    case CHAR:
      {
        using col_type = char;
        col_type * l_col = static_cast<col_type*>(l_column.data);
        col_type * r_col = static_cast<col_type*>(r_column.data);
        l_col[l_index] += r_col[r_index];
        break;
      }
    case INT:
      {
        using col_type = int;
        col_type * l_col = static_cast<col_type*>(l_column.data);
        col_type * r_col = static_cast<col_type*>(r_column.data);
        l_col[l_index] += r_col[r_index];
        break;
      }
    case FLOAT:
      {
        using col_type = float;
        col_type * l_col = static_cast<col_type*>(l_column.data);
        col_type * r_col = static_cast<col_type*>(r_column.data);
        l_col[l_index] += r_col[r_index];
        break;
      }
    case DOUBLE:
      {
        using col_type = double;
        col_type * l_col = static_cast<col_type*>(l_column.data);
        col_type * r_col = static_cast<col_type*>(r_column.data);
        l_col[l_index] += r_col[r_index];
        break;
      }
    default:
      return;
  }
}

template <typename input_type>
std::vector<input_type> run_cpu_switch_test(std::vector<input_type> left,
                                            std::vector<input_type> right,
                                            const int num_iterations) 
{
  std::vector<input_type> result{left};

  column left_col(result);
  column right_col(right);

  auto start = std::chrono::high_resolution_clock::now();
  for(int iter = 0; iter < num_iterations + NUM_BURN_IN; ++iter)
  {
    if(NUM_BURN_IN == iter)
      start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < left.size(); ++i)
    {
      add_column_elements(left_col, i, right_col, i);
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = stop - start;

  
  std::cout << "CPU Switch test elapsed time(s): " << elapsed.count() << "\n";

  return result;
}
template std::vector<int> run_cpu_switch_test<int>(std::vector<int> left,
                                                   std::vector<int> right,
                                                   const int num_iterations);
template std::vector<double> run_cpu_switch_test<double>(std::vector<double> left,
                                                         std::vector<double> right,
                                                         const int num_iterations);

__global__
void test_kernel(column left, column right, size_t size)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  while(idx < size)
  {
    add_column_elements(left, idx, right, idx);
    idx += gridDim.x * blockDim.x;
  }
}

template <typename input_type>
std::vector<input_type> run_gpu_switch_test(std::vector<input_type> left,
                                            std::vector<input_type> right,
                                            const int num_iterations)
{
  std::vector<input_type> result{left};

  bool device_column = true;
  column left_col(result, device_column);
  column right_col(right, device_column);

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

    test_kernel<<<grid_size, block_size>>>(left_col, right_col, left.size());
  }

  if(cudaSuccess != cudaDeviceSynchronize()){
    std::cout << "GPU Switch Test failed!\n";
  }

  auto stop = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = stop - start;


  std::cout << "GPU Switch test elapsed time(s): " << elapsed.count() << "\n";

  cudaMemcpy(result.data(), left_col.data, left.size() * sizeof(input_type), cudaMemcpyDeviceToHost);

  cudaFree(left_col.data);
  cudaFree(right_col.data);

  return result;
}

template std::vector<int> run_gpu_switch_test<int>(std::vector<int> left,
                                                   std::vector<int> right,
                                                   const int num_iterations);
template std::vector<double> run_gpu_switch_test<double>(std::vector<double> left,
                                                         std::vector<double> right,
                                                         const int num_iterations);
