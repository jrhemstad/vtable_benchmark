#ifndef COLUMN_H
#define COLUMN_H
#include <type_traits>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <thrust/device_vector.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

enum types
{
  CHAR,
  INT,
  FLOAT,
  DOUBLE
};

struct column
{
  template <typename T>
  column(std::vector<T> & v, bool device_column = false) : size{v.size()}, on_device{device_column}
  {

    if(true == device_column) {
      gpuErrchk(cudaMalloc(&data, size * sizeof(T)));
      gpuErrchk(cudaMemcpy(data, v.data(), size*sizeof(T), cudaMemcpyHostToDevice));
    }
    else {
      data = v.data();
    }

    if(std::is_same<T, char>::value) t = CHAR;
    else if(std::is_same<T, int>::value) t = INT;
    else if(std::is_same<T, float>::value) t = FLOAT;
    else if(std::is_same<T, double>::value) t = DOUBLE;
    else
    {
      std::cout << "Invalid vector type.\n";
    }
  }

  ~column()
  {
    //if(true == on_device)
    //{
    //  gpuErrchk(cudaFree(data));
    //}
  }

  void * data;
  types t;
  size_t size;
  bool on_device;
};

#endif
