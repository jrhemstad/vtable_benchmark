#ifndef COLUMN_H
#define COLUMN_H
#include <type_traits>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <thrust/device_vector.h>

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

    if(true == device_column)
    {
      cudaMalloc(&data, size * sizeof(T));
      cudaMemcpy(data, v.data(), size*sizeof(T), cudaMemcpyHostToDevice);
    }
    else
      data = v.data();

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
    if(on_device)
      cudaFree(data);
  }

  void * data;
  types t;
  size_t size;
  bool on_device;
};

#endif
