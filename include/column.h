#ifndef COLUMN_H
#define COLUMN_H
#include <type_traits>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

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
  column(std::vector<T> & v)
  {
    data = v.data();
    size = v.size();

    if(std::is_same<T, char>::value) t = CHAR;
    else if(std::is_same<T, int>::value) t = INT;
    else if(std::is_same<T, float>::value) t = FLOAT;
    else if(std::is_same<T, double>::value) t = DOUBLE;
    else
    {
      std::cout << "Invalid vector type.\n";
    }
  }

  void * data;
  types t;
  int size;
};

#endif
