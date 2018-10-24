#ifndef SWITCH_H
#define SWITCH_H
#include "column.h"
#include <numeric>
#include <algorithm>
#include <random>

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

struct switch_test
{
  template <typename input_type>
  double run_cpu_test(const int input_size, const int num_iterations) 
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

    auto start = std::chrono::high_resolution_clock::now();
    for(int iter = 0; iter < num_iterations; ++iter)
    {
      for(int i = 0; i < input_size; ++i)
      {
        add_column_elements(left_col, i, right_col, i);
      }
    }
    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = stop - start;

    return elapsed.count();
  }
};

#endif
