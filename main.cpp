#include <vector>
#include <iostream>
#include <typeinfo>
#include <chrono>

#include "include/column.h"
#include "include/switch_test.h"
#include "include/virtual_test.h"


constexpr size_t INPUT_SIZE{1000000000};

int main()
{
  std::vector<int> left(INPUT_SIZE,1);
  std::vector<int> right(INPUT_SIZE,2);

  column left_column{left};
  column right_column{right};

  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < INPUT_SIZE; ++i)
    add_column_elements(left_column, i,
                        right_column, i);
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = stop - start;

  std::cout << "Switch duration(s): " << duration.count() << "\n";

  BaseColumn * left_base{new TypedColumn<int>(left_column)};
  BaseColumn * right_base{new TypedColumn<int>(right_column)};

  start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < INPUT_SIZE; ++i)
    left_base->add_element(*right_base, i, i);
  stop = std::chrono::high_resolution_clock::now();
  duration = stop - start;

  std::cout << "Vtable duration(s): " << duration.count() << "\n";
}
