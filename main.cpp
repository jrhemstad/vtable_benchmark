#include <vector>
#include <iostream>
#include <typeinfo>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <chrono>

enum types
{
  CHAR,
  INT,
  FLOAT,
  DOUBLE
};

struct column
{
  template<typename T>
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

struct BaseColumn
{
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
