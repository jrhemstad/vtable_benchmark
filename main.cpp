#include <vector>
#include <iostream>
#include <typeinfo>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator

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

protected:
  BaseColumn(column the_column) : base_data{the_column.data}, size{the_column.size}
  {}
  void * base_data;
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
    // Check the overhead of this...
    try {
      TypedColumn<T> const& actual_column = dynamic_cast< TypedColumn<T> const&>(other_column);
      std::cout << "Adding " << data[my_index] << " and " << actual_column.data[other_index] << std::endl;
      data[my_index] += actual_column.data[other_index];
    } 
    catch(const std::bad_cast& e) {
      printf("Tried to add two columns of different types.\n");
    }
  }

private:
  T * data;
};

int main()
{
  std::vector<int> left(10,1);
  std::vector<int> right(10,2);

  column left_column{left};
  column right_column{right};

  BaseColumn * left_base{new TypedColumn<int>(left_column)};
  BaseColumn * right_base{new TypedColumn<int>(right_column)};

  left_base->add_element(*right_base, 0,0);

  std::cout << left[0] << std::endl;
}
