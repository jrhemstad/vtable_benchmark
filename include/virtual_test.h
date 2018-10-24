#ifndef VIRTUAL_H
#define VIRTUAL_H
#include "column.h"

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
#endif
