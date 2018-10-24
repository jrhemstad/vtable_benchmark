#ifndef SWITCH_H
#define SWITCH_H
#include "column.h"
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
#endif
