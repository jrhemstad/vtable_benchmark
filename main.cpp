

enum types
{
  CHAR,
  INT,
  FLOAT,
  DOUBLE,
}

struct column
{
  void * data;
  types t;
};

template <typname T>
column make_column(std::vector<T> v)
{
  column the_column;

  the_column.data = v.data();

  if(std::is_same<T, char>::value) v.t = CHAR;
  else if(std::is_same<T, int>::value) v.t = INT;
  else if(std::is_same<T, float>::value) v.t = FLOAT;
  else if(std::is_same<T, double>::value) v.t = DOUBLE;
  else
  {
    std::cout << "Invalid vector type.\n";
  }

  return the_column;
}

void add_column_elements(column const& l_column, const int l_index,
                         column const& r_column, const int r_index)
{
  switch(l_column.t)
  {
    case CHAR:
      using col_type = char;
      col_type l_col = static_cast<col_type>(l_column.data);
      col_type r_col = static_cast<col_type>(r_column.data);
      l_col[l_index] += r_col[r_index];
      break;
    case INT:
      using col_type = int;
      col_type l_col = static_cast<col_type>(l_column.data);
      col_type r_col = static_cast<col_type>(r_column.data);
      l_col[l_index] += r_col[r_index];
      break;
    case FLOAT:
      using col_type = float;
      col_type l_col = static_cast<col_type>(l_column.data);
      col_type r_col = static_cast<col_type>(r_column.data);
      l_col[l_index] += r_col[r_index];
      break;
    case DOUBLE:
      using col_type = double;
      col_type l_col = static_cast<col_type>(l_column.data);
      col_type r_col = static_cast<col_type>(r_column.data);
      l_col[l_index] += r_col[r_index];
      break;
    default:
  }
}

struct BaseColumn
{
  virtual void add_element(BaseColumn const& other_column, const int my_index, const int other_index ) = 0;
};

template <typename T>
struct TypedColumn : BaseColumn
{

  TypeColumn(std::vector<T> v) : data{v.data()}, size(v.size())
  { }

  TypedColumn(T * _data, int _size) : data{_data}, size{_size}
  { }

  virtual void add_element(BaseColumn const& other_column, const int my_index, const int other_index ) override
  {
    data[my_index] += outher_column.data[other_index];
  }

private:
  T * data;
  int size;
};

int main()
{
  std::vector<int> left(100);
  std::vecotr<int> right(100);
}
