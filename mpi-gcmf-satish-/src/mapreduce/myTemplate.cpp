
#include<vector>

using namespace std;


template <typename T>
int func(vector<T> param)
{
  
  return 0;
}

int main()
{
  vector<int*> v;
  func(v);
  return 0;
}