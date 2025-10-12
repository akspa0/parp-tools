#include <vector>

#include <binary_io.h>


namespace bin_io
{

  template<typename T> T get (const std::vector<char> &data, std::size_t offset)
  {
      return T (*reinterpret_cast<const T*> (&data[offset]));
  }

  unsigned short ReadInt16(const std::vector<char> &data, int offset)
  {
    return get<unsigned short> (data, offset);
  }
  unsigned int ReadInt32(const std::vector<char> &data, int offset)
  {
    return get<unsigned int> (data, offset);
  }
  float ReadFloat(const std::vector<char> &data, int offset)
  {
    return get<float> (data, offset);
  }

}
