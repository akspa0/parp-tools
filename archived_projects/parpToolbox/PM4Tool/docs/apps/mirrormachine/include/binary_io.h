#ifndef BINARY_IO_H
#define BINARY_IO_H

#include <vector>



// this namespace provides some functions to use during dealing with vector<char>
// for input or output purpose
namespace bin_io
{

  // gets a var from a vector<char> at some offset
  unsigned short ReadInt16(const std::vector<char> &data, int offset);
  unsigned int ReadInt32(const std::vector<char> &data, int offset);
  float ReadFloat(const std::vector<char> &data, int offset);

  // gets a vector<char> from some values
  // this is pure bitwise and doesn't care about unsigned types
  inline std::vector<char> vc_Short(const short &value)
  {
    std::vector<char> vc;
    vc.push_back(value & 0xff);
    vc.push_back((value >> 8) & 0xff);
    return vc;
  }
  inline std::vector<char> vc_Int(const int &value)
  {
    std::vector<char> vc;
    vc.push_back(value & 0xff);
    vc.push_back((value >> 8) & 0xff);
    vc.push_back((value >> 16) & 0xff);
    vc.push_back((value >> 24) & 0xff);
    return vc;
  }
  inline std::vector<char> vc_Float(const float &value)
  {
    std::vector<char> vc;
    unsigned char* tmp = (unsigned char *)&value;
    for (unsigned int i(0) ; i < sizeof(value) ; ++i)
      vc.push_back(tmp[i]);
    return vc;
  }

}


#endif // BINARY_IO_H
