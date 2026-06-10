#ifndef _UTILITIES_UTILITIES_H_
#define _UTILITIES_UTILITIES_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

namespace Utilities
{
  template<typename T, typename U> T get (const std::vector<U>& data, std::size_t offset)
  {
    return T (*reinterpret_cast<const T*> (&data[offset]));
  }

  template<typename T, typename U> std::vector<T> vecTo (const std::vector<U>& someData) // TODO : cannot work, not reliable at all. To change !
  {
    std::vector<T> content (0);
  
    typename std::vector<U>::const_iterator dataIter;
    int currentStart (0);
  
    for (dataIter = someData.begin() ; dataIter != someData.end() ; ++dataIter)
    {
      if ( ( dataIter - someData.begin() ) % sizeof(T) == 0 )
	    {
	      content.push_back( get<T>( someData, currentStart ) );
	      currentStart += sizeof(T);
	    }
    }

    return content;
  }

  void getWholeFile(const std::string & adtFileName, std::vector<char> & buffer);

  std::string getStringFromCharVector(const std::vector<char> & someData, const int start, const int stringLength);

  std::string getLettersFromFile(std::ifstream & adtFile, const int position);
  int getIntFromFile(std::ifstream & adtFile, const int position);
  std::vector<char> getCharVectorFromFile(std::ifstream & adtFile, const int position, const int length);

  std::vector<char> getCharVectorFromInt(const int someValue);
  std::vector<char> getCharVectorFromFloat(const float someValue);

  bool flagsExist(const int & bitmask, const int & whichFlags);

  int getAdtVersion(const std::string & adtName);
  int getWdtVersion(const std::string & wdtName);

  std::vector<std::string> getFileNames(const std::vector<char> & someData);

  struct Point
  {
    float x;
    float y;
    float z;
  };

};

#endif
