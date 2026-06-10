#ifndef _WOWFILES_MAIN_H_
#define _WOWFILES_MAIN_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>

class Main : public Chunk
{
  public:
  
  Main();
  Main(std::ifstream & adtFile, int offsetInFile);
  Main(const  std::string letters, int givenSize, const std::vector<char> & chunkData);
};

#endif