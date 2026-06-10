#ifndef _WOWFILES_MMID_H_
#define _WOWFILES_MMID_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <wowfiles/Chunk.h>

class Mmid : public Chunk
{
  public:

    Mmid();
    Mmid(std::ifstream & adtFile, int offsetInFile);
	  Mmid(const std::vector<char> & wholeFile, int offsetInFile);
    Mmid(std::string letters, int givenSize, const std::vector<char> & chunkData);
	
    Mmid( const std::vector<int> & indicesFromMmdx );

    friend std::ostream & operator<<(std::ostream & os, const Mmid & mmid);	
};

#endif