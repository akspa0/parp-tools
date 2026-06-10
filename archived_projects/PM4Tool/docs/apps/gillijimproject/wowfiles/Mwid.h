#ifndef _WOWFILES_MWID_H_
#define _WOWFILES_MWID_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <wowfiles/Chunk.h>

class Mwid : public Chunk
{
  public:

    Mwid();
    Mwid(std::ifstream & adtFile, int offsetInFile);
	  Mwid(const std::vector<char> & wholeFile, int offsetInFile);
    Mwid(std::string letters, int givenSize, const std::vector<char> & chunkData);
    Mwid( const std::vector<int> & indicesFromMwmo );

    friend std::ostream & operator<<(std::ostream & os, const Mwid & mwid);	
};

#endif