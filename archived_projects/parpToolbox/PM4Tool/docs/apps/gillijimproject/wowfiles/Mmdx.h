#ifndef _WOWFILES_MMDX_H_
#define _WOWFILES_MMDX_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <wowfiles/Chunk.h>

class Mmdx : public Chunk
{
  public:

    Mmdx();
    Mmdx(std::ifstream & adtFile, int offsetInFile);
	  Mmdx(const std::vector<char> & wholeFile, int offsetInFile);
    Mmdx(std::string letters, int givenSize, const std::vector<char> & chunkData);
    Mmdx(const std::vector<int> & indices, const std::vector<std::string> & allFileNames);
	
    std::vector<int> getIndicesForMmid() const;
    std::vector<std::string> getM2Names() const;

    friend std::ostream & operator<<(std::ostream & os, const Mmdx & mmdx);	
};

#endif