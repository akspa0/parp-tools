#ifndef _WOWFILES_ALPHA_MDNM_H_
#define _WOWFILES_ALPHA_MDNM_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>

class Mdnm : public Chunk
{
  public:
  
    Mdnm();
    Mdnm(std::ifstream & wdtAlphaFile, int offsetInFile);
    Mdnm(const std::string letters, int givenSize, const std::vector<char> & chunkData);

    std::vector<std::string> getFilesNames() const;

    friend std::ostream & operator<<(std::ostream & os, const Mdnm & mdnm);
};

#endif