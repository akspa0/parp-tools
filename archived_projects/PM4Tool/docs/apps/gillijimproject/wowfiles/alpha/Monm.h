#ifndef _WOWFILES_ALPHA_MONM_H_
#define _WOWFILES_ALPHA_MONM_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>

class Monm : public Chunk
{
  public:
  
    Monm();
    Monm(std::ifstream & wdtAlphaFile, int offsetInFile);
    Monm(const  std::string letters, int givenSize, const std::vector<char> & chunkData);
    
    std::vector<std::string> getFilesNames() const;

    Chunk toMwmo() const;

    friend std::ostream & operator<<(std::ostream & os, const Monm & monm);
};

#endif