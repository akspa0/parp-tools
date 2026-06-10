#ifndef _WOWFILES_CATACLYSM_MCNKCATAOBJECTS_H_
#define _WOWFILES_CATACLYSM_MCNKCATAOBJECTS_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>
#include <wowfiles/Mcnk.h>

class McnkCataObjects : public Mcnk
{
  public:

    McnkCataObjects(const std::vector<char> & adtFile, int offsetInFile);
    McnkCataObjects(std::string letters, int givenSize, const std::vector<char> &data);

    std::vector<char> getWholeChunk() const;
	
	  friend std::ostream & operator<<(std::ostream & os, const McnkCataObjects & mcnkCataObjects);
	
  private:

    Chunk mcrd;
    Chunk mcrw;
	  std::vector<Chunk> objectsMcnkUnknown;
};

#endif