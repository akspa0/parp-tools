#ifndef _WOWFILES_CATACLYSM_MCNKCATATERRAIN_H_
#define _WOWFILES_CATACLYSM_MCNKCATATERRAIN_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iostream>
#include <wowfiles/Chunk.h>
#include <wowfiles/Mcnk.h>
#include <wowfiles/lichking/McnkLk.h>
#include <wowfiles/cataclysm/McnrCata.h>

class McnkCataTerrain : public Mcnk
{
  public:

    McnkCataTerrain(const std::vector<char> & adtFile, int offsetInFile, const int & headerSize);
    McnkCataTerrain(std::string letters, int givenSize, const std::vector<char> &data);

    void addToRelativeHeight(const int & heightToAdd);

    std::vector<char> getWholeChunk() const;
    McnkLk toMcnkLk();

    friend std::ostream & operator<<(std::ostream & os, const McnkCataTerrain & mcnkCataTerrain);

  private:

    McnkHeader mcnkHeader;
    Chunk mcvt;
    Chunk mcbb;
    Chunk mcbi;
    Chunk mcbv;
    Chunk mccv;
    Chunk mclv;
    McnrCata mcnr;
    Chunk mclq;
    Chunk mcse;
	  std::vector<Chunk> terrainMcnkUnknown;
};

#endif