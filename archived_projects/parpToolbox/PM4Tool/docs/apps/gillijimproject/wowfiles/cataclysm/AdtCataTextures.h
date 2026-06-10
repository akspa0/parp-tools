#ifndef _WOWFILES_CATACLYSM_ADTCATATEXTURES_H_
#define _WOWFILES_CATACLYSM_ADTCATATEXTURES_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/WowChunkedFormat.h>
#include <wowfiles/Chunk.h>
#include <wowfiles/Mcnk.h>
#include <wowfiles/cataclysm/McnkCataTextures.h>

class AdtCataTextures : public WowChunkedFormat
{
  public:

    friend std::ostream & operator<<(std::ostream & os, const AdtCataTextures & adtCataTextures);
    friend class AdtCata;

  private:

    AdtCataTextures();
    AdtCataTextures(const std::string & adtName, const std::vector<char> & adtFile);

    void toFile();
    void toFile(const std::string & fileName);

    std::string adtName;
    Chunk texturesMver;
    Chunk mamp;
    Chunk mtex;
    std::vector<McnkCataTextures> texturesMcnks;
    Chunk mtxf;
	  Chunk mtxp;
	  std::vector<Chunk> texturesUnknown;
};

#endif