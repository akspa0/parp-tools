#ifndef _WOWFILES_CATACLYSM_ADTCATA_H_
#define _WOWFILES_CATACLYSM_ADTCATA_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/WowChunkedFormat.h>
#include <wowfiles/Chunk.h>
#include <wowfiles/Mhdr.h>
#include <wowfiles/Mh2o.h>
#include <wowfiles/Mcnk.h>
#include <wowfiles/cataclysm/AdtCataTerrain.h>
#include <wowfiles/cataclysm/AdtCataTextures.h>
#include <wowfiles/cataclysm/AdtCataObjects.h>
#include <wowfiles/cataclysm/McnkCataTerrain.h>
#include <wowfiles/lichking/AdtLk.h>

class AdtCata : public WowChunkedFormat
{
  public:

    AdtCata(const std::string & adtName, const std::vector<char> & adtFile);

    void changeHeight(const int & heightToAdd);

    void toFile();
    void toFile(const std::string & fileName);
    AdtLk toAdtLk();

    friend std::ostream & operator<<(std::ostream & os, const AdtCata & adtCata);

  private:

    std::string adtName;
	  std::string texName;
	  std::string objName;

    AdtCataTerrain terrain;
	  AdtCataTextures textures;
	  AdtCataObjects objects;
};

#endif