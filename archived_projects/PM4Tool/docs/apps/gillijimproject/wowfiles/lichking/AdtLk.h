#ifndef _WOWFILES_LICHKING_ADTLK_H_
#define _WOWFILES_LICHKING_ADTLK_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/WowChunkedFormat.h>
#include <wowfiles/Chunk.h>
#include <wowfiles/ChunkHeaders.h>
#include <wowfiles/Mh2o.h>
#include <wowfiles/Mcin.h>
#include <wowfiles/Mddf.h>
#include <wowfiles/Modf.h>
#include <wowfiles/Mhdr.h>
#include <wowfiles/Mmdx.h>
#include <wowfiles/Mmid.h>
#include <wowfiles/Mwmo.h>
#include <wowfiles/Mwid.h>
#include <wowfiles/lichking/McnkLk.h>

class AdtLk : public WowChunkedFormat
{
  public:

    AdtLk(const std::vector<char> & adtFile, const std::string & adtFileName);
    AdtLk(const std::string & name
      , const Chunk & cMver
      , const int & mhdrFlags
      , const Mh2o & cMh2o
      , const Chunk & cMtex
      , const Mmdx & cMmdx
      , const Mmid & cMmid
      , const Mwmo & cMwmo
      , const Mwid & cMwid
      , const Mddf & cMddf
      , const Modf & cModf
      , const std::vector<McnkLk> & cMcnks
      , const Chunk & cMfbo
      , const Chunk & cMtxf
      );

    void toFile();
    void toFile(std::string fileName);
    int getMcnksWholeSize();
    bool checkIntegrity();

    void mh2oToFile();
    void importMh2o(std::string mh2oName);

    std::vector<std::string> getAllM2Names() const; // TODO : change all that.
    std::vector<std::string> getAllWmoNames() const;
    std::vector<Utilities::Point> getAllM2Coords() const; 
    std::vector<Utilities::Point> getAllWmoCoords() const;
    std::vector<int> getAllM2Indices() const;
    std::vector<int> getAllWmoIndices() const;

    friend std::ostream & operator<<(std::ostream & os, const AdtLk & adtLk);

  private:

    int getMhdrFlags();
    bool checkMcinOffsets();
    bool checkMhdrOffsets();
    void updateOrCreateMhdrAndMcin();

    std::string adtName;
    Chunk mver;
    Mhdr mhdr;
    Mcin mcin;
    Mh2o mh2o;
    Chunk mtex;
    Mmdx mmdx;
    Mmid mmid;
    Mwmo mwmo;
    Mwid mwid;
    Mddf mddf;
    Modf modf;
    std::vector<McnkLk> mcnks;
    Chunk mfbo;
    Chunk mtxf;
};

#endif