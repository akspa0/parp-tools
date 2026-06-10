#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>
#include <wowfiles/cataclysm/AdtCataTerrain.h>
#include <wowfiles/Chunk.h>
#include <wowfiles/Mddf.h>
#include <wowfiles/Modf.h>
#include <wowfiles/Mhdr.h>
#include <wowfiles/Mh2o.h>
#include <wowfiles/Mcnk.h>
#include <wowfiles/cataclysm/McnkCataTerrain.h>
#include <wowfiles/cataclysm/McnrCata.h>
#include <utilities/Utilities.h>
#include <wowfiles/lichking/AdtLk.h>

AdtCataTerrain::AdtCataTerrain()
{
  // TODO : something
}

AdtCataTerrain::AdtCataTerrain(const std::string & adtFileName, const std::vector<char> & adtFile) : adtName(adtFileName)
{
  const int fileSize = adtFile.size();
  int offsetInFile (0);
  int currentChunkSize (0);

  int chunkName;

  while (offsetInFile < fileSize)
  {
    chunkName = Utilities::get<int>(adtFile, offsetInFile);

    switch (chunkName)
    {
      case 'MVER' :
        terrainMver = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      case 'MHDR' :
        mhdr = Mhdr(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      case 'MH2O' :
        mh2o = Mh2o(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      case 'MCNK' :
        terrainMcnks.push_back(McnkCataTerrain(adtFile, offsetInFile, mcnkTerrainHeaderSize));
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      case 'MBMH' :
        mbmh = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      case 'MBMI' :
        mbmi = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      case 'MBMV' :
        mbmv = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      case 'MFBO' :
        mfbo = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      default :
        terrainUnknown.push_back(Chunk(adtFile, offsetInFile));
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
    }
  }
}

void AdtCataTerrain::addToTerrainHeight(const int & heightToAdd)
{
  std::vector<McnkCataTerrain>::iterator mcnkIter;
  
  for (mcnkIter = terrainMcnks.begin() ; mcnkIter != terrainMcnks.end() ; ++mcnkIter)
  {
    mcnkIter->addToRelativeHeight(heightToAdd);
  }  
}

void AdtCataTerrain::toFile()
{
  std::string fileName (adtName);
  fileName.append("_new");

  toFile(fileName);
}

void AdtCataTerrain::toFile(const std::string & fileName)
{
  std::vector<char> wholeAdt(0);
  
  std::vector<char> tempData(terrainMver.getWholeChunk());
  wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());
  
  tempData = mhdr.getWholeChunk();
  wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());

  if (!mh2o.isEmpty())
  {
    tempData = mh2o.getWholeChunk();
    wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());
  }

  int currentMcnk;

  for (currentMcnk = 0 ; currentMcnk < 256 ; ++currentMcnk)
  {
    tempData =  terrainMcnks[currentMcnk].getWholeChunk();
    wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());
  }

  if (!mbmh.isEmpty())
  {
    tempData = mbmh.getWholeChunk();
    wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());
  }

  if (!mbmi.isEmpty())
  {
    tempData = mbmi.getWholeChunk();
    wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());
  }

  if (!mbmv.isEmpty())
  {
    tempData = mbmv.getWholeChunk();
    wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());
  }

  if (!mfbo.isEmpty())
  {
    tempData = mfbo.getWholeChunk();
    wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());
  }

  std::ofstream outputFile (fileName.c_str(), std::ios::out|std::ios::binary);

  if (outputFile.is_open())
    outputFile.write((char *)&wholeAdt[0], sizeof(char) * wholeAdt.size());

  outputFile.close();
}

std::ostream & operator<<(std::ostream & os, const AdtCataTerrain & adtCataTerrain)
{
  os << adtCataTerrain.adtName << std::endl;
  os << "------------------------------" << std::endl;
  os << adtCataTerrain.terrainMver;
  os << adtCataTerrain.mhdr;
  os << adtCataTerrain.mh2o;

  std::vector<McnkCataTerrain>::const_iterator mcnksIter;
  int i (0);

  for (mcnksIter = adtCataTerrain.terrainMcnks.begin() ; mcnksIter != adtCataTerrain.terrainMcnks.end() ; ++mcnksIter)
  {
    os << "MCNK #" << i << " : " << std::endl;
    os << *mcnksIter;
    ++i;
  }

  os << adtCataTerrain.mbmh;
  os << adtCataTerrain.mbmi;
  os << adtCataTerrain.mbmv;
  os << adtCataTerrain.mfbo;

  std::vector<Chunk>::const_iterator terrainUnknownIter;

  for (terrainUnknownIter = adtCataTerrain.terrainUnknown.begin() ; terrainUnknownIter != adtCataTerrain.terrainUnknown.end() ; ++terrainUnknownIter)
  {
    os << *terrainUnknownIter;
  }  
    
  return os;
}