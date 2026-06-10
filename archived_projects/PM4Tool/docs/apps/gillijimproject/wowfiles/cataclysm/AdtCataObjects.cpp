#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>
#include <wowfiles/cataclysm/AdtCataObjects.h>
#include <wowfiles/Chunk.h>
#include <wowfiles/Mddf.h>
#include <wowfiles/Modf.h>
#include <wowfiles/Mmid.h>
#include <wowfiles/Mwid.h>
#include <wowfiles/cataclysm/McnkCataObjects.h>
#include <utilities/Utilities.h>

AdtCataObjects::AdtCataObjects()
{
  // TODO : something
}

AdtCataObjects::AdtCataObjects(const std::string & adtName, const std::vector<char> & adtFile) : adtName(adtName)
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
        objectsMver = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      case 'MMDX' :
        mmdx = Mmdx(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;
		
      case 'MMID' :
        mmid = Mmid(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      case 'MWMO' :
        mwmo = Mwmo(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      case 'MWID' :
        mwid = Mwid(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      case 'MDDF' :
        mddf = Mddf(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;	 		
		
      case 'MODF' :
        modf = Modf(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;	  	

      case 'MCNK' :
        objectsMcnks.push_back(McnkCataObjects(adtFile, offsetInFile));
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      default :
        objectsUnknown.push_back(Chunk(adtFile, offsetInFile));
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
    }
  }
}

void AdtCataObjects::addToObjectsHeight(const int & heightToAdd)
{
  mddf.addToObjectsHeight(heightToAdd);
  modf.addToObjectsHeight(heightToAdd);
}

void AdtCataObjects::toFile()
{
  std::string fileName (adtName);
  fileName.append("_new");

  toFile(fileName);
}

std::vector<Utilities::Point> AdtCataObjects::getAllObjectsCoords() const
{
  std::vector<Utilities::Point> coords ( mddf.getAllM2Coords() );
  std::vector<Utilities::Point> wmoCoords ( modf.getAllWmoCoords() ) ;

  coords.insert( coords.end(), wmoCoords.begin(), wmoCoords.end() );

  return coords;
}

void AdtCataObjects::toFile(const std::string & fileName)
{
  std::vector<char> wholeAdt(0);
  
  std::vector<char> tempData(objectsMver.getWholeChunk());
  wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());
  
  tempData = mmdx.getWholeChunk();
  wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());

  tempData = mmid.getWholeChunk();
  wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());

  tempData = mwmo.getWholeChunk();
  wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());

  tempData = mwid.getWholeChunk();
  wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());

  tempData = mddf.getWholeChunk();
  wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());

  tempData = modf.getWholeChunk();
  wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());

  int currentMcnk;

  for (currentMcnk = 0 ; currentMcnk < 256 ; ++currentMcnk)
  {
    tempData =  objectsMcnks[currentMcnk].getWholeChunk();
    wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());
  }

  std::ofstream outputFile (fileName.c_str(), std::ios::out|std::ios::binary);

  if (outputFile.is_open())
    outputFile.write((char *)&wholeAdt[0], sizeof(char) * wholeAdt.size());

  outputFile.close();
}

std::vector<std::string> AdtCataObjects::getAllM2Names() const
{
  return mmdx.getM2Names();
}

std::vector<std::string> AdtCataObjects::getAllWmoNames() const
{
  return mwmo.getWmoNames();
}

std::vector<Utilities::Point> AdtCataObjects::getAllM2Coords() const
{
  return mddf.getAllM2Coords();
}

std::vector<Utilities::Point> AdtCataObjects::getAllWmoCoords() const
{
  return modf.getAllWmoCoords();
}

std::vector<int> AdtCataObjects::getAllM2Indices() const
{
  return mddf.getEntriesIndices();
}

std::vector<int> AdtCataObjects::getAllWmoIndices() const
{
  return modf.getEntriesIndices();
}

std::ostream & operator<<(std::ostream & os, const AdtCataObjects & adtCataObjects)
{
  os << "------------------------------" << std::endl;
  os << adtCataObjects.adtName << std::endl;
  os << "------------------------------" << std::endl;
  /*os << adtCataObjects.objectsMver;
  os << adtCataObjects.mmdx;
  os << adtCataObjects.mmid;
  os << adtCataObjects.mwmo;
  os << adtCataObjects.mwid;
  os << adtCataObjects.mddf;*/
  os << adtCataObjects.modf;

  /*std::vector<McnkCataObjects>::const_iterator mcnksObjectsIter;
  int i (0);
  
  for (mcnksObjectsIter = adtCataObjects.objectsMcnks.begin() ; mcnksObjectsIter != adtCataObjects.objectsMcnks.end() ; ++mcnksObjectsIter)
  {
    os << "MCNK (objects) #" << i << " : " << std::endl;
    os << *mcnksObjectsIter;
    ++i;
  }

  std::vector<Chunk>::const_iterator objectsUnknownIter;

  for (objectsUnknownIter = adtCataObjects.objectsUnknown.begin() ; objectsUnknownIter != adtCataObjects.objectsUnknown.end() ; ++objectsUnknownIter)
  {
    os << *objectsUnknownIter;
  }*/    
  
  return os;
}
