#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>
#include <wowfiles/cataclysm/AdtCataTextures.h>
#include <wowfiles/Chunk.h>
#include <wowfiles/cataclysm/McnkCataTextures.h>
#include <utilities/Utilities.h>

AdtCataTextures::AdtCataTextures()
{
  // TODO : something
}

AdtCataTextures::AdtCataTextures(const std::string & adtName, const std::vector<char> & adtFile) : adtName(adtName)
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
        texturesMver = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      case 'MAMP' :
        mamp = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      case 'MTEX' :
        mtex = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;		
		
      case 'MCNK' :
        texturesMcnks.push_back(McnkCataTextures(adtFile, offsetInFile));
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      case 'MTXF' :
        mtxf = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;	

      case 'MTXP' :
        mtxp = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;
		
      default :
        texturesUnknown.push_back(Chunk(adtFile, offsetInFile));
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
    }
  }	
}

void AdtCataTextures::toFile()
{
  std::string fileName (adtName);
  fileName.append("_new");

  toFile(fileName);
}

void AdtCataTextures::toFile(const std::string & fileName)
{
  std::vector<char> wholeAdt(0);
  
  std::vector<char> tempData(texturesMver.getWholeChunk());
  wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());
  
  tempData = mamp.getWholeChunk();
  wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());

  tempData = mtex.getWholeChunk();
  wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());

  int currentMcnk;

  for (currentMcnk = 0 ; currentMcnk < 256 ; ++currentMcnk)
  {
    tempData =  texturesMcnks[currentMcnk].getWholeChunk();
    wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());
  }

  if (!mtxf.isEmpty())
  {
    tempData = mtxf.getWholeChunk();
    wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());
  }

  if (!mtxp.isEmpty())
  {
    tempData = mtxp.getWholeChunk();
    wholeAdt.insert(wholeAdt.end(), tempData.begin(), tempData.end());
  }

  std::ofstream outputFile (fileName.c_str(), std::ios::out|std::ios::binary);

  if (outputFile.is_open())
    outputFile.write((char *)&wholeAdt[0], sizeof(char) * wholeAdt.size());

  outputFile.close();
}

std::ostream & operator<<(std::ostream & os, const AdtCataTextures & adtCataTextures)
{
  os << "------------------------------" << std::endl;
  os << adtCataTextures.adtName << std::endl;
  os << "------------------------------" << std::endl;
  os << adtCataTextures.texturesMver;
  os << adtCataTextures.mamp;
  os << adtCataTextures.mtex;

  std::vector<McnkCataTextures>::const_iterator mcnksTexturesIter;
  int i (0);

  for (mcnksTexturesIter = adtCataTextures.texturesMcnks.begin() ; mcnksTexturesIter != adtCataTextures.texturesMcnks.end() ; ++mcnksTexturesIter)
  {
    os << "MCNK (textures) #" << i << " : " << std::endl;
    os << *mcnksTexturesIter;
    ++i;
  }

  os << adtCataTextures.mtxf;
  os << adtCataTextures.mtxp;

  std::vector<Chunk>::const_iterator texturesUnknownIter;

  for (texturesUnknownIter = adtCataTextures.texturesUnknown.begin() ; texturesUnknownIter != adtCataTextures.texturesUnknown.end() ; ++texturesUnknownIter)
  {
    os << *texturesUnknownIter;
  }     
  
  return os;
}