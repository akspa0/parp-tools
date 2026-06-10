#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>
#include <wowfiles/Mcnk.h>
#include <wowfiles/cataclysm/McnkCataTextures.h>
#include <utilities/Utilities.h>

McnkCataTextures::McnkCataTextures(const std::vector<char> & adtFile, int offsetInFile) : Mcnk(adtFile, offsetInFile, 0)
{
  const int absoluteMcnkEnd = offsetInFile + chunkLettersAndSize + givenSize;

  offsetInFile = chunkLettersAndSize + offsetInFile;
  
  int chunkName;
  int currentChunkSize (0);

  while (offsetInFile < absoluteMcnkEnd)
  {
    chunkName = Utilities::get<int>(adtFile, offsetInFile);

    switch (chunkName)
    {
      case 'MCLY' :
        mcly = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;  

      case 'MCSH' :
        mcsh = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break; 

      case 'MCAL' :
        mcal = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;  		

      case 'MCMT' :
        mcmt = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;  

      default :
        texturesMcnkUnknown.push_back(Chunk(adtFile, offsetInFile));
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
    }
  }	
}

std::vector<char> McnkCataTextures::getWholeChunk() const
{
  std::vector<char> wholeChunk (0);

  std::vector<char> tempData (letters.begin(), letters.end());
  wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());

  tempData = Utilities::getCharVectorFromInt(givenSize);
  wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());
  
  tempData = mcly.getWholeChunk();
  wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());

  if (!mcsh.isEmpty())
  {
    tempData = mcsh.getWholeChunk();
    wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());
  }

  tempData = mcal.getWholeChunk();
  wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());

  if (!mcmt.isEmpty())
  {
    tempData = mcmt.getWholeChunk();
    wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());
  }

  return wholeChunk;
}

std::ostream & operator<<(std::ostream & os, const McnkCataTextures & mcnkCataTextures)
{
  os << "Chunk letters : " << mcnkCataTextures.letters << std::endl;
  os << "Chunk givenSize : " << mcnkCataTextures.givenSize << std::endl;

  os << "------------------------------" << std::endl;

  os << mcnkCataTextures.mcly;
  os << mcnkCataTextures.mcsh;
  os << mcnkCataTextures.mcal;
  os << mcnkCataTextures.mcmt;

  os << "Hi, in here we're unknown !" << std::endl;
  os << std::endl;

  std::vector<Chunk>::const_iterator unknownIter;

  for (unknownIter = mcnkCataTextures.texturesMcnkUnknown.begin() ; unknownIter != mcnkCataTextures.texturesMcnkUnknown.end() ; ++unknownIter)
  {
    os << *unknownIter;
  }  
  
  os << "------------------------------" << std::endl;
  return os;
}