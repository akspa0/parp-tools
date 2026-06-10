#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>
#include <wowfiles/Mcnk.h>
#include <wowfiles/cataclysm/McnkCataObjects.h>
#include <utilities/Utilities.h>

McnkCataObjects::McnkCataObjects(const std::vector<char> & adtFile, int offsetInFile) : Mcnk(adtFile, offsetInFile, 0)
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
      case 'MCRD' :
        mcrd = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;  

      case 'MCRW' :
        mcrw = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;

      default :
        objectsMcnkUnknown.push_back(Chunk(adtFile, offsetInFile));
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
    }
  }
}

std::vector<char> McnkCataObjects::getWholeChunk() const
{
  std::vector<char> wholeChunk (0);

  std::vector<char> tempData (letters.begin(), letters.end());
  wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());

  tempData = Utilities::getCharVectorFromInt(givenSize);
  wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());

  if (!mcrd.isEmpty())
  {
    tempData = mcrd.getWholeChunk();
    wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());
  }

  if (!mcrw.isEmpty())
  {
    tempData = mcrw.getWholeChunk();
    wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());
  }

  return wholeChunk;
}

std::ostream & operator<<(std::ostream & os, const McnkCataObjects & mcnkCataObjects)
{
  os << "Chunk letters : " << mcnkCataObjects.letters << std::endl;
  os << "Chunk givenSize : " << mcnkCataObjects.givenSize << std::endl;

  os << "------------------------------" << std::endl;

  os << mcnkCataObjects.mcrd;
  os << mcnkCataObjects.mcrw;

  os << "Hi, in here we're unknown !" << std::endl;
  os << std::endl;

  std::vector<Chunk>::const_iterator unknownIter;

  for (unknownIter = mcnkCataObjects.objectsMcnkUnknown.begin() ; unknownIter != mcnkCataObjects.objectsMcnkUnknown.end() ; ++unknownIter)
  {
    os << *unknownIter;
  }  
  
  os << "------------------------------" << std::endl;
  return os;
}