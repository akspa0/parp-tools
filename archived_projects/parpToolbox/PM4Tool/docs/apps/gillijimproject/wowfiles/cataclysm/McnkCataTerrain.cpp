#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>
#include <wowfiles/Mcnk.h>
#include <wowfiles/cataclysm/McnkCataTerrain.h>
#include <wowfiles/cataclysm/McnrCata.h>
#include <wowfiles/lichking/McnkLk.h>
#include <utilities/Utilities.h>

McnkCataTerrain::McnkCataTerrain(const std::vector<char> & adtFile, int offsetInFile, const int & headerSize) : Mcnk(adtFile, offsetInFile, mcnkTerrainHeaderSize)
{
  const int headerStartOffset (offsetInFile + chunkLettersAndSize);
  const int absoluteMcnkEnd = offsetInFile + chunkLettersAndSize + givenSize;

  offsetInFile = chunkLettersAndSize + offsetInFile;

  std::vector<char> headerContent (0);
  headerContent.assign( adtFile.begin() + offsetInFile, adtFile.begin() + offsetInFile + mcnkTerrainHeaderSize );
  mcnkHeader = *reinterpret_cast<McnkHeader*>(&headerContent[0]);

  offsetInFile = headerStartOffset + mcnkTerrainHeaderSize;
  
  int chunkName;
  int currentChunkSize (0);

  while (offsetInFile < absoluteMcnkEnd)
  {
    chunkName = Utilities::get<int>(adtFile, offsetInFile);

    switch (chunkName)
    {
      case 'MCVT' :
        mcvt = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;  

      case 'MCBB' :
        mcbb = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;  

      case 'MCBI' :
        mcbi = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;  

      case 'MCBV' :
        mcbv = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;  

      case 'MCCV' :
        mccv = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;  		

      case 'MCLV' :
        mclv = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;  

      case 'MCNR' :
        mcnr = McnrCata(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;  		

      case 'MCLQ' :
        mclq = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break;  
		
      case 'MCSE' :
        mcse = Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        break; 	
		
      default :
        terrainMcnkUnknown.push_back(Chunk(adtFile, offsetInFile));
        offsetInFile += 4;
        currentChunkSize = Utilities::get<int>(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
    }
  }		
}

void McnkCataTerrain::addToRelativeHeight(const int & heightToAdd)
{
  mcnkHeader.posZ += heightToAdd;
}

std::vector<char> McnkCataTerrain::getWholeChunk() const
{
  std::vector<char> wholeChunk (0);

  std::vector<char> tempData (letters.begin(), letters.end());
  wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());

  const char* temp = reinterpret_cast<const char*>(&mcnkHeader);
  std::vector<char> headerContent ( mcnkTerrainHeaderSize );
  headerContent.assign(temp, temp + mcnkTerrainHeaderSize );
  wholeChunk.insert(wholeChunk.end(), headerContent.begin(), headerContent.end());

  tempData = mcvt.getWholeChunk();
  wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());

  if (!mcbb.isEmpty())
  {
    tempData = mcbb.getWholeChunk();
    wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());
  }  

  if (!mcbi.isEmpty())
  {
    tempData = mcbi.getWholeChunk();
    wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());
  }  

  if (!mcbv.isEmpty())
  {
    tempData = mcbv.getWholeChunk();
    wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());
  } 

  if (!mccv.isEmpty())
  {
    tempData = mccv.getWholeChunk();
    wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());
  }

  if (!mclv.isEmpty())
  {
    tempData = mclv.getWholeChunk();
    wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());
  }

  tempData = mcnr.getWholeChunk();
  wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());

  if (!mclq.isEmpty())
  {
    tempData = mclq.getWholeChunk();
    wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());
  }

  if (!mcse.isEmpty())
  {
    tempData = mcse.getWholeChunk();
    wholeChunk.insert(wholeChunk.end(), tempData.begin(), tempData.end());
  }  

  int tempSize ( wholeChunk.size() - letters.size() );
  tempData = Utilities::getCharVectorFromInt(tempSize);
  wholeChunk.insert( wholeChunk.begin() + 4, tempData.begin(), tempData.end() );

  return wholeChunk;
}

McnkLk McnkCataTerrain::toMcnkLk() // TODO : probably shouldn't be here.
{
  std::vector<char> emptyData (0); 
  Chunk (emptyChunk); 

  McnrLk cMcnr(mcnr.toMcnrLk());

  Chunk (cMccv);
  if (!mccv.isEmpty())
    cMccv = mccv;

  Chunk cMcly ("YLCM", 0, emptyData);
  Mcrf cMcrf ("FRCM", 0, emptyData);
  Chunk cMcsh ("HSCM", 0, emptyData);
  Mcal (cMcal);

  McnkLk mcnkLk = McnkLk(mcnkHeader, mcvt, cMccv, cMcnr, cMcly, cMcrf, cMcsh, cMcal, emptyChunk, emptyChunk);
  return mcnkLk;
}

std::ostream & operator<<(std::ostream & os, const McnkCataTerrain & mcnkCataTerrain)
{
  os << "Chunk letters : " << mcnkCataTerrain.letters << std::endl;
  os << "Chunk givenSize : " << mcnkCataTerrain.givenSize << std::endl;

  os << std::endl;

  os << "flags : " << std::hex << "0x" << mcnkCataTerrain.mcnkHeader.flags << std::dec << std::endl;
  os << "indexX : " << mcnkCataTerrain.mcnkHeader.indexX << std::endl;
  os << "indexY : " << mcnkCataTerrain.mcnkHeader.indexY << std::endl;
  os << "areaID : " << mcnkCataTerrain.mcnkHeader.areaId << std::endl;
  os << "posX : " << mcnkCataTerrain.mcnkHeader.posX << std::endl;
  os << "posY : " << mcnkCataTerrain.mcnkHeader.posY << std::endl;
  os << "posZ : " << mcnkCataTerrain.mcnkHeader.posZ << std::endl;

  os << "------------------------------" << std::endl;

  os << mcnkCataTerrain.mcvt;
  os << mcnkCataTerrain.mcbb;
  os << mcnkCataTerrain.mcbi;
  os << mcnkCataTerrain.mcbv;
  os << mcnkCataTerrain.mccv;
  os << mcnkCataTerrain.mclv;
  os << mcnkCataTerrain.mcnr;
  os << mcnkCataTerrain.mclq;
  os << mcnkCataTerrain.mcse;

  os << "Hi, in here we're unknown !" << std::endl;
  os << std::endl;

  std::vector<Chunk>::const_iterator unknownIter;

  for (unknownIter = mcnkCataTerrain.terrainMcnkUnknown.begin() ; unknownIter != mcnkCataTerrain.terrainMcnkUnknown.end() ; ++unknownIter)
  {
    os << *unknownIter;
  }
  
  os << "------------------------------" << std::endl;
  return os;
}