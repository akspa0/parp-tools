#include <vector>
#include <iostream>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>
#include <wowfiles/ChunkHeaders.h>
#include <wowfiles/alpha/McnkAlpha.h>
#include <wowfiles/alpha/McnrAlpha.h>
#include <wowfiles/alpha/McvtAlpha.h>
#include <wowfiles/lichking/McnrLk.h>
#include <wowfiles/Mcrf.h>
#include <utilities/Utilities.h>

McnkAlpha::McnkAlpha(std::ifstream & wdtAlphaFile, int offsetInFile, const int headerSize, int adtNum) : Mcnk(wdtAlphaFile, offsetInFile, mcnkTerrainHeaderSize), adtNumber(adtNum)
{  
  const int headerStartOffset (offsetInFile);
  const int mcvtSize (580);
  const int mcnrSize (448);

  offsetInFile = chunkLettersAndSize + offsetInFile;

  wdtAlphaFile.seekg(offsetInFile, std::ios::beg);
  char * dataBuffer = new char[mcnkTerrainHeaderSize];
  wdtAlphaFile.read(dataBuffer, mcnkTerrainHeaderSize);
  mcnkAlphaHeader = *reinterpret_cast<McnkAlphaHeader*>(dataBuffer);
  delete[] dataBuffer;

  offsetInFile = headerStartOffset + mcnkTerrainHeaderSize + chunkLettersAndSize + mcnkAlphaHeader.mcvtOffset;
  std::vector<char> mcvtData (Utilities::getCharVectorFromFile(wdtAlphaFile, offsetInFile, mcvtSize));
  mcvt = McvtAlpha("TVCM", mcvtSize, mcvtData);

  offsetInFile = headerStartOffset + mcnkTerrainHeaderSize + chunkLettersAndSize + mcnkAlphaHeader.mcnrOffset;
  std::vector<char> mcnrData (Utilities::getCharVectorFromFile(wdtAlphaFile, offsetInFile, mcnrSize));
  mcnrAlpha = McnrAlpha("RNCM", mcnrSize, mcnrData);

  offsetInFile = headerStartOffset + mcnkTerrainHeaderSize + chunkLettersAndSize + mcnkAlphaHeader.mclyOffset;
  mcly = Chunk(wdtAlphaFile, offsetInFile);

  offsetInFile = headerStartOffset + mcnkTerrainHeaderSize + chunkLettersAndSize + mcnkAlphaHeader.mcrfOffset;
  mcrf = Mcrf(wdtAlphaFile, offsetInFile);

  offsetInFile = headerStartOffset + mcnkTerrainHeaderSize + chunkLettersAndSize + mcnkAlphaHeader.mcshOffset;
  std::vector<char> mcshData (Utilities::getCharVectorFromFile(wdtAlphaFile, offsetInFile, mcnkAlphaHeader.mcshSize));
  mcsh = Chunk("HSCM", mcnkAlphaHeader.mcshSize, mcshData);

  offsetInFile = headerStartOffset + mcnkTerrainHeaderSize + chunkLettersAndSize + mcnkAlphaHeader.mcalOffset;
  std::vector<char> mcalData (Utilities::getCharVectorFromFile(wdtAlphaFile, offsetInFile, mcnkAlphaHeader.mcalSize));
  mcal = Mcal("LACM", mcnkAlphaHeader.mcalSize, mcalData);

  offsetInFile = headerStartOffset + mcnkTerrainHeaderSize + chunkLettersAndSize + mcnkAlphaHeader.mclqOffset;
  const int mclqSize (mcnkAlphaHeader.mcnkChunksSize - mcnkAlphaHeader.mclqOffset);
  std::vector<char> mclqData (Utilities::getCharVectorFromFile(wdtAlphaFile, offsetInFile, mclqSize));
  mclq = Chunk("QLCM", mclqSize, mclqData);
}

McnkLk McnkAlpha::toMcnkLk( std::vector<int> alphaM2Indices, std::vector<int> alphaWmoIndices ) const
{
  McnkHeader (cMcnkHeader);  
  int offsetInHeader (chunkLettersAndSize + mcnkTerrainHeaderSize);

  cMcnkHeader.flags = mcnkAlphaHeader.flags;
  cMcnkHeader.indexX = mcnkAlphaHeader.indexX;
  cMcnkHeader.indexY = mcnkAlphaHeader.indexY;
  cMcnkHeader.nLayers = mcnkAlphaHeader.nLayers;
  cMcnkHeader.m2Number = mcnkAlphaHeader.m2Number;
  
  cMcnkHeader.mcvtOffset = offsetInHeader;
  offsetInHeader = offsetInHeader + chunkLettersAndSize + mcvt.getRealSize();  
  
  cMcnkHeader.mcnrOffset = offsetInHeader;  
  offsetInHeader = offsetInHeader + chunkLettersAndSize + mcnrAlpha.getRealSize();  
  
  cMcnkHeader.mclyOffset = offsetInHeader; 
  offsetInHeader = offsetInHeader + chunkLettersAndSize + mcly.getRealSize();
  
  cMcnkHeader.mcrfOffset = offsetInHeader;   
  const int mcshOffset (offsetInHeader + chunkLettersAndSize + mcrf.getRealSize());  
  offsetInHeader = mcshOffset + chunkLettersAndSize + mcsh.getRealSize();  
  
  cMcnkHeader.mcalOffset = offsetInHeader;
  offsetInHeader = offsetInHeader + chunkLettersAndSize + mcal.getRealSize();
  const int mclqOffset (offsetInHeader);
  
  cMcnkHeader.mcalSize = mcal.getRealSize() + chunkLettersAndSize;  
  offsetInHeader = mcshOffset;
  
  cMcnkHeader.mcshOffset = mcshOffset;
  cMcnkHeader.mcshSize = mcsh.getRealSize();
  cMcnkHeader.areaId = mcnkAlphaHeader.unknown3; // TODO : I don't know... AreaID should be here, but results are not really convincing -_-
  cMcnkHeader.wmoNumber = mcnkAlphaHeader.wmoNumber;
  cMcnkHeader.holes = mcnkAlphaHeader.holes;
  cMcnkHeader.groundEffectsMap1 = mcnkAlphaHeader.groundEffectsMap1;    
  cMcnkHeader.groundEffectsMap2 = mcnkAlphaHeader.groundEffectsMap2; 
  cMcnkHeader.groundEffectsMap3 = mcnkAlphaHeader.groundEffectsMap3; 
  cMcnkHeader.groundEffectsMap4 = mcnkAlphaHeader.groundEffectsMap4;   
  cMcnkHeader.predTex = 0; 
  cMcnkHeader.nEffectDoodad = 0;  
  cMcnkHeader.mcseOffset = 0;    
  cMcnkHeader.nSndEmitters = 0; 
  cMcnkHeader.mclqOffset = mclqOffset; 
  
  if (mclq.getRealSize() != 0) 
    cMcnkHeader.mclqSize = mclq.getRealSize() + chunkLettersAndSize; 
  else 
    cMcnkHeader.mclqSize = 0;

  // ------------- junk

  int adtX (adtNumber % 64);
  int adtY (adtNumber / 64);

  cMcnkHeader.posY = ( ( ( (533.33333 / 16) * cMcnkHeader.indexX) + (533.33333 * adtX) ) - (533.33333 * 32) ) * -1; // TODO : make this prettier and more consistent (X - Y inversions look baaad)
  cMcnkHeader.posX = ( ( ( (533.33333 / 16) * cMcnkHeader.indexY) + (533.33333 * adtY) ) - (533.33333 * 32) ) * -1; // TODO : make this prettier and more consistent (X - Y inversions look baaad)

  // ------------- end junk

  cMcnkHeader.posZ = 0;
  cMcnkHeader.mccvOffset = 0;
  cMcnkHeader.mclvOffset = 0;
  cMcnkHeader.unused = 0;  
  
  std::vector<char> emptyData (0); 
  Chunk (emptyChunk); 

  McnrLk cMcnr (mcnrAlpha.toMcnrLk());  
  
  Chunk cMcvt ("TVCM", 0, emptyData);
  cMcvt = mcvt.toMcvt();

  Mcrf cMcrf (mcrf);
  cMcrf.updateIndicesForLk( alphaM2Indices, mcnkAlphaHeader.m2Number, alphaWmoIndices, mcnkAlphaHeader.wmoNumber );

  McnkLk mcnkLk = McnkLk(cMcnkHeader, cMcvt, emptyChunk, cMcnr, mcly, cMcrf, mcsh, mcal, mclq, emptyChunk);
  return mcnkLk;
}

std::ostream & operator<<(std::ostream & os, const McnkAlpha & mcnkAlpha)
{
  os << "Chunk letters : " << mcnkAlpha.letters << std::endl;
  os << "Chunk givenSize : " << mcnkAlpha.givenSize << std::endl;

  os << "------------------------------" << std::endl;

  os << "Header : " << std::endl;
  
  // TODO : make this look nicer. Somehow. 
 
  os << "#0x00 Flags \t\t\t\t\t\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.flags << std::dec << std::endl;
  os << "#0x04 IndexX\t\t\t\t\t\t: " << mcnkAlpha.mcnkAlphaHeader.indexX << std::dec << std::endl;
  os << "#0x08 IndexY\t\t\t\t\t\t: " << mcnkAlpha.mcnkAlphaHeader.indexY << std::dec << std::endl;
  os << "#0x0C I don't know\t\t\t\t\t: " << mcnkAlpha.mcnkAlphaHeader.unknown1 << std::endl; 
  os << "#0x10 Layers number\t\t\t\t\t: " << mcnkAlpha.mcnkAlphaHeader.nLayers << std::endl;
  os << "#0x14 Doodads number\t\t\t\t: " << mcnkAlpha.mcnkAlphaHeader.m2Number << std::endl; 
  os << "#0x18 MCVT offset\t\t\t\t\t: 0x" << mcnkAlpha.mcnkAlphaHeader.mcvtOffset << std::dec << std::endl; 
  os << "#0x1C MCNR offset\t\t\t\t\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.mcnrOffset << std::dec << std::endl; 
  os << "#0x20 MCLY offset\t\t\t\t\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.mclyOffset << std::dec << std::endl;
  os << "#0x24 MCRF offset\t\t\t\t\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.mcrfOffset << std::dec << std::endl; 
  os << "#0x28 MCAL offset\t\t\t\t\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.mcalOffset << std::dec << std::endl; 
  os << "#0x2C MCAL size\t\t\t\t\t\t: " << mcnkAlpha.mcnkAlphaHeader.mcalSize << std::dec << std::endl; 
  os << "#0x30 MCSH offset\t\t\t\t\t: 0x" << mcnkAlpha.mcnkAlphaHeader.mcshOffset << std::dec << std::endl;
  os << "#0x34 MCSH size\t\t\t\t\t\t: " << mcnkAlpha.mcnkAlphaHeader.mcshSize << std::dec << std::endl; 
  os << "#0x38 -> #0x3B\t\t\t\t\t\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.unknown3 << std::dec << std::endl; 
  os << "#0x3C Wmo number\t\t\t\t\t: " << mcnkAlpha.mcnkAlphaHeader.wmoNumber << std::endl;
  os << "#0x40 -> Holes #0x43\t\t\t\t\t\t: 0x" << mcnkAlpha.mcnkAlphaHeader.holes << std::dec << std::endl;
  os << "#0x44 -> #0x47 Low Texturing map\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.groundEffectsMap1 << std::dec << std::endl;
  os << "#0x48 -> #0x4B Low Texturing map\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.groundEffectsMap2 << std::dec << std::endl;
  os << "#0x4C -> #0x4F Low Texturing map\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.groundEffectsMap3 << std::dec << std::endl;
  os << "#0x50 -> #0x53 Low Texturing map\t: 0x" << mcnkAlpha.mcnkAlphaHeader.groundEffectsMap4 << std::dec << std::endl;
  os << "#0x54 -> #0x57\t\t\t\t\t\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.unknown6 << std::dec << std::endl; 
  os << "#0x58 -> #0x5B\t\t\t\t\t\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.unknown7 << std::dec << std::endl; 
  os << "#0x5C MCNK size (minus header)\t\t: " << mcnkAlpha.mcnkAlphaHeader.mcnkChunksSize << std::endl; 
  os << "#0x60 -> #0x63\t\t\t\t\t\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.unknown8 << std::dec << std::endl;
  os << "#0x64 MCLQ offset\t\t\t\t\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.mclqOffset << std::dec << std::endl; 
  os << "#0x68 -> #0x6B\t\t\t\t\t\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.unused1 << std::dec << std::endl; 
  os << "#0x6C -> #0x6F\t\t\t\t\t\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.unused2 << std::dec << std::endl; 
  os << "#0x70 -> #0x73\t\t\t\t\t\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.unused3 << std::dec << std::endl;
  os << "#0x74 -> #0x77\t\t\t\t\t\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.unused4 << std::dec << std::endl; 
  os << "#0x78 -> #0x7B\t\t\t\t\t\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.unused5 << std::dec << std::endl; 
  os << "#0x7C -> #0x7F\t\t\t\t\t\t: 0x" << std::hex << mcnkAlpha.mcnkAlphaHeader.unused6 << std::dec << std::endl;

  os << "------------------------------" << std::endl;

  os << mcnkAlpha.mcvt;
  os << mcnkAlpha.mcnrAlpha;
  os << mcnkAlpha.mcly;
  os << mcnkAlpha.mcrf;

  std::vector<int>::const_iterator mcrfContentIter;

  std::vector<int> mcrfDoodads ( mcnkAlpha.mcrf.getDoodadsIndices( mcnkAlpha.mcnkAlphaHeader.m2Number ) );
  os << "Doodads indices : ";
  for (mcrfContentIter = mcrfDoodads.begin() ; mcrfContentIter != mcrfDoodads.end() ; ++mcrfContentIter)
  {
    os << *mcrfContentIter << " ";
  }
  os << std::endl;

  std::vector<int> mcrfWmos ( mcnkAlpha.mcrf.getWmosIndices( mcnkAlpha.mcnkAlphaHeader.wmoNumber ) );
  os << "Wmos indices : ";
  for (mcrfContentIter = mcrfWmos.begin() ; mcrfContentIter != mcrfWmos.end() ; ++mcrfContentIter)
  {
    os << *mcrfContentIter << " ";
  }
  os << std::endl;
  os << "------------------------------" << std::endl;

  os << mcnkAlpha.mcsh;
  os << mcnkAlpha.mcal;
  os << mcnkAlpha.mclq;

  os << "------------------------------" << std::endl;
  return os;
}
