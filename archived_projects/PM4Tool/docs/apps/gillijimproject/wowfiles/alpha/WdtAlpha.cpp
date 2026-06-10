#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <wowfiles/alpha/WdtAlpha.h>
#include <wowfiles/alpha/MphdAlpha.h>
#include <wowfiles/Wdt.h>
#include <wowfiles/alpha/AdtAlpha.h>
#include <wowfiles/alpha/Monm.h>
#include <wowfiles/alpha/Mdnm.h>
#include <wowfiles/Chunk.h>
#include <wowfiles/alpha/MainAlpha.h>
#include <utilities/Utilities.h>

WdtAlpha::WdtAlpha(const std::string & wdtAlphaName) : wdtName(wdtAlphaName)
{
  std::ifstream wdtAlphaFile;
  wdtAlphaFile.open(wdtAlphaName.c_str(), std::ios::binary);

  const int mdnmOffset (4);
  const int monmOffset (12);

  int offsetInFile (0);
  int currentChunkSize (0);

  mver = Chunk(wdtAlphaFile, offsetInFile);
  offsetInFile += 4;
  currentChunkSize = Utilities::getIntFromFile(wdtAlphaFile, offsetInFile);
  offsetInFile = 4 + offsetInFile + currentChunkSize;

  const int MphdStartOffset (offsetInFile + chunkLettersAndSize);

  mphd = MphdAlpha(wdtAlphaFile, offsetInFile);
  offsetInFile += 4;
  currentChunkSize = Utilities::getIntFromFile(wdtAlphaFile, offsetInFile);
  offsetInFile = 4 + offsetInFile + currentChunkSize;

  main = MainAlpha(wdtAlphaFile, offsetInFile);
  offsetInFile += 4;
  currentChunkSize = Utilities::getIntFromFile(wdtAlphaFile, offsetInFile);
  offsetInFile = 4 + offsetInFile + currentChunkSize;

  offsetInFile = Utilities::getIntFromFile(wdtAlphaFile, MphdStartOffset + mdnmOffset);
  mdnm = Mdnm(wdtAlphaFile, offsetInFile);

  offsetInFile = Utilities::getIntFromFile(wdtAlphaFile, MphdStartOffset + monmOffset);
  monm = Monm(wdtAlphaFile, offsetInFile);  

  offsetInFile += 4;
  currentChunkSize = Utilities::getIntFromFile(wdtAlphaFile, offsetInFile);
  offsetInFile = 4 + offsetInFile + currentChunkSize;

  if (mphd.isWmoBased())
  {
    modf = Chunk(wdtAlphaFile, offsetInFile);
    offsetInFile += 4;
    currentChunkSize = Utilities::getIntFromFile(wdtAlphaFile, offsetInFile);
    offsetInFile = 4 + offsetInFile + currentChunkSize;
  }

  wdtAlphaFile.close();
}

Wdt WdtAlpha::toWdt() const
{
  std::string name (wdtName);

  Mphd cMphd (mphd.toMphd());
  Main cMain (main.toMain());

  std::vector<char> emptyData (0);
  Chunk cMwmo ("OMWM", 0, emptyData);
  Chunk(cModf);

  if (mphd.isWmoBased())
  {
    cMwmo = monm.toMwmo();
    cModf = modf;
  }

  Wdt wdtLk = Wdt(name, mver, cMphd, cMain, cMwmo, cModf);
  return wdtLk;
}

std::vector<int> WdtAlpha::getExistingAdtsNumbers() const
{
  std::vector<int> adtsOffsets (main.getMhdrOffsets());

  std::vector<int> existingAdts (0);

  int currentAdt;

  for (currentAdt = 0 ; currentAdt < 4096 ; currentAdt++)
  {  
    if (adtsOffsets[currentAdt] != 0)
    {
      existingAdts.push_back(currentAdt);
    }
  }

  return existingAdts;
}

std::vector<int> WdtAlpha::getAdtOffsetsInMain() const
{
  return main.getMhdrOffsets();
}

std::vector<std::string> WdtAlpha::getMdnmFileNames() const
{
  return mdnm.getFilesNames();
}

std::vector<std::string> WdtAlpha::getMonmFileNames() const
{
  return monm.getFilesNames();
}

std::ostream & operator<<(std::ostream & os, const WdtAlpha & wdtAlpha)
{
  os << wdtAlpha.wdtName << std::endl;
  os << "------------------------------" << std::endl;
  os << wdtAlpha.mver;
  os << wdtAlpha.mphd;
  os << wdtAlpha.main;
  os << wdtAlpha.mdnm;
  os << wdtAlpha.monm;
  os << wdtAlpha.modf;

  return os;
}
