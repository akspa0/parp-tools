#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <wowfiles/Wdt.h>
#include <wowfiles/Mphd.h>
#include <wowfiles/Chunk.h>
#include <utilities/Utilities.h>

Wdt::Wdt(const std::vector<char> & wdtFile, const std::string & wdtFileName) : wdtName(wdtFileName)
{
  int offsetInFile (0);
  int currentChunkSize (0);

  mver = Chunk(wdtFile, offsetInFile);
  offsetInFile += 4;
  currentChunkSize = Utilities::get<int>(wdtFile, offsetInFile);
  offsetInFile = 4 + offsetInFile + currentChunkSize;

  mphd = Mphd(wdtFile, offsetInFile);
  offsetInFile += 4;
  currentChunkSize = Utilities::get<int>(wdtFile, offsetInFile);
  offsetInFile = 4 + offsetInFile + currentChunkSize;

  main = Chunk(wdtFile, offsetInFile);
  offsetInFile += 4;
  currentChunkSize = Utilities::get<int>(wdtFile, offsetInFile);
  offsetInFile = 4 + offsetInFile + currentChunkSize;

  if (Utilities::flagsExist(mphd.getFlags(), 0x1))
  {
    mwmo = Chunk(wdtFile, offsetInFile);
    offsetInFile += 4;
    currentChunkSize = Utilities::get<int>(wdtFile, offsetInFile);
    offsetInFile = 4 + offsetInFile + currentChunkSize;
  }

  if (mwmo.getRealSize() != 0)
  {
    modf = Chunk(wdtFile, offsetInFile);
    offsetInFile += 4;
    currentChunkSize = Utilities::get<int>(wdtFile, offsetInFile);
    offsetInFile = 4 + offsetInFile + currentChunkSize;
  }
}

Wdt::Wdt(const std::string & name
  , const Chunk & cMver
  , const Mphd & cMphd
  , const Chunk & cMain
  , const Chunk & cMwmo
  , const Chunk & cModf
  ) : wdtName(name)
  , mver(cMver)
  , mphd(cMphd)
  , main(cMain)
  , mwmo(cMwmo)
  , modf(cModf)
{
}

void Wdt::toFile()
{
  std::vector<char> wholeWdt(mver.getWholeChunk());

  std::vector<char> tempData(mphd.getWholeChunk());
  wholeWdt.insert(wholeWdt.end(), tempData.begin(), tempData.end());

  tempData = main.getWholeChunk();
  wholeWdt.insert(wholeWdt.end(), tempData.begin(), tempData.end());

  if (!mwmo.isEmpty())
  {
    tempData = mwmo.getWholeChunk();
    wholeWdt.insert(wholeWdt.end(), tempData.begin(), tempData.end());
  }

  if (!modf.isEmpty())
  {
    tempData = modf.getWholeChunk();
    wholeWdt.insert(wholeWdt.end(), tempData.begin(), tempData.end());
  }

  std::string fileName (wdtName);
  fileName.append("_new");
  std::ofstream outputFile (fileName.c_str(), std::ios::out|std::ios::binary);

  if (outputFile.is_open())
    outputFile.write((char *)&wholeWdt[0], sizeof(char) * wholeWdt.size());

  outputFile.close();
}

std::ostream & operator<<(std::ostream & os, const Wdt & wdt)
{
  os << wdt.wdtName << std::endl;
  os << "------------------------------" << std::endl;
  os << wdt.mver;
  os << wdt.mphd;
  os << wdt.main;
  os << wdt.mwmo;
  os << wdt.modf;

  return os;
}
