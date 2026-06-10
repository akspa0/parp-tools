#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>
#include <wowfiles/alpha/Monm.h>
#include <utilities/Utilities.h>

Monm::Monm() : Chunk()
{
}

Monm::Monm(std::ifstream & wdtAlphaFile, int offsetInFile) : Chunk(wdtAlphaFile, offsetInFile)
{
}

Monm::Monm(std::string letters, int givenSize, const std::vector<char> & data) : Chunk("MNOM", givenSize, data) 
{
}

std::vector<std::string> Monm::getFilesNames() const
{
  return  Utilities::getFileNames(data);
}

Chunk Monm::toMwmo() const
{
  Chunk mwmoLk ("OMWM", givenSize, data);
  return mwmoLk;
}

std::ostream & operator<<(std::ostream & os, const Monm & monm)
{
  os << "Chunk letters : " << monm.letters << std::endl;
  os << "Chunk givenSize : " << monm.givenSize << std::endl;
  os << "File names in MONM : " << std::endl;

  std::vector<std::string> monmFiles ( monm.getFilesNames() );

  std::vector<std::string>::iterator filesIter;

  for (filesIter = monmFiles.begin() ; filesIter != monmFiles.end() ; ++filesIter)
  {
    os << *filesIter << std::endl;
  }

  os << "------------------------------" << std::endl;
  return os;
}