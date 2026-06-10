#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>
#include <wowfiles/Mmdx.h>
#include <utilities/Utilities.h>

Mmdx::Mmdx() : Chunk()
{
}

Mmdx::Mmdx(std::ifstream & adtFile, int offsetInFile) : Chunk(adtFile, offsetInFile)
{
}

Mmdx::Mmdx(const std::vector<char> & wholeFile, int offsetInFile) : Chunk(wholeFile, offsetInFile)
{
}

Mmdx::Mmdx(std::string letters, int givenSize, const std::vector<char> & data) : Chunk("XDMM", givenSize, data)
{
}

Mmdx::Mmdx(const std::vector<int> & indices, const std::vector<std::string> & allFileNames)
{
  letters = "XDMM";

  std::vector<std::string> files (0);

  std::vector<int>::const_iterator indicesIter;
  
  for ( indicesIter = indices.begin() ; indicesIter != indices.end() ; ++indicesIter )
  {
	  files.push_back( allFileNames[*indicesIter] );
	}

  for ( int currentFile = 0 ; currentFile < files.size() ; ++currentFile )
  {
    for ( int i = 0 ; i < files[currentFile].size() ; ++i )
    {
      data.push_back( files[currentFile][i] );
    }
    data.push_back( 0x0 );
	}

  givenSize = data.size();
}

std::vector<int> Mmdx::getIndicesForMmid() const
{
  std::vector<int> mmidData (0);
  mmidData.push_back( 0x0 );

  for ( int currentChar = 0 ; currentChar < data.size() ; ++currentChar )
  {
    if ( data[currentChar] == 0x0 )
    {
      mmidData.push_back( currentChar + 1 );
    }
	}

  return mmidData;
}

std::vector<std::string> Mmdx::getM2Names() const
{
  return Utilities::getFileNames( data );
}

std::ostream & operator<<(std::ostream & os, const Mmdx & mmdx)
{
  os << "Chunk letters : " << mmdx.letters << std::endl;
  os << "Chunk givenSize : " << mmdx.givenSize << std::endl;
  os << "File names in MMDX : " << std::endl;

  std::vector<std::string> mmdxFiles = Utilities::getFileNames(mmdx.data);

  std::vector<std::string>::iterator filesIter;

  for (filesIter = mmdxFiles.begin() ; filesIter != mmdxFiles.end() ; ++filesIter)
  {
    os << *filesIter << std::endl;
  }

  os << "------------------------------" << std::endl;
  return os;
}