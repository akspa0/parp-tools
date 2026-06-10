#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>
#include <wowfiles/Mwmo.h>
#include <utilities/Utilities.h>

Mwmo::Mwmo() : Chunk()
{
}

Mwmo::Mwmo(std::ifstream & adtFile, int offsetInFile) : Chunk(adtFile, offsetInFile)
{
}

Mwmo::Mwmo(const std::vector<char> & wholeFile, int offsetInFile) : Chunk(wholeFile, offsetInFile)
{
}

Mwmo::Mwmo(std::string letters, int givenSize, const std::vector<char> & data) : Chunk("OMWM", givenSize, data)
{
}

Mwmo::Mwmo(const std::vector<int> & indices, const std::vector<std::string> & allFileNames)
{
  letters = "OMWM";

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

std::vector<int> Mwmo::getIndicesForMwid() const
{
  std::vector<int> mwidData (0);
  mwidData.push_back( 0x0 );

  for ( int currentChar = 0 ; currentChar < data.size() ; ++currentChar )
  {
    if ( data[currentChar] == 0x0 )
    {
      mwidData.push_back( currentChar + 1 );
    }
	}

  return mwidData;
}

std::vector<std::string> Mwmo::getWmoNames() const
{
  return Utilities::getFileNames( data );
}

std::ostream & operator<<(std::ostream & os, const Mwmo & mwmo)
{
  os << "Chunk letters : " << mwmo.letters << std::endl;
  os << "Chunk givenSize : " << mwmo.givenSize << std::endl;
  os << "File names in MWMO : " << std::endl;

  std::vector<std::string> mwmoFiles = Utilities::getFileNames(mwmo.data);

  std::vector<std::string>::iterator filesIter;

  for (filesIter = mwmoFiles.begin() ; filesIter != mwmoFiles.end() ; ++filesIter)
  {
    os << *filesIter << std::endl;
  }

  os << std::endl;

  os << "------------------------------" << std::endl;
  return os;
}