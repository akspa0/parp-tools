#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>
#include <wowfiles/Mwid.h>
#include <utilities/Utilities.h>

Mwid::Mwid() : Chunk()
{
}

Mwid::Mwid(std::ifstream & adtFile, int offsetInFile) : Chunk(adtFile, offsetInFile)
{
}

Mwid::Mwid(const std::vector<char> & wholeFile, int offsetInFile) : Chunk(wholeFile, offsetInFile)
{
}

Mwid::Mwid(std::string letters, int givenSize, const std::vector<char> & data) : Chunk("DIWM", givenSize, data)
{
}

Mwid::Mwid( const std::vector<int> & indicesFromMwmo )
{
  letters = "DIWM";

  for ( int currentIndex = 0 ; currentIndex < indicesFromMwmo.size() ; ++currentIndex )
  {
      std::vector<char> temp ( Utilities::getCharVectorFromInt( indicesFromMwmo[currentIndex] ) );
      data.insert( data.end(), temp.begin(), temp.end() );
	}

  givenSize = data.size();
}

std::ostream & operator<<(std::ostream & os, const Mwid & mwid)
{
  os << "Chunk letters : " << mwid.letters << std::endl;
  os << "Chunk givenSize : " << mwid.givenSize << std::endl;
  os << "Indices in Mwid : ";

  std::vector<int> mwidIndices = Utilities::vecTo<int>( mwid.data ) ;

  std::vector<int>::iterator indicesIter;

  for (indicesIter = mwidIndices.begin() ; indicesIter != mwidIndices.end() ; ++indicesIter)
  {
    os << *indicesIter << " ";
  }

  os << std::endl;

  os << "------------------------------" << std::endl;
  return os;
}