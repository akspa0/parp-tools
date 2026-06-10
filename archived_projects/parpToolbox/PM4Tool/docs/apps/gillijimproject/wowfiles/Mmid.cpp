#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>
#include <wowfiles/Mmid.h>
#include <utilities/Utilities.h>

Mmid::Mmid() : Chunk()
{
}

Mmid::Mmid(std::ifstream & adtFile, int offsetInFile) : Chunk(adtFile, offsetInFile)
{
}

Mmid::Mmid(const std::vector<char> & wholeFile, int offsetInFile) : Chunk(wholeFile, offsetInFile)
{
}

Mmid::Mmid(std::string letters, int givenSize, const std::vector<char> & data) : Chunk("DIMM", givenSize, data)
{
}

Mmid::Mmid( const std::vector<int> & indicesFromMmdx )
{
  letters = "DIMM";

  for ( int currentIndex = 0 ; currentIndex < indicesFromMmdx.size() ; ++currentIndex )
  {
      std::vector<char> temp ( Utilities::getCharVectorFromInt( indicesFromMmdx[currentIndex] ) );
      data.insert( data.end(), temp.begin(), temp.end() );
	}

  givenSize = data.size();
}

std::ostream & operator<<(std::ostream & os, const Mmid & mmid)
{
  os << "Chunk letters : " << mmid.letters << std::endl;
  os << "Chunk givenSize : " << mmid.givenSize << std::endl;
  os << "Indices in Mmid : ";

  std::vector<int> mmidIndices = Utilities::vecTo<int>( mmid.data ) ;

  std::vector<int>::iterator indicesIter;

  for (indicesIter = mmidIndices.begin() ; indicesIter != mmidIndices.end() ; ++indicesIter)
  {
    os << *indicesIter << " ";
  }

  os << std::endl;

  os << "------------------------------" << std::endl;
  return os;
}