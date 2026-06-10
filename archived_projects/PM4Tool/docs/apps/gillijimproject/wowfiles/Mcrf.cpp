#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>
#include <wowfiles/Mcrf.h>
#include <utilities/Utilities.h>

Mcrf::Mcrf() : Chunk()
{
}

Mcrf::Mcrf(std::ifstream & adtFile, int offsetInFile) : Chunk(adtFile, offsetInFile)
{
}

Mcrf::Mcrf(const std::vector<char> & wholeFile, int offsetInFile) : Chunk(wholeFile, offsetInFile)
{
}

Mcrf::Mcrf(std::string letters, int givenSize, const std::vector<char> & data) : Chunk("FRCM", givenSize, data)
{
}

std::vector<int> Mcrf::getDoodadsIndices(const int & doodadsNumber) const
{
  std::vector<int> mcrfIndices ( Utilities::vecTo<int>(data) );
  std::vector<int> doodadsIndices ( mcrfIndices.begin(), mcrfIndices.begin() + doodadsNumber );

  return doodadsIndices;
}

std::vector<int> Mcrf::getWmosIndices(const int & wmosNumber) const
{
  std::vector<int> mcrfIndices ( Utilities::vecTo<int>(data) );
  std::vector<int> wmosIndices ( mcrfIndices.end() - wmosNumber, mcrfIndices.end() );

  return wmosIndices;
}

void Mcrf::updateIndicesForLk( std::vector<int> & alphaM2Indices, int m2Number, std::vector<int> & alphaWmoIndices, int wmoNumber ) // TODO : that thing looks awful. Needs to be totally different (at least it does what it's supposed to do though).
{
  std::vector<int> mcrfAlphaM2Indices( getDoodadsIndices(m2Number) );
  std::vector<int> mcrfAlphaWmoIndices( getWmosIndices(wmoNumber) );

  // TODO : add check to be sure mcrfAlphaM2Indices.size() + mcrfAlphaWmoIndices.size() == data.size()

  for ( unsigned int i = 0 ; i < mcrfAlphaM2Indices.size() ; ++i )
  {
    for ( unsigned int j = 0 ; j < alphaM2Indices.size() ; ++j )
    {
      if ( mcrfAlphaM2Indices[i] == alphaM2Indices[j] )
	    {
        mcrfAlphaM2Indices[i] = j;
	    }
    }
  }

  for ( unsigned int i = 0 ; i < mcrfAlphaWmoIndices.size() ; ++i )
  {
    for ( unsigned int j = 0 ; j < alphaWmoIndices.size() ; ++j )
    {
      if ( mcrfAlphaWmoIndices[i] == alphaWmoIndices[j] )
	    {
        mcrfAlphaWmoIndices[i] = j;
	    }
    }
  }

  std::vector<char> newMcrfData (0);

  std::vector<int>::const_iterator indicesIter;
  
  for ( indicesIter = mcrfAlphaM2Indices.begin() ; indicesIter != mcrfAlphaM2Indices.end() ; ++indicesIter )
  {
    std::vector<char> currentIndex ( Utilities::getCharVectorFromInt( *indicesIter ) );
    for ( unsigned int i = 0 ; i < currentIndex.size() ; ++i )
    {
      newMcrfData.push_back( currentIndex[i] );
    }
  } 

  for ( indicesIter = mcrfAlphaWmoIndices.begin() ; indicesIter != mcrfAlphaWmoIndices.end() ; ++indicesIter )
  {
    std::vector<char> currentIndex ( Utilities::getCharVectorFromInt( *indicesIter ) );
    for ( unsigned  int i = 0 ; i < currentIndex.size() ; ++i )
    {
      newMcrfData.push_back( currentIndex[i] );
    }
  }
  
  data = newMcrfData;
}

std::ostream & operator<<(std::ostream & os, const Mcrf & mcrf)
{
  os << "Chunk letters : " << mcrf.letters << std::endl;
  os << "Chunk givenSize : " << mcrf.givenSize << std::endl;
  os << "Chunk full content : ";

  if ( mcrf.data.size() != 0 )
  {
    std::vector<int> mcrfIndices ( Utilities::vecTo<int>(mcrf.data) );

    std::vector<int>::const_iterator mcrfContentIter;

    for (mcrfContentIter = mcrfIndices.begin() ; mcrfContentIter != mcrfIndices.end() ; ++mcrfContentIter)
    {
      os << *mcrfContentIter << " ";
    }
  }

  os << std::endl;

  os << "------------------------------" << std::endl;
  return os;
}