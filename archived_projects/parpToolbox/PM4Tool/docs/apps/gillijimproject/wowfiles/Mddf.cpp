#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>
#include <wowfiles/Mddf.h>
#include <utilities/Utilities.h>

Mddf::Mddf() : Chunk()
{
}

Mddf::Mddf(std::ifstream & adtFile, int offsetInFile) : Chunk(adtFile, offsetInFile)
{
}

Mddf::Mddf(const std::vector<char> & wholeFile, int offsetInFile) : Chunk(wholeFile, offsetInFile)
{
}

Mddf::Mddf(std::string letters, int givenSize, const std::vector<char> & data) : Chunk("FDDM", givenSize, data)
{
}

void Mddf::addToObjectsHeight(const int & heightToAdd)
{
  if (givenSize <= 0) return;

  std::vector<float> mddfHeights( getObjectsHeights() );

  for (int i = 0 ; i < mddfHeights.size() ; ++i)
  {
    mddfHeights[i] += (float)heightToAdd;
  }

  const int entrySize (36);
  std::vector<char> newMddfData (0);

  std::vector<char>::const_iterator dataIter;
  int newIndex (0);
  
  for (dataIter = data.begin() ; dataIter != data.begin() + 12 ; ++dataIter)
  {
    newMddfData.push_back( *dataIter );
  }

  std::vector<char> currentIndex ( Utilities::getCharVectorFromFloat( mddfHeights[newIndex] ) );
  newMddfData.insert( newMddfData.end(), currentIndex.begin(), currentIndex.end() );
  ++newIndex;

  for (dataIter = data.begin() + 16 ; dataIter != data.end() ; ++dataIter)
  {
    if ( ( dataIter - data.begin() + 24 ) % entrySize == 0 )
	  {
      std::vector<char> currentIndex ( Utilities::getCharVectorFromFloat( mddfHeights[newIndex] ) );
      newMddfData.insert( newMddfData.end(), currentIndex.begin(), currentIndex.end() );
      ++newIndex;
	    dataIter += 3;
	  }
    else
    {
      newMddfData.push_back( *dataIter );
    }
  }  
  
  data = newMddfData;
}

std::vector<Utilities::Point> Mddf::getAllM2Coords() const
{
  const int entrySize (36);
  std::vector<Utilities::Point> coords (0);

  std::vector<char>::const_iterator dataIter;
  int currentStart (8);
  
  for (dataIter = data.begin() ; dataIter != data.end() ; ++dataIter)
  {
    if ( ( dataIter - data.begin() + 28 ) % entrySize == 0 )
	  {
      Utilities::Point currentPoint;
      currentPoint.x = Utilities::get<float>( data, currentStart );
      currentPoint.z = Utilities::get<float>( data, currentStart + 4 );
      currentPoint.y = Utilities::get<float>( data, currentStart + 8 );
      coords.push_back( currentPoint );
	    currentStart = ( dataIter - data.begin() ) + entrySize;
	  }
  }

  return coords;
}

std::vector<int> Mddf::getEntriesIndices() const
{
  const int entrySize (36);
  std::vector<int> indices (0);

  std::vector<char>::const_iterator dataIter;
  int currentStart (0);
  
  for (dataIter = data.begin() ; dataIter != data.end() ; ++dataIter)
  {
    if ( ( dataIter - data.begin() ) % entrySize == 0 )
	  {
      indices.push_back( Utilities::get<int> ( data, currentStart ) );
	    currentStart = ( dataIter - data.begin() ) + entrySize;
	  }
  }

  return indices;
}

std::vector<float> Mddf::getObjectsHeights() const
{
  const int entrySize (36);
  std::vector<float> heights (0);

  std::vector<char>::const_iterator dataIter;
  int currentStart (12);
  
  for (dataIter = data.begin() ; dataIter != data.end() ; ++dataIter)
  {
    if ( ( dataIter - data.begin() + 24 ) % entrySize == 0 )
	  {
      heights.push_back( Utilities::get<float> ( data, currentStart ) );
	    currentStart = ( dataIter - data.begin() ) + entrySize;
	  }
  }

  return heights;
}

void Mddf::updateIndicesForLk(std::vector<int> alphaIndices) // TODO : that thing looks awful. Needs to be totally different (at least it does what it's supposed to do though).
{
  std::vector<int> mddfAlphaIndices( getEntriesIndices() );

  // TODO : add check to be sure alphaIndices.size() == data.size()

  for (int i = 0 ; i < mddfAlphaIndices.size() ; ++i)
  {
    for (int j = 0 ; j < alphaIndices.size() ; ++j)
    {
      if ( mddfAlphaIndices[i] == alphaIndices[j] )
	    {
        mddfAlphaIndices[i] = j;
	    }
    }
  }

  const int entrySize (36);
  std::vector<char> newMddfData (0);

  std::vector<char>::const_iterator dataIter;
  int newIndex (0);
  
  for (dataIter = data.begin() ; dataIter != data.end() ; ++dataIter)
  {
    if ( ( dataIter - data.begin() ) % entrySize == 0 )
	  {
      std::vector<char> currentIndex ( Utilities::getCharVectorFromInt( mddfAlphaIndices[newIndex] ) );
      newMddfData.insert( newMddfData.end(), currentIndex.begin(), currentIndex.end() );
      ++newIndex;
	    dataIter += 3;
	  }
    else
    {
      newMddfData.push_back( *dataIter );
    }
  }  
  
  data = newMddfData;
}

std::vector<int> Mddf::getM2IndicesForMmdx() const
{
  std::vector<int> mddfIndices( getEntriesIndices() );
  std::set<int> s;
  
  int size ( mddfIndices.size() );
  int i;

  for( i = 0 ; i < size ; ++i ) 
  {
    s.insert( mddfIndices[i] );
  }

  std::vector<int> indicesForMmdx( s.begin(), s.end() );

  return indicesForMmdx;
}

std::ostream & operator<<(std::ostream & os, const Mddf & mddf)
{
  os << "Chunk letters : " << mddf.letters << std::endl;
  os << "Chunk givenSize : " << mddf.givenSize << std::endl;
  os << "Real chunk Size : " << mddf.getRealSize() << std::endl;
  os << "Indices : " << std::endl;

  std::vector<int> indices ( mddf.getEntriesIndices() );

  std::vector<int>::const_iterator mddfIndicesIter;

  for (mddfIndicesIter = indices.begin() ; mddfIndicesIter != indices.end() ; ++mddfIndicesIter)
  {
    os << *mddfIndicesIter << " ";
  }

  os << std::endl;

  os << "Heights : " << std::endl;

  std::vector<float> heights ( mddf.getObjectsHeights() );

  std::vector<float>::const_iterator mddfHeightsIter;

  for (mddfHeightsIter = heights.begin() ; mddfHeightsIter != heights.end() ; ++mddfHeightsIter)
  {
    os << *mddfHeightsIter << " ";
  }

  os << std::endl;

  os << "------------------------------" << std::endl;

  return os;
}
