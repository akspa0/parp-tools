#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>
#include <wowfiles/Modf.h>
#include <utilities/Utilities.h>

Modf::Modf() : Chunk()
{
}

Modf::Modf(std::ifstream & adtFile, int offsetInFile) : Chunk(adtFile, offsetInFile)
{
}

Modf::Modf(const std::vector<char> & wholeFile, int offsetInFile) : Chunk(wholeFile, offsetInFile)
{
}

Modf::Modf(std::string letters, int givenSize, const std::vector<char> & data) : Chunk("FDOM", givenSize, data)
{
}

void Modf::addToObjectsHeight(const int & heightToAdd)
{
  if (givenSize <= 0) return;

  std::vector<float> modfHeights( getObjectsHeights() );

  for (int i = 0 ; i < modfHeights.size() ; ++i)
  {
    modfHeights[i] += (float)heightToAdd;
  }

  const int entrySize (64);
  std::vector<char> newModfData (0);

  std::vector<char>::const_iterator dataIter;
  int newIndex (0);
  
  for (dataIter = data.begin() ; dataIter != data.begin() + 12 ; ++dataIter)
  {
    newModfData.push_back( *dataIter );
  }

  std::vector<char> currentIndex ( Utilities::getCharVectorFromFloat( modfHeights[newIndex] ) );
  newModfData.insert( newModfData.end(), currentIndex.begin(), currentIndex.end() );
  ++newIndex;

  for (dataIter = data.begin() + 16 ; dataIter != data.end() ; ++dataIter)
  {
    if ( ( dataIter - data.begin() + 52 ) % entrySize == 0 )
	  {
      std::vector<char> currentIndex ( Utilities::getCharVectorFromFloat( modfHeights[newIndex] ) );
      newModfData.insert( newModfData.end(), currentIndex.begin(), currentIndex.end() );
      ++newIndex;
	    dataIter += 3;
	  }
    else
    {
      newModfData.push_back( *dataIter );
    }
  }  
  
  data = newModfData;
}

std::vector<Utilities::Point> Modf::getAllWmoCoords() const
{
  const int entrySize (64);
  std::vector<Utilities::Point> coords (0);

  std::vector<char>::const_iterator dataIter;
  int currentStart (8);
  
  for (dataIter = data.begin() ; dataIter != data.end() ; ++dataIter)
  {
    if ( ( dataIter - data.begin() + 56 ) % entrySize == 0 )
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

std::vector<int> Modf::getEntriesIndices() const
{
  const int entrySize (64);
  std::vector<int> indices (0);

  std::vector<char>::const_iterator dataIter;
  int currentStart (0);
  
  for (dataIter = data.begin() ; dataIter != data.end() ; ++dataIter)
  {
    if ( ( dataIter - data.begin() ) % entrySize == 0 )
	  {
      indices.push_back( Utilities::get<int> ( data, currentStart ));
	    currentStart = ( dataIter - data.begin() ) + entrySize;
	  }
  }

  return indices;
}

std::vector<float> Modf::getObjectsHeights() const
{
  const int entrySize (64);
  std::vector<float> heights (0);

  std::vector<char>::const_iterator dataIter;
  int currentStart (12);
  
  for (dataIter = data.begin() ; dataIter != data.end() ; ++dataIter)
  {
    if ( ( dataIter - data.begin() + 52 ) % entrySize == 0 )
	  {
      heights.push_back( Utilities::get<float> ( data, currentStart ) );
	    currentStart = ( dataIter - data.begin() ) + entrySize;
	  }
  }

  return heights;
}

std::vector<int> Modf::getWmoIndicesForMwmo() const
{
  std::vector<int> mddfIndices( getEntriesIndices() );
  std::set<int> s;
  
  int size ( mddfIndices.size() );
  int i;

  for( i = 0 ; i < size ; ++i ) 
  {
    s.insert( mddfIndices[i] );
  }

  std::vector<int> indicesForMmdx ( s.begin(), s.end() );

  return indicesForMmdx;
}

void Modf::updateIndicesForLk(std::vector<int> alphaIndices) // TODO : that thing looks awful. Needs to be totally different (at least it does what it's supposed to do though).
{
  std::vector<int> modfAlphaIndices( getEntriesIndices() );

  // TODO : add check to be sure alphaIndices.size() == data.size()

  for (int i = 0 ; i < modfAlphaIndices.size() ; ++i)
  {
    for (int j = 0 ; j < alphaIndices.size() ; ++j)
    {
      if ( modfAlphaIndices[i] == alphaIndices[j] )
	    {
        modfAlphaIndices[i] = j;
	    }
    }
  }

  const int entrySize (64);
  std::vector<char> newModfData (0);

  std::vector<char>::const_iterator dataIter;
  int newIndex (0);
  
  for (dataIter = data.begin() ; dataIter != data.end() ; ++dataIter)
  {
    if ( ( dataIter - data.begin() ) % entrySize == 0 )
	  {
      std::vector<char> currentIndex ( Utilities::getCharVectorFromInt( modfAlphaIndices[newIndex] ) );
      newModfData.insert( newModfData.end(), currentIndex.begin(), currentIndex.end() );
      ++newIndex;
	    dataIter += 3;
	  }
    else
    {
      newModfData.push_back( *dataIter );
    }
  }  
  
  data = newModfData;
}

std::ostream & operator<<(std::ostream & os, const Modf & modf)
{
  os << "Chunk letters : " << modf.letters << std::endl;
  os << "Chunk givenSize : " << modf.givenSize << std::endl;
  os << "Real chunk Size : " << modf.getRealSize() << std::endl;
  os << "Indices : " << std::endl;

  std::vector<int> indices ( modf.getEntriesIndices() );

  std::vector<int>::const_iterator modfIndicesIter;

  for (modfIndicesIter = indices.begin() ; modfIndicesIter != indices.end() ; ++modfIndicesIter)
  {
    os << *modfIndicesIter << " ";
  }

  os << std::endl;

  os << "Heights : " << std::endl;

  std::vector<float> heights ( modf.getObjectsHeights() );

  std::vector<float>::const_iterator modfHeightsIter;

  for (modfHeightsIter = heights.begin() ; modfHeightsIter != heights.end() ; ++modfHeightsIter)
  {
    os << *modfHeightsIter << " ";
  }

  os << std::endl;

  os << "------------------------------" << std::endl;
  return os;
}
