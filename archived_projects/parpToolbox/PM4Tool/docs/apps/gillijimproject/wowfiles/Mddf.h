#ifndef _WOWFILES_MDDF_H_
#define _WOWFILES_MDDF_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <wowfiles/Chunk.h>
#include <utilities/Utilities.h>

class Mddf : public Chunk
{
  public:

    Mddf();
    Mddf(std::ifstream & adtFile, int offsetInFile);
	  Mddf(const std::vector<char> & wholeFile, int offsetInFile);
    Mddf(std::string letters, int givenSize, const std::vector<char> & chunkData);
	
    void addToObjectsHeight(const int & heightToAdd);

    std::vector<Utilities::Point> getAllM2Coords() const;

    std::vector<int> getM2IndicesForMmdx() const;
    void updateIndicesForLk(std::vector<int> alphaIndices);
    std::vector<int> getEntriesIndices() const;

    friend std::ostream & operator<<(std::ostream & os, const Mddf & mddf);	

  private: 

    std::vector<float> getObjectsHeights() const;

};

#endif
