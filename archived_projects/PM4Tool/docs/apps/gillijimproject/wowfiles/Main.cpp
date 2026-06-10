#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <wowfiles/Chunk.h>
#include <wowfiles/Main.h>

Main::Main() : Chunk()
{
}

Main::Main(std::ifstream & adtFile, int offsetInFile) : Chunk(adtFile, offsetInFile)
{
}

Main::Main(std::string letters, int givenSize, const std::vector<char> & data) : Chunk("NIAM", givenSize, data) 
{
}