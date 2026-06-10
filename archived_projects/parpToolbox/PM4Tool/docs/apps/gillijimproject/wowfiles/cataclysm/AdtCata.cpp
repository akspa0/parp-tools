#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>
#include <wowfiles/cataclysm/AdtCata.h>
#include <wowfiles/Chunk.h>
#include <wowfiles/Mddf.h>
#include <wowfiles/Modf.h>
#include <wowfiles/Mhdr.h>
#include <wowfiles/Mh2o.h>
#include <wowfiles/Mcnk.h>
#include <wowfiles/cataclysm/McnkCataTerrain.h>
#include <wowfiles/cataclysm/McnrCata.h>
#include <utilities/Utilities.h>
#include <wowfiles/lichking/AdtLk.h>

AdtCata::AdtCata(const std::string & adtFileName, const std::vector<char> & adtFile) : adtName(adtFileName)
{
  terrain = AdtCataTerrain(adtName, adtFile);
  
  texName = adtFileName;
  texName = texName.substr(0, texName.size() - 4);
  texName.append("_tex0.adt");

  std::vector<char> texFile(0);
  Utilities::getWholeFile(texName, texFile);

  textures = AdtCataTextures(texName, texFile);

  objName = adtFileName;
  objName = objName.substr(0, objName.size() - 4);
  objName.append("_obj0.adt");
  
  std::vector<char> objFile(0);
  Utilities::getWholeFile(objName, objFile);

  objects = AdtCataObjects(objName, objFile);
}

void AdtCata::changeHeight(const int & heightToAdd)
{
  objects.addToObjectsHeight(heightToAdd);
  terrain.addToTerrainHeight(heightToAdd);
}

void AdtCata::toFile()
{
  terrain.toFile();
  textures.toFile();
  objects.toFile();
}

void AdtCata::toFile(const std::string & fileName)
{
  terrain.toFile(fileName);
  
  std::string texturesName = fileName;
  texturesName = texturesName.substr(0, texturesName.size() - 4);
  texturesName.append("_tex0.adt");  
  textures.toFile(texturesName);
  
  std::string objectsName = fileName;
  objectsName = objectsName.substr(0, objectsName.size() - 4);
  objectsName.append("_obj0.adt");  
  objects.toFile(objectsName);
}

AdtLk AdtCata::toAdtLk() // TODO : conversion is only partial, only terrain
{
  std::vector<char> emptyData(0);

  std::vector<McnkLk> cMcnks;
  int currentMcnk;

  for (currentMcnk = 0 ; currentMcnk < 256 ; ++currentMcnk)
  {
    cMcnks.push_back(terrain.terrainMcnks[currentMcnk].toMcnkLk()); // TODO : only terrain, not finished...
  }

  AdtLk adtLk(adtName
            , terrain.terrainMver
            , terrain.mhdr.getFlags()
            , terrain.mh2o
            , textures.mtex
            , objects.mmdx
            , objects.mmid
            , objects.mwmo
            , objects.mwid
            , objects.mddf
            , objects.modf
            , cMcnks
            , terrain.mfbo
            , textures.mtxf
            );
  return adtLk;  
}

std::ostream & operator<<(std::ostream & os, const AdtCata & adtCata)
{
  os << adtCata.terrain << std::endl;
  os << "------------------------------" << std::endl;
  os << adtCata.textures << std::endl;
  os << "------------------------------" << std::endl;
  os << adtCata.objects << std::endl;

  return os;
}