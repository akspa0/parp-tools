#ifndef MODEL_OBJECT_H
#define MODEL_OBJECT_H

#include <string>
#include <vector>

#include <3D_types.h>



namespace WMO14
{
  enum MOGP_flags
  {
    HAS_COLORS = 0x4,           // MOCV - colors
    HAS_DOODADS = 0x800,        // MODR - doodads
    HAS_WATER = 0x1000,		      // MLIQ - liquids
    ISNT_OCEAN = 0x80000        // is not ocean - liquid related stuff
  };

  struct MOGP
  {
    quint32 groupName;
    quint32 groupDescription;
    quint32 flags;
    float boundingBox[3*2];
    quint16 unknown1;
    quint16 unknown2;
    quint16 portals_index;
    quint16 portals_used;
    quint8 fog[4];
    quint32 liquidType;
    quint32 batchinfo[16];
    quint32 wmoGroupID;
  };
}


// a ModelObject is a part, a mesh, of a model
// stores the interesting 3D data
class ModelObject
{

public:

  // object name, public because because
  std::string _name;

  ModelObject() : _name("") { }


  // add stuff
  void addVertex(const Vertex &v);
  void addFace(const Face &f);
  void addUVcoords(const UVcoords &uv);

  void setFaceMaterial(const unsigned short &f, const unsigned short &mat);


  // build normals by hand
  void buildFaceNormals();
  void buildVertexNormals();

private:

  // object bounding box
  BoundingBox bbox_;

  // the data !
  std::vector<Vertex> vertices_;
  std::vector<Face> faces_;
  std::vector<UVcoords> mapping_coords_;
  std::vector<BspNode> bsp_nodes_;
  std::vector<BspRef> bsp_refs_;

  // stores data that can be handled specifically for some formats only
  // WMO/v14
  WMO14::MOGP _WMO14_mogp;
  std::vector<quint16> _WMO14_molr;
  std::vector<quint16> _WMO14_modr;
  std::vector<quint32> _WMO14_mocv;
  std::vector<char> _WMO14_mliq;


  // calculate mesh bounding box
  void computeBoundingBox();


  friend class Model;
  friend class WMO_exporter;
  friend class BSPTreeGenerator;

};

#include <model_object-inl.h>


#endif // MODEL_OBJECT_H
