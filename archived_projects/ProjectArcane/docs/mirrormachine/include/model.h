#ifndef MODEL_H
#define MODEL_H

#include <3D_types.h>
#include <model_object.h>
#include <material.h>


// Alpha WMO features
namespace WMO14
{
  struct MOHD
  {
    quint32 nMaterials;
    quint32 nGroups;
    quint32 nPortals;
    quint32 nLights;
    quint32 nDoodad_names;
    quint32 nDoodad_definitions;
    quint32 nDoodad_sets;
    quint32 ambient_color;
    quint32 wmo_ID;
    float bounding_box[6];
    quint32 liquid_related;
  };

  struct PortalVertices
  {
    Vertex vertices[4];
  };

  struct PortalInfo
  {
    quint16 base_index;
    quint16 nVertices;
    float vector[3];
    float unknown;
  };

  struct PortalRelation
  {
    quint16 portal_index;
    quint16 group_index;
    quint16 side;
    quint16 filler;
  };

  struct DoodadSet
  {
    char name[20];
    quint32 first_instance;
    quint32 nDoodads;
    quint32 unk;
  };

  struct MODN
  {
    std::vector<char> model_paths;
  };

  struct DoodadDefinition
  {
    quint32 name_offset;
    float position[3];
    float rotation[4];
    float scale;
    quint32 color;
  };
}

namespace OBJ
{
struct OBJface;
}



// Model represents every kind of 3d model we imported
// and contains methods to compute missing informations
// currently can be built from 3ds files only
class Model
{

public:

  // keep a track of the original file format
  int _import_type;
  enum import_type
  {
    FROM_3DS = 0,
    FROM_OBJ = 1,
    FROM_WMO14 = 2
  };

  // import flags are used to see how the convert process ended in detail
  unsigned int _import_flag;
  enum import_flags
  {
    FATAL_ERROR = 0x1,
    CANT_OPEN_FILE = 0x2,
    MISSING_UV = 0x4,
    OUTRANGE_UV = 0x8,
    BAD_OBJ_ORDER = 0x10
  };

  // cst/dst
  Model();
  Model(const char *path, const int &file_type);
  ~Model();


  // get a vector<string> of the group name (pretty useless)
  std::vector<std::string> getGroupNames() const;

  // get textures informations
  // must be careful at the padding of the texture list
  // these functions are used by WMO exporter only
  std::vector<std::string> getTexturesPaths() const;
  std::vector<unsigned int> getTexturePathOffsets(
      const unsigned int &tex) const;

  // get material index in the _materials using its _name
  unsigned char getMaterialIndex(const std::string &name) const;


  // asks by-ModelObject normals building
  void buildFaceNormals();
  void buildVertexNormals();


  // little hacks
  void setTexturesExtToBLP();
  void setTexturesCustomPath();

private:

  // object/material list
  std::vector<ModelObject> objects_;
  std::vector<Material> materials_;

  // global bounding box
  BoundingBox bbox_;

  // model ambient color
  std::vector<char> _ambient_color;

  // stores data that can be handled specifically for some formats only
  // DOES ANYONE HAVE AN IDEA ABOUT A BETTER DESIGN ??
  WMO14::MOHD _WMO14_mohd;
  std::vector<WMO14::PortalVertices> _WMO14_mopv;
  std::vector<WMO14::PortalInfo> _WMO14_mopt;
  std::vector<WMO14::PortalRelation> _WMO14_mopr;
  std::vector<WMO14::DoodadSet> _WMO14_mods;
  WMO14::MODN _WMO14_modn;
  std::vector<WMO14::DoodadDefinition> _WMO14_modd;


  // our loaders : take a file and import the stuff
  // should I take them out of here ?
  unsigned int load3ds(const char *file_path);
  unsigned int loadOBJ(const char *file_path);
  void loadOBJ_ComputeObject(const std::vector<OBJ::OBJface> &obj_faces,
                             const std::vector<Vertex> &vertices,
                             const std::vector<Normal> &normals,
                             const std::vector<UVcoords> &uv_coords,
                             int id_obj, size_t /*num_vertices*/);
  void loadOBJ_materials(const std::string &mtl_path);
  unsigned int loadWMOalpha(const char *file_path);

  // computing bounding boxes
  void computeBoundingBox();

  // sorts polygons according to their material ID
  static bool sortTwoFaces(const Face &a, const Face &b);
  void sortPolygonsByMat();


  friend class WMO_exporter;

};



#endif
