#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>

#include <QSettings>

#include <binary_io.h>
#include <model_object.h>
#include <bsptreegenerator.h>
#include <material.h>
#include <model.h>


Model::Model()
{

}

Model::Model(const char *path, const int &file_type)
    : _import_type(file_type)
    , _import_flag(0)
    , objects_(0)
    , materials_(0)
    , _ambient_color(4) // RGBA
{
  QSettings opt("egamh9", "mirrormachine");

  // chooses what kind of import we do
  switch (file_type)
  {
    case 0: // 3DS
      _import_flag = load3ds(path);
      if (!(_import_flag & FATAL_ERROR))
      {
        computeBoundingBox();
        sortPolygonsByMat();
        for (unsigned int i = 0; i < objects_.size(); ++i)
          objects_.at(i).computeBoundingBox();
        // as it can take much time, you can disable it for quick testing
        if (opt.value("create_normals", true).toBool())
        {
          buildFaceNormals();
          buildVertexNormals();
        }
        if (opt.value("generate_bsp", true).toBool())
        {
          for (size_t i = 0; i < objects_.size(); ++i)
          {
            BSPTreeGenerator generator(&objects_.at(i));
            generator.Process();
          }
        }
        if (opt.value("set_blp_extension", true).toBool())
        {
          setTexturesExtToBLP();
        }
        if (!opt.value("custom_path", "").toString().isEmpty())
        {
          setTexturesCustomPath();
        }
      }
      break;
    case 1: // OBJ
      _import_flag = loadOBJ(path);
      if (!(_import_flag & FATAL_ERROR))
      {
        computeBoundingBox();
        sortPolygonsByMat();
        for (unsigned int i = 0; i < objects_.size(); ++i)
          objects_.at(i).computeBoundingBox();
        if (opt.value("create_normals", true).toBool())
        {
          buildFaceNormals();
          buildVertexNormals();
        }
        if (opt.value("generate_bsp", true).toBool())
        {
          for (size_t i = 0; i < objects_.size(); ++i)
          {
            BSPTreeGenerator generator(&objects_.at(i));
            generator.Process();
          }
        }
        if (opt.value("set_blp_extension", true).toBool())
        {
          setTexturesExtToBLP();
        }
        if (!opt.value("custom_path", "").toString().isEmpty())
        {
          setTexturesCustomPath();
        }
      }
      break;
    case 2:
      _import_flag = loadWMOalpha(path);
      if (!(_import_flag & FATAL_ERROR))
      {
        computeBoundingBox();
        for (unsigned int i = 0; i < objects_.size(); ++i)
          objects_.at(i).computeBoundingBox();
      }
      break;
    default:
      break;
  }
}

Model::~Model()
{

}


// get data

void Model::computeBoundingBox()
{
  unsigned int nVertices = 0;
  Vertex v;

  const unsigned int nObjects = objects_.size();

  // can't use (0,0,0) (0,0,0) as a base
  // unless there are no vertices (or no objects)
  if (nObjects > 0 && objects_.at(0).vertices_.size() > 0)
  {
    v = objects_.at(0).vertices_.at(0);
    bbox_.a.x = v.x;
    bbox_.a.y = v.y;
    bbox_.a.z = v.z;
    bbox_.b.x = v.x;
    bbox_.b.y = v.y;
    bbox_.b.z = v.z;
  }

  // actually searches for a good box
  for (unsigned int i = 0; i < nObjects; ++i)
  {
    nVertices = objects_.at(i).vertices_.size();
    for (unsigned int j = 0; j < nVertices; ++j)
    {
      v = objects_.at(i).vertices_.at(j);
      if (v.x < bbox_.a.x) bbox_.a.x = v.x;
      if (v.y < bbox_.a.y) bbox_.a.y = v.y;
      if (v.z < bbox_.a.z) bbox_.a.z = v.z;
      if (v.x > bbox_.b.x) bbox_.b.x = v.x;
      if (v.y > bbox_.b.y) bbox_.b.y = v.y;
      if (v.z > bbox_.b.z) bbox_.b.z = v.z;
    }
  }
}


bool Model::sortTwoFaces(const Face &a, const Face &b)
{
  return a.mat < b.mat;
}
void Model::sortPolygonsByMat()
{
  unsigned int nObjects = objects_.size();

  // resort the polygons in face list for each object
  for (unsigned int i = 0; i < nObjects; ++i)
  {
    std::sort(objects_.at(i).faces_.begin(),
              objects_.at(i).faces_.end(),
              sortTwoFaces);
  }
}





std::vector<std::string> Model::getGroupNames() const
{
  std::vector<std::string> group_names;
  const unsigned int nObjects = objects_.size();
  for (unsigned int i = 0; i < nObjects; ++i)
    group_names.push_back(objects_.at(i)._name);
  return group_names;
}


std::vector<std::string> Model::getTexturesPaths() const
{
  std::vector<std::string> paths;
  const unsigned int nMaterials = materials_.size();
  for (unsigned int i = 0; i < nMaterials; ++i)
    paths.push_back(materials_.at(i)._path);
  return paths;
}


// delivers the texture path offset in MOTX chunk at result.at(0)
// result.at(1) is the offset to the 4-byte padding, used in MOMT.texture2
std::vector<unsigned int> Model::getTexturePathOffsets(
    const unsigned int &tex) const
{
  std::vector<unsigned int> lengths(2);
  unsigned int length = 0;

  for (unsigned int i = 0; i <= tex; ++i)
  {
    if (i == tex)
      lengths.at(0) = length;

    length += materials_.at(i)._path.length();
    ++length; // '\0'
    while (length % 4 != 0) // path 4-byte alignment
      ++length;

    if (i == tex)
      lengths.at(1) = length;

    ++length; // forced new 4byte alignment
    while (length % 4 != 0)
      ++length;
  }
  return lengths;
}


unsigned char Model::getMaterialIndex(const std::string &name) const
{
  unsigned char i = 0, nMaterials = materials_.size();
  for (i = 0; i < nMaterials; ++i)
  {
    if (materials_.at(i)._name.compare(name) == 0)
      break;
  }
  if (i == nMaterials)
    return 0xFF;
  else
    return i;
}





void Model::buildFaceNormals()
{
  const unsigned int nObjects = objects_.size();
  for (unsigned int i = 0; i < nObjects; ++i)
    objects_.at(i).buildFaceNormals();
}
void Model::buildVertexNormals()
{
  // must be called once faces have their normals.
  // this algorithm can be very low for a high nb of vertices !
  const unsigned int nObjects = objects_.size();
  for (unsigned int i = 0; i < nObjects; ++i)
    objects_.at(i).buildVertexNormals();
}





// delete texture path extension if there is one, and set ".blp"
void Model::setTexturesExtToBLP()
{
  std::string str;
  size_t index;

  const unsigned int nMaterials = materials_.size();
  for (unsigned int i = 0; i < nMaterials; ++i)
  {
    str = materials_.at(i)._path;
    index = str.find_last_of('.');
    if (index != std::string::npos)
      str = str.substr(0, index);
    str.append(".BLP");
    materials_.at(i)._path = str;
  }
}


// add the user custom path to every texture
void Model::setTexturesCustomPath()
{
  QSettings opt("egamh9", "mirrormachine");
  std::string custom_path =
      opt.value("custom_path", "").toString().toStdString();

  const unsigned int nMaterial = materials_.size();
  for (unsigned int i = 0; i < nMaterial; ++i)
    materials_.at(i)._path = custom_path + materials_.at(i)._path;
}





// ------------------------------------------------------------
// IMPORTING FROM A 3DS FILE
// ------------------------------------------------------------

unsigned int Model::load3ds(const char *file_path)
{
  using std::endl;

  unsigned int return_code = 0;

  QSettings opt("egamh9", "mirrormachine");
  bool verbose_log = opt.value("detail_log", false).toBool();
  bool mirror_v_mapping = opt.value("mirror_v_mapping", true).toBool();

  std::ofstream log_file("3ds_import.log");

  std::ifstream ifile(file_path, std::ios::binary);
  if (ifile.good())
  {
    log_file << file_path << " loaded." << endl;
  }
  else
  {
    log_file << "File can't be loaded :(" << endl;
    return (FATAL_ERROR | CANT_OPEN_FILE);
  }

  ifile.seekg(0, std::ifstream::end);
  unsigned long file_len = ifile.tellg();
  ifile.seekg(0, std::ifstream::beg);
  log_file << "Length : " << file_len << "." << endl;

  char *file_data = new char[file_len];
  ifile.read(file_data, file_len);
  ifile.close();
  std::vector<char> data_v(file_len);
  data_v.assign(file_data, file_data + file_len);
  delete[] file_data;

  unsigned int offset = 0;
  // must increase them while parsed
  int id_obj = -1;
  int id_mat = -1;
  unsigned short chunk_id = 0;
  unsigned int chunk_len = 0;

  // some values to make some checks and stuff
  ModelObject obj;
  Material mat;

  // these bools will be toggled if a warning appears
  bool wrong_version = false;
  bool missing_uv = false;
  /*bool outranged_texture_mapping = false;*/

  // this is the chunk length without the info bytes
#define CHUNK_DATA_LEN chunk_len - (sizeof(chunk_id) + sizeof(chunk_len))

  // loop to read the whole file
  while (offset < file_len)
  {
    chunk_id = bin_io::ReadInt16(data_v, offset);
    offset += sizeof(chunk_id);
    chunk_len = bin_io::ReadInt32(data_v, offset);
    offset += sizeof(chunk_len);

    switch (chunk_id)
    {

      // main chunk
    case 0x4D4D:
      break;

      // version (informative)
    case 0x0002:
    {
      unsigned int version = bin_io::ReadInt32(data_v, offset);
      log_file << "Detected version : " << version << "." << endl;
      if (version != 3)
        wrong_version = true;
      offset += 4;
    }
      break;

      // 3d editor chunk
    case 0x3D3D:
      break;

      // master scale
    case 0x0100:
    {
      float master_scale = bin_io::ReadFloat(data_v, offset);
      log_file << "Master scale : " << master_scale << endl;
      if (master_scale != 1)
        log_file << "WARNING : master scale can't be applied to WMOs !"
                 << endl;
      offset += 4;
    }
      break;

      // ambient color (whole model)
      // may have different kind of color encoding ?
    case 0x2100:
    {
      Color24 color;
      unsigned short ident = bin_io::ReadInt16(data_v, offset);
      if (ident == 0x0011)
      {
        color.r = data_v.at(offset + 6);
        color.g = data_v.at(offset + 7);
        color.b = data_v.at(offset + 8);
        log_file << "Ambient RGB color of the model parsed." << endl;
        _ambient_color.at(0) = color.r;
        _ambient_color.at(1) = color.g;
        _ambient_color.at(2) = color.b;
      }
      offset += CHUNK_DATA_LEN;
    }
      break;

      // mesh_version
    case 0x3D3E:
    {
      unsigned int mesh_version = bin_io::ReadInt32(data_v, offset);
      log_file << "Mesh version : " << mesh_version << " (no fucks given)"
               << endl;
      offset += 4;
    }
      break;

      // ********** MATERIALS **********

      // material block
      // contains every information for 1 material
    case 0xAFFF:
      id_mat++;
      log_file << "--------------------" << endl
               << "\t\tMATERIAL #" << id_mat << endl;
      materials_.push_back(mat);
      break;

      // material name
    case 0xA000:
    {
      unsigned int str_offset = 0;
      char c = ' ';
      while (true)
      {
        c = data_v.at(offset + str_offset);
        if (c == '\0')
          break;
        materials_.at(id_mat)._name += c;
        ++str_offset;
      }
      offset += str_offset + 1;
      log_file << "Material name : " << materials_.at(id_mat)._name << "."
               << endl;
    }
      break;

      // material ambient color
    case 0xA010:
    {
      Color24 color;
      unsigned short ident = bin_io::ReadInt16(data_v, offset);
      if (ident == 0x0011)
      {
        color.r = data_v.at(offset + 6);
        color.g = data_v.at(offset + 7);
        color.b = data_v.at(offset + 8);
        log_file << "Material ambient RGB color parsed." << endl;
        materials_.at(id_mat).setAmbientColor(color);
      }
      offset += CHUNK_DATA_LEN;
    }
      break;

      // material diffuse color
    case 0xA020:
    {
      Color24 color;
      unsigned short ident = bin_io::ReadInt16(data_v, offset);
      if (ident == 0x0011)
      {
        color.r = data_v.at(offset + 6);
        color.g = data_v.at(offset + 7);
        color.b = data_v.at(offset + 8);
        log_file << "Material diffuse RGB color parsed." << endl;
        materials_.at(id_mat).setDiffuseColor(color);
      }
      offset += CHUNK_DATA_LEN;
    }
      break;

      // material specular color
    case 0xA030:
    {
      Color24 color;
      unsigned short ident = bin_io::ReadInt16(data_v, offset);
      if (ident == 0x0011)
      {
        color.r = data_v.at(offset + 6);
        color.g = data_v.at(offset + 7);
        color.b = data_v.at(offset + 8);
        log_file << "Material specular RGB color parsed." << endl;
        materials_.at(id_mat).setSpecularColor(color);
      }
      offset += CHUNK_DATA_LEN;
    }
      break;

      // material shading value
    case 0xA100:
    {
      unsigned short shading = bin_io::ReadInt16(data_v, offset);
      log_file << "Material shading value : "
               << shading << endl;
      materials_.at(id_mat).setShadingValue(shading);
      offset += 2;
    }
      break;

      // material two-sided flag
    case 0xA081:
      log_file << "This material has TWO_SIDED flag." << endl;
      materials_.at(id_mat).setTwoSided(true);
      break;

      // material texture map
      // contains : perc, map path, tiling, blur, etc
    case 0xA200:
      log_file << "Texture map found." << endl;
      break;

      // map path
    case 0xA300:
    {
      unsigned int str_offset = 0;
      char c = ' ';
      while (true)
      {
        c = data_v[offset + str_offset];
        if (c == '\0')
          break;
        materials_.at(id_mat)._path += c;
        ++str_offset;
      }
      log_file << "Texture path : " << materials_.at(id_mat)._path << endl;
      offset += str_offset + 1;
    }
      break;

      // map flag
    case 0xA351:
    {
      unsigned short flags = bin_io::ReadFloat(data_v, offset);
      log_file << "Texture flags : " << std::hex << flags << endl;
      materials_.at(id_mat).setFlags(flags);
      offset += 2;
    }
      break;

      // ********** OBJECTS **********

      // object block
      // data : cstring model_name (20 char max ?)
    case 0x4000:
    {
      id_obj++;
      log_file << "--------------------" << endl
               << "\t\tOBJECT #" << id_obj << endl;
      objects_.push_back(obj);
      unsigned int str_offset = 0;
      char c = ' ';
      while (true)
      {
        c = data_v[offset + str_offset];
        if (c == '\0')
          break;
        objects_.at(id_obj)._name += c;
        ++str_offset;
      }
      log_file << "Object little name : " << objects_.at(id_obj)._name << "."
               << endl;
      offset += str_offset + 1;
    }
      break;

      // triangular mesh
    case 0x4100:
      break;

      // vertices list
    case 0x4110:
    {
      unsigned int nVertices = bin_io::ReadInt16(data_v, offset);
      offset += 2;
      log_file << "\tVERTICES LIST (" << nVertices << " detected)." << endl;
      Vertex v;
      for (unsigned int i = 0; i < nVertices; ++i)
      {
        v.x = bin_io::ReadFloat(data_v, offset);
        v.y = bin_io::ReadFloat(data_v, offset + 0x4);
        v.z = bin_io::ReadFloat(data_v, offset + 0x8);
        v.n.x = 0.0; v.n.y = 0.0; v.n.z = 0.0;
        offset += 12;
        if (verbose_log)
          log_file << "Vertex #" << i << " : "
                   << v.x << "\t" << v.y << "\t" << v.z
                   << endl;
        objects_.at(id_obj).addVertex(v);
      }
    }
      break;

      // faces list
    case 0x4120:
    {
      unsigned int nFaces = bin_io::ReadInt16(data_v, offset); // nb of faces
      offset += 2;
      log_file << "\tFACES LIST (" << nFaces << " detected)." << endl;
      Face f;
      for (unsigned int i = 0; i < nFaces; ++i)
      {
        f.a = bin_io::ReadInt16(data_v, offset);
        f.b = bin_io::ReadInt16(data_v, offset + 0x2);
        f.c = bin_io::ReadInt16(data_v, offset + 0x4);
        f.flags_3DS = bin_io::ReadInt16(data_v, offset + 0x6);
        f.mat = 0;
        if (verbose_log)
          log_file << "Face #" << std::dec << i << " : (" << std::dec << f.a
                   << ", " << std::dec << f.b << ", " << std::dec << f.c
                   << ") (flags : 0x" << std::hex << f.flags_3DS << ")"
                   << endl;
        objects_.at(id_obj).addFace(f);
        offset += 8;
      }
    }
      break;

      // face material list
      // says for one material (by its name) the faces that use it
    case 0x4130:
    {
      // 1) gets the material name (ASCIIZ)
      std::string material_name = "";
      unsigned int str_offset = 0;
      char c = ' ';
      while (true)
      {
        c = data_v[offset + str_offset];
        if (c == '\0')
          break;
        material_name += c;
        ++str_offset;
      }
      offset += str_offset + 1;
      // 2) then uses it to find what is it's index (this is... well...)
      unsigned int material_index = getMaterialIndex(material_name);
      unsigned short nFaces = bin_io::ReadInt16(data_v, offset);
      offset += 2;
      log_file << "\tFACE MATERIAL LIST (" << std::dec << nFaces
               << " faces detected) for the "
               << material_name << " mat (#" << material_index << ")" << endl;
      // 3) now for each face listed, we set its mat index to our material
      unsigned short current_face = 0;
      for (unsigned short i = 0; i < nFaces; ++i)
      {
        current_face = bin_io::ReadInt16(data_v, offset);
        objects_.at(id_obj).setFaceMaterial(current_face, material_index);
        if (verbose_log)
          log_file << "Face #" << i << " " << current_face
                   << " is set for material "
                   << material_index << "." << endl;
        offset += 2;
      }
    }
      break;

      // mapping coordinates by vertex
    case 0x4140:
    {
      unsigned short nVertices = bin_io::ReadInt16(data_v, offset);
      offset += 2;
      log_file << "\tMAPPING COORDINATES ("
               << nVertices << " vertices detected)." << endl;
      /*float outrange_tolerance_min = -0.02, outrange_tolerance_max = 1.02;*/
      UVcoords uv;
      for (unsigned int i = 0; i < nVertices; ++i)
      {
        uv.u = bin_io::ReadFloat(data_v, offset);
        uv.v = bin_io::ReadFloat(data_v, offset + 4);
        if (verbose_log)
          log_file << "UV coordinates #" << i << " : ("
                   << uv.u << ", " << uv.v << ")" << endl;
        if (mirror_v_mapping)
          uv.v = 1 - uv.v;
        /*if (uv.u < outrange_tolerance_min ||
            uv.u > outrange_tolerance_max ||
            uv.v < outrange_tolerance_min ||
            uv.v > outrange_tolerance_max)
          outranged_texture_mapping = true;*/
        objects_.at(id_obj).addUVcoords(uv);
        offset += 8;
      }
    }
      break;

      // local coordinates
      // float[4][3]
    case 0x4160:
    {
      log_file << "Local coordinates found for the object #"
               << id_obj << " : " << endl;
      for (unsigned int i = 0; i < 4; ++i)
      {
        switch (i)
        {
        case 0: log_file << "\tX1 : "; break;
        case 1: log_file << "\tX2 : "; break;
        case 2: log_file << "\tX3 : "; break;
        case 3: log_file << "\tOrigin : "; break;
        }
        for (unsigned int j = 0; j < 3; ++j)
        {
          float tmp_float = bin_io::ReadFloat(data_v, offset);
          log_file << tmp_float;
          if (j == 2)
            log_file << endl;
          else
            log_file << ", ";
          offset += 4;
        }
      }
    }
      break;

      // ********** DEFAULT **********

      // default case is unknown chunk
    default:
      log_file << "Unknown chunk skipped (id = " << std::hex << chunk_id
               << ", size = " << std::dec << chunk_len << ") !" << endl;
      offset += CHUNK_DATA_LEN;
   }

  }
  log_file << id_obj + 1 << " objects have been found." << endl;

#undef CHUNK_DATA_LEN

  // awful shit happened
  if (offset != file_len)
  {
    log_file << "WARNING : offset is not where it should !" << endl;
    return_code |= FATAL_ERROR;
  }

  // check if an object have missing UV coords
  for (unsigned int i = 0; i < objects_.size(); ++i)
  {
    if (objects_.at(i).mapping_coords_.size() == 0)
    {
      log_file << "Object #" << i << " doesn't have mapping coordinates !"
               << endl;
      missing_uv = true;
    }
  }

  // if a material doesn't have its path,
  // assumes the material name is the texture path
  for (unsigned int i = 0; i < materials_.size(); ++i)
  {
    if (materials_.at(i)._path.empty())
    {
      materials_.at(i)._path = materials_.at(i)._name;
      log_file << "Material #" << i << " (" << materials_.at(i)._name
               << ") has an empty path : using its name as path." << endl;
    }
  }

  // ********** WARNINGS **********

  if (wrong_version)
    log_file << "WARNING : "
                "this soft has been thought for v3 3ds files only !"
             << endl;

  if (missing_uv)
  {
    log_file << "WARNING : "
                "an object doesn't have mapping coordinates, "
                "I hope you know what you're doing, boy !"
             << endl;
    return_code |= MISSING_UV;
  }

  /* are they ABSOLUTELY mandatory ? can't decide
  if (outranged_texture_mapping)
  {
    log_file << "WARNING : "
                "some mapping coordinates are completely out "
                "of the [0, 1] range, which is mandatory for WMO."
             << endl;
    return_code |= OUTRANGE_UV;
  }*/

  if (verbose_log)
    log_file << "return_code = " << std::hex << return_code << endl;

  log_file.close();
  return return_code; // everything went better than expected (y)
}

// ------------------------------------------------------------









// ------------------------------------------------------------
// IMPORTING FROM A OBJ FILE
// ------------------------------------------------------------
namespace OBJ
{

struct OBJface
{
  unsigned short a, b, c;
  unsigned short na, nb, nc;
  unsigned short uva, uvb, uvc;
  std::string used_mat;
};

}

unsigned int Model::loadOBJ(const char *file_path)
{
  using std::endl;

  QSettings opt("egamh9", "mirrormachine");
  bool verbose_log = opt.value("detail_log", false).toBool();

  std::ifstream obj_file(file_path);
  std::ofstream log_file("OBJ_import.log");

  if (obj_file.good())
  {
    log_file << file_path << " loaded." << endl;
  }
  else
  {
    log_file << "File can't be loaded :(" << endl;
    log_file.close();
    return (FATAL_ERROR | CANT_OPEN_FILE);
  }

  bool mirror_v_mapping = opt.value("mirror_v_mapping", true).toBool();

  // running vars
  unsigned int return_code = 0;
  int id_obj = -1;
  unsigned int num_vertices = 0;

  // storing vars
  Vertex tmp_vert;
  UVcoords tmp_uv;
  Normal tmp_norm;
  OBJ::OBJface tmp_objface;
  std::string tmp_mtl;
  ModelObject obj;
  std::vector<Vertex> vertices;
  std::vector<Normal> vertex_normals;
  std::vector<UVcoords> uv_coords;
  std::vector<OBJ::OBJface> obj_faces;

  // OBJ is read line after line
  std::string line, line_ident;
  std::istringstream line_ss;
  unsigned int n_line = 0;

  // while there is stuff awaiting to be read
  while (obj_file.good())
  {
    std::getline(obj_file, line);

    // if file has been completly parsed
    if (obj_file.eof())
      break;
    // if a reading failed
    if (obj_file.fail())
    {
      log_file << "Failed to read some data line " << n_line
               << ", probably wrong input file." << endl;
      log_file.close();
      return_code |= FATAL_ERROR;
      break;
    }

    ++n_line;

    if (line.empty())
      continue;

    line_ss.clear();
    line_ss.str(line);
    line_ss >> line_ident;

    // --------------------
    // MATERIAL LIBRARY
    if (line_ident.compare("mtllib") == 0)
    {
      // gets material library path
      std::string material_lib;
      line_ss >> material_lib;
      std::string material_lib_path = file_path;
      material_lib_path =
          material_lib_path.substr(0, material_lib_path.find_last_of('/'));
      material_lib_path.append("/");
      material_lib_path.append(material_lib);
      // loads it
      log_file << n_line << ": MATERIAL LIBRARY " << material_lib
               << " required (path : " << material_lib_path << ")" << endl;
      loadOBJ_materials(material_lib_path);
    }

    // --------------------
    // OBJECT
    else if (line_ident.compare("o") == 0)
    {
      ++id_obj;
      objects_.push_back(obj);
      std::string object_name;
      line_ss >> object_name;
      objects_.at(id_obj)._name = object_name;
      log_file << n_line << ": OBJECT #" << id_obj << " called "
               << object_name << endl;

      if (id_obj > 0)
      {
        loadOBJ_ComputeObject(obj_faces, vertices, vertex_normals,
                              uv_coords, id_obj - 1, num_vertices);
        obj_faces.clear();
      }

      num_vertices = 0;
    }

    // --------------------
    // VERTICES
    // As it's just an array of vertices randomly accessed, it can
    // be stored in the object without problems.
    else if (line_ident.compare("v") == 0)
    {
      if (id_obj == -1)
        return (FATAL_ERROR | BAD_OBJ_ORDER);
      line_ss >> tmp_vert.x >> tmp_vert.y >> tmp_vert.z;
      vertices.push_back(tmp_vert);
      ++num_vertices;
    }

    // --------------------
    // UV COORDS
    else if (line_ident.compare("vt") == 0)
    {
      line_ss >> tmp_uv.u >> tmp_uv.v;
      if (mirror_v_mapping)
        tmp_uv.v = 1 - tmp_uv.v;
      uv_coords.push_back(tmp_uv);
    }

    // --------------------
    // VERTEX NORMALS
    else if (line_ident.compare("vn") == 0)
    {
      line_ss >> tmp_norm.x >> tmp_norm.y >> tmp_norm.z;
      vertex_normals.push_back(tmp_norm);
    }

    // --------------------
    // USE THIS MATERIAL
    else if (line_ident.compare("usemtl") == 0)
    {
      line_ss >> tmp_mtl;
      log_file << n_line << ": the following faces use the material "
               << tmp_mtl << endl;
    }

    // --------------------
    // FACES
    else if (line_ident.compare("f") == 0)
    {
      line_ss >> tmp_objface.a;
      line_ss.ignore();
      line_ss >> tmp_objface.uva;
      line_ss.ignore();
      line_ss >> tmp_objface.na;

      line_ss >> tmp_objface.b;
      line_ss.ignore();
      line_ss >> tmp_objface.uvb;
      line_ss.ignore();
      line_ss >> tmp_objface.nb;

      line_ss >> tmp_objface.c;
      line_ss.ignore();
      line_ss >> tmp_objface.uvc;
      line_ss.ignore();
      line_ss >> tmp_objface.nc;

      tmp_objface.used_mat = tmp_mtl;
      obj_faces.push_back(tmp_objface);
    }

    // --------------------
    // default : unparsed stuff
    else
    {
      log_file << n_line << ": unknown data (" << line_ident
               << ", " << line << ")" << endl;
    }

  } // reading loop end
  obj_file.close();

  loadOBJ_ComputeObject(
    obj_faces, vertices, vertex_normals, uv_coords, id_obj, num_vertices);

  // if a material doesn't have its path,
  // assumes the material name is the texture path
  for (unsigned int i = 0; i < materials_.size(); ++i)
  {
    if (materials_.at(i)._path.empty())
    {
      materials_.at(i)._path = materials_.at(i)._name;
      log_file << "Material #" << i << " (" << materials_.at(i)._name
               << ") has an empty path : using its name as path." << endl;
    }
  }

  if (verbose_log)
    log_file << "return_code = " << return_code << endl;

  log_file.close();
  return return_code;
}





void Model::loadOBJ_ComputeObject(const std::vector<OBJ::OBJface>& obj_faces,
                                  const std::vector<Vertex>& vertices,
                                  const std::vector<Normal>& normals,
                                  const std::vector<UVcoords>& uv_coords,
                                  int id_obj,
                                  size_t /*num_vertices*/)
{
  // Gotta be careful using these vectors : OBJ format uses index starting
  // from 1 and not 0.
  size_t num_faces = obj_faces.size();

  objects_.at(id_obj).vertices_.resize(num_faces * 3);
  objects_.at(id_obj).mapping_coords_.resize(num_faces * 3);

//  std::cout << "num faces " << num_faces << std::endl;
  for (size_t i = 0; i < num_faces; ++i)
  {
    // Let's do something more pragmatic and create a new index
    // as Blizz did during alpha.

    Face f;
    f.a = i*3;
    f.b = i*3 + 1;
    f.c = i*3 + 2;

    objects_.at(id_obj).vertices_.at(i*3).copy(
      vertices.at(obj_faces.at(i).a - 1));
    objects_.at(id_obj).vertices_.at(i*3 + 1).copy(
      vertices.at(obj_faces.at(i).b - 1));
    objects_.at(id_obj).vertices_.at(i*3 + 2).copy(
      vertices.at(obj_faces.at(i).c - 1));

    objects_.at(id_obj).mapping_coords_.at(i*3).copy(
      uv_coords.at(obj_faces.at(i).uva - 1));
    objects_.at(id_obj).mapping_coords_.at(i*3 + 1).copy(
      uv_coords.at(obj_faces.at(i).uvb - 1));
    objects_.at(id_obj).mapping_coords_.at(i*3 + 2).copy(
      uv_coords.at(obj_faces.at(i).uvc - 1));

    objects_.at(id_obj).vertices_.at(i*3).n.copy(
      normals.at(obj_faces.at(i).na - 1));
    objects_.at(id_obj).vertices_.at(i*3 + 1).n.copy(
      normals.at(obj_faces.at(i).nb - 1));
    objects_.at(id_obj).vertices_.at(i*3 + 2).n.copy(
      normals.at(obj_faces.at(i).nc - 1));

    f.mat = getMaterialIndex(obj_faces.at(i).used_mat);

    objects_.at(id_obj).faces_.push_back(f);

    /*
     * Naive attempt : this won't work because the v/vt/vn buffers aren't
     * reset when a new object is queried according to the OBJ format spec.

//    std::cout << "face VI #" << i << std::endl;
    Face f;
    f.a = obj_faces.at(i).a - 1;
    f.b = obj_faces.at(i).b - 1;
    f.c = obj_faces.at(i).c - 1;

//    std::cout << "face normals #" << i << std::endl;
    objects_.at(id_obj).vertices_.at(f.a).n.copy(
      normals.at(obj_faces.at(i).na - 1));
    objects_.at(id_obj).vertices_.at(f.b).n.copy(
      normals.at(obj_faces.at(i).nb - 1));
    objects_.at(id_obj).vertices_.at(f.c).n.copy(
      normals.at(obj_faces.at(i).nc - 1));

//    std::cout << "face uv #" << i << std::endl;
    objects_.at(id_obj).mapping_coords_.at(f.a).copy(
      uv_coords.at(obj_faces.at(i).uva - 1));
    objects_.at(id_obj).mapping_coords_.at(f.b).copy(
      uv_coords.at(obj_faces.at(i).uvb - 1));
    objects_.at(id_obj).mapping_coords_.at(f.c).copy(
      uv_coords.at(obj_faces.at(i).uvc - 1));

//    std::cout << "mat to face #" << i << std::endl;
    f.mat = getMaterialIndex(obj_faces.at(i).used_mat);

//    std::cout << "face added to object, #" << i << std::endl;
    objects_.at(id_obj).faces_.push_back(f);

    */
  }
}





void Model::loadOBJ_materials(const std::string &mtl_path)
{
  using std::endl;

  QSettings opt("egamh9", "mirrormachine");
  bool verbose_log = opt.value("detail_log", false).toBool();

  std::ifstream mtl_file(mtl_path.c_str());
  std::ofstream log_file("OBJ_mtllib_import.log");

  // running vars
  int id_mat = -1;

  // storing vars
  Material material;

  // and our vars for line after line reading
  std::string line, line_ident;
  std::istringstream line_ss;
  unsigned int n_line = 0;

  while (mtl_file)
  {
    std::getline(mtl_file, line);

    // if file has been completly parsed
    if (mtl_file.eof())
    {
      break;
    }
    // if a reading failed
    else if (mtl_file.fail())
    {
      log_file << "Failed to read some data line " << n_line
               << ", probably wrong input file." << endl;
      log_file.close();
      break;
    }
    // else let's read this mtllib

    ++n_line;

    if (!line.empty())
    {
      line_ss.clear();
      line_ss.str(line);
      line_ss >> line_ident;

      // --------------------
      // NEW MATERIAL
      if (line_ident.compare("newmtl") == 0)
      {
        ++id_mat;
        materials_.push_back(material);
        line_ss >> materials_.at(id_mat)._name;
        log_file << n_line << ": MATERIAL #" << id_mat << " : "
                 << materials_.at(id_mat)._name << endl;
      }

      // --------------------
      // SHININESS
      else if (line_ident.compare("Ns") == 0)
      {
          float shininess;
          line_ss >> shininess;
          log_file << n_line << ": shininess (unused) = " << shininess << endl;
      }

      // --------------------
      // AMBIENT COLOR
      else if (line_ident.compare("Ka") == 0)
      {
        Color24 ambientColor;
        float component;
        line_ss >> component;
        ambientColor.r = static_cast<unsigned char>(component * 255);
        line_ss >> component;
        ambientColor.g = static_cast<unsigned char>(component * 255);
        line_ss >> component;
        ambientColor.b = static_cast<unsigned char>(component * 255);
        log_file << n_line << ": ambient color parsed" << endl;
        materials_.at(id_mat).setAmbientColor(ambientColor);
      }

      // --------------------
      // DIFFUSE COLOR
      else if (line_ident.compare("Kd") == 0)
      {
        Color24 diffuseColor;
        float component;
        line_ss >> component;
        diffuseColor.r = static_cast<unsigned char>(component * 255);
        line_ss >> component;
        diffuseColor.g = static_cast<unsigned char>(component * 255);
        line_ss >> component;
        diffuseColor.b = static_cast<unsigned char>(component * 255);
        log_file << n_line << ": diffuse color parsed" << endl;
        materials_.at(id_mat).setDiffuseColor(diffuseColor);
      }

      // --------------------
      // SPECULAR COLOR
      else if (line_ident.compare("Ks") == 0)
      {
        /*Color24 specularColor;
        float component;
        line_ss >> component;
        specularColor.r = static_cast<unsigned char>(component * 255);
        line_ss >> component;
        specularColor.g = static_cast<unsigned char>(component * 255);
        line_ss >> component;
        specularColor.b = static_cast<unsigned char>(component * 255);*/
        log_file << n_line << ": specular color (unused) parsed" << endl;
      }

      // --------------------
      // TEXTURE PATH
      else if (line_ident.compare("map_Kd") == 0)
      {
        line_ss.ignore();
        materials_.at(id_mat)._path = line.substr(line_ss.tellg());
        log_file << n_line << ": texture path : "
                 << materials_.at(id_mat)._path << endl;
      }

      // --------------------
      // default : unparsed stuff
      else
      {
        if (verbose_log)
          log_file << n_line << ": unknown data (" << line_ident
                   << ", " << line << ")" << endl;
      }
    }



  } // end of material lib parsing

  mtl_file.close();
  log_file.close();
}

// ------------------------------------------------------------









// ------------------------------------------------------------
// IMPORTING FROM AN ALPHA WOW WORLD MODEL OBJECT
// ------------------------------------------------------------

unsigned int Model::loadWMOalpha(const char *file_path)
{
  using std::endl;

  /*QSettings opt("egamh9", "mirrormachine");
  bool verbose_log = opt.value("detail_log", false).toBool();*/

  std::ifstream alpha_file(file_path);
  std::ofstream log_file("WMO14_import.log");

  if (!alpha_file.is_open() || !log_file.is_open())
    return (FATAL_ERROR | CANT_OPEN_FILE);

  log_file << "Importing file " << file_path << endl;

  // get file length
  alpha_file.seekg(0, std::ios::end);
  unsigned long file_length = alpha_file.tellg();
  alpha_file.seekg(0, std::ios::beg);

  // import the file in a vector<char>
  char *file_data = new char[file_length];
  alpha_file.read(file_data, file_length);
  alpha_file.close();
  std::vector<char> data(file_length);
  data.assign(file_data, file_data + file_length);
  delete[] file_data;

  // storing vars
  unsigned int return_code = 0;
  unsigned long textures_offset = 0;

  quint32 chunk_id;
  quint32 chunk_length;

  unsigned long offset = 0;
  unsigned long chunk_offset = 0;

  while (chunk_offset < file_length)
  {
    // stores the chunk ID and length
    // offset is moved to the beginning of the chunk data
    offset = chunk_offset;
    chunk_id = bin_io::ReadInt32(data, offset);
    // have to stop when reaches the first MOGP
    if (chunk_id == 0x4D4F4750)
      break;
    offset += 4;
    chunk_length = bin_io::ReadInt32(data, offset);
    offset += 4;

    switch (chunk_id)
    {
      // MVER - not that usefull... should be 14
      case 0x4D564552:
      {
        log_file << "MVER : file version = "
                 << bin_io::ReadInt32(data, offset) << endl;
      }
        break;

      // MOMO - have to trick this one
      case 0x4D4F4D4F:
        chunk_length = 0;
        break;

      // MOHD - header
      // lot of neat stuff
      case 0x4D4F4844:
      {
        _WMO14_mohd.nMaterials = bin_io::ReadInt32(data, offset);
        _WMO14_mohd.nGroups = bin_io::ReadInt32(data, offset + 0x4);
        _WMO14_mohd.nPortals = bin_io::ReadInt32(data, offset + 0x8);
        _WMO14_mohd.nLights = bin_io::ReadInt32(data, offset + 0xC);
        _WMO14_mohd.nDoodad_names = bin_io::ReadInt32(data, offset + 0x10);
        _WMO14_mohd.nDoodad_definitions = bin_io::ReadInt32(data, offset +0x14);
        _WMO14_mohd.nDoodad_sets = bin_io::ReadInt32(data, offset + 0x18);
        _WMO14_mohd.ambient_color = bin_io::ReadInt32(data, offset + 0x1C);
        _WMO14_mohd.wmo_ID = bin_io::ReadInt32(data, offset + 0x20);
        _WMO14_mohd.liquid_related = bin_io::ReadInt32(data, offset + 0x38);
        log_file << "MOHD parsed" << endl;
      }
        break;

      // MOTX - texture files paths
      case 0x4D4F5458:
      {
        textures_offset = offset;
        log_file << "MOTX parsed" << endl;
      }
        break;

      // MOMT - texture files paths
      case 0x4D4F4D54:
      {
        std::string texture_path;
        char c;
        unsigned int ptr;
        Color24 ambient, diffuse;
        unsigned int nMaterials = chunk_length / 44;
        for (unsigned int i = 0; i < nMaterials; ++i)
        {
          Material mat;
          // flags - see if backface culling is on
          mat.setTwoSided((bin_io::ReadInt32(data, offset) & 0x4) != 0);

          // get the path from MOTX
          texture_path.clear();
          c = ' ';
          ptr = bin_io::ReadInt32(data, offset + 0xC);
          while (c != '\0')
          {
            c = data.at(textures_offset + ptr);
            if (c == '\0')
              break;
            texture_path.push_back(c);
            ++ptr;
          }
          mat._path = texture_path;

          // colors
          ambient.b = data.at(offset + 0x10);
          ambient.g = data.at(offset + 0x11);
          ambient.r = data.at(offset + 0x12);
          diffuse.b = data.at(offset + 0x1C);
          diffuse.g = data.at(offset + 0x1D);
          diffuse.r = data.at(offset + 0x1E);
          mat.setAmbientColor(ambient);
          mat.setDiffuseColor(diffuse);

          materials_.push_back(mat);
          offset += 44;
        }
        log_file << "MOMT parsed (" << nMaterials << " materials found)"
                 << endl;
      }
        break;

      // MOPV - portal vertices
      case 0x4D4F5056:
      {
        Vertex v;
        WMO14::PortalVertices pv;
        unsigned int nEntries = chunk_length / 48;
        for (unsigned int i = 0; i < nEntries; ++i)
        {
          v.x = bin_io::ReadFloat(data, offset);
          v.y = bin_io::ReadFloat(data, offset + 0x4);
          v.z = bin_io::ReadFloat(data, offset + 0x8);
          pv.vertices[0] = v;
          v.x = bin_io::ReadFloat(data, offset + 0xC);
          v.y = bin_io::ReadFloat(data, offset + 0x10);
          v.z = bin_io::ReadFloat(data, offset + 0x14);
          pv.vertices[1] = v;
          v.x = bin_io::ReadFloat(data, offset + 0x18);
          v.y = bin_io::ReadFloat(data, offset + 0x1C);
          v.z = bin_io::ReadFloat(data, offset + 0x20);
          pv.vertices[2] = v;
          v.x = bin_io::ReadFloat(data, offset + 0x24);
          v.y = bin_io::ReadFloat(data, offset + 0x28);
          v.z = bin_io::ReadFloat(data, offset + 0x2C);
          pv.vertices[3] = v;
          _WMO14_mopv.push_back(pv);
          offset += 48;
        }
        log_file << "MOPR parsed (" << nEntries << " portals found)" << endl;
      }
        break;

      // MOPT - portal informations
      case 0x4D4F5054:
      {
        WMO14::PortalInfo info;
        unsigned int nEntries = chunk_length / 20;
        for (unsigned int i = 0; i < nEntries; ++i)
        {
          info.base_index = bin_io::ReadInt16(data, offset);
          info.nVertices = bin_io::ReadInt16(data, offset + 0x2);
          info.vector[0] = bin_io::ReadFloat(data, offset + 0x4);
          info.vector[1] = bin_io::ReadFloat(data, offset + 0x8);
          info.vector[2] = bin_io::ReadFloat(data, offset + 0xC);
          info.unknown = bin_io::ReadFloat(data, offset + 0x10);
          _WMO14_mopt.push_back(info);
          offset += 20;
        }
        log_file << "MOPT parsed (" << nEntries << " infos found)" << endl;
      }
        break;

      // MOPR - portal relationships
      case 0x4D4F5052:
      {
        WMO14::PortalRelation relation;
        unsigned int nEntries = chunk_length / 8;
        for (unsigned int i = 0; i < nEntries; ++i)
        {
          relation.portal_index = bin_io::ReadInt16(data, offset);
          relation.group_index = bin_io::ReadInt16(data, offset + 0x2);
          relation.side = bin_io::ReadInt16(data, offset + 0x4);
          relation.filler = bin_io::ReadInt16(data, offset + 0x6);
          _WMO14_mopr.push_back(relation);
          offset += 8;
        }
        log_file << "MOPR parsed (" << nEntries << " relations found)" << endl;
      }
        break;

      // MODS - doodad sets
      case 0x4D4F4453:
      {
        WMO14::DoodadSet set;
        unsigned int nSets = chunk_length / 32;
        for (unsigned int i = 0; i < nSets; ++i)
        {
          for (unsigned char c = 0; c < 20; ++c)
            set.name[c] = data.at(offset + c);
          set.first_instance = bin_io::ReadInt32(data, offset + 0x14);
          set.nDoodads = bin_io::ReadInt32(data, offset + 0x18);
          set.unk = bin_io::ReadInt32(data, offset + 0x1C);
          _WMO14_mods.push_back(set);
          offset += 32;
        }
        log_file << "MODS parsed (" << nSets << " sets found)" << endl;
      }
        break;

      // MODN - doodad paths
      case 0x4D4F444E:
      {
        for (unsigned int i = 0; i < chunk_length; ++i)
          _WMO14_modn.model_paths.push_back(data.at(offset + i));
        log_file << "MODN parsed" << endl;
      }
        break;

      // MODD - doodad definitions
      case 0x4D4F4444:
      {
        WMO14::DoodadDefinition doodad;
        unsigned int nDefs = chunk_length / 40;
        for (unsigned int i = 0; i < nDefs; ++i)
        {
          doodad.name_offset = bin_io::ReadInt32(data, offset);
          doodad.position[0] = bin_io::ReadFloat(data, offset + 0x4);
          doodad.position[1] = bin_io::ReadFloat(data, offset + 0x8);
          doodad.position[2] = bin_io::ReadFloat(data, offset + 0xC);
          doodad.rotation[0] = bin_io::ReadFloat(data, offset + 0x10);
          doodad.rotation[1] = bin_io::ReadFloat(data, offset + 0x14);
          doodad.rotation[2] = bin_io::ReadFloat(data, offset + 0x18);
          doodad.rotation[3] = bin_io::ReadFloat(data, offset + 0x1C);
          doodad.scale = bin_io::ReadFloat(data, offset + 0x20);
          doodad.color = bin_io::ReadInt32(data, offset + 0x24);
          _WMO14_modd.push_back(doodad);
          offset += 40;
        }
        log_file << "MODD parsed (" << nDefs << " definitions found)" << endl;
      }
        break;

      // unparsed
      default:
        log_file << "Unparsed chunk (ID " << std::hex << chunk_id
                 << ", length = " << std::dec << chunk_length
                 << ")" << endl;
    } // end of chunk ID switch

    // 8 + chunk_size = the whole chunk size
    chunk_offset += 8 + chunk_length;

  } // end of root parsing

  int id_obj = -1;
  ModelObject object;

  // now we have to parse every group
  // offset is at the first MOGP.data
  while (chunk_offset < file_length)
  {
    offset = chunk_offset;
    chunk_id = bin_io::ReadInt32(data, offset);
    offset += 4;
    chunk_length = bin_io::ReadInt32(data, offset);
    offset += 4;

    switch (chunk_id)
    {
      // MOGP
      // size encompass the whole file so gotta hack it a bit
      case 0x4D4F4750:
      {
        ++id_obj;
        objects_.push_back(object);
        log_file << "New object (#" << id_obj << ")" << endl;
        objects_.at(id_obj)._WMO14_mogp.flags =
            bin_io::ReadInt32(data, offset + 0x8);
        // 128 is the MOGP data size without its sub-chunks
        chunk_length = 128;
      }
        break;

      // MOPY - polygons infos
      // prepare the face list for the MOIN
      case 0x4D4F5059:
      {
        unsigned int nPolygons = chunk_length / 4;
        Face f;
        objects_.at(id_obj).faces_.assign(nPolygons, f);
        for (unsigned int i = 0; i < nPolygons; ++i)
        {
          // get the material ID
          objects_.at(id_obj).faces_.at(i).mat = data.at(offset + 2);
          offset += 4;
        }
        log_file << "MOPY parsed (" << nPolygons << " entries)" << endl;
      }
        break;

      // MOVT - vertices
      case 0x4D4F5654:
      {
        unsigned int nVertices = chunk_length / 12;
        Vertex vertex;
        for (unsigned int i = 0; i < nVertices; ++i)
        {
          vertex.x = bin_io::ReadFloat(data, offset);
          vertex.y = bin_io::ReadFloat(data, offset + 4);
          vertex.z = bin_io::ReadFloat(data, offset + 8);
          objects_.at(id_obj).addVertex(vertex);
          offset += 12;
        }
        log_file << "MOVT parsed (" << nVertices << " entries)" << endl;
      }
        break;

      // MONR - normals
      case 0x4D4F4E52:
      {
        unsigned int nNormals = chunk_length / 12;
        Normal normal;
        for (unsigned int i = 0; i < nNormals; ++i)
        {
          normal.x = bin_io::ReadFloat(data, offset);
          normal.y = bin_io::ReadFloat(data, offset + 4);
          normal.z = bin_io::ReadFloat(data, offset + 8);
          objects_.at(id_obj).vertices_.at(i).n = normal;
          offset += 12;
        }
        log_file << "MONR parsed (" << nNormals << " entries)" << endl;
      }
        break;

      // MOTV - mapping coordinates
      case 0x4D4F5456:
      {
        unsigned int nCoords = chunk_length / 8;
        UVcoords uv;
        for (unsigned int i = 0; i < nCoords; ++i)
        {
          uv.u = bin_io::ReadFloat(data, offset);
          uv.v = bin_io::ReadFloat(data, offset + 4);
          objects_.at(id_obj).addUVcoords(uv);
          offset += 8;
        }
        log_file << "MOTV parsed (" << nCoords << " entries)" << endl;
      }
        break;

      // MOIN - vertex indices
      case 0x4D4F494E:
      {
        unsigned int nFaces = objects_.at(id_obj).faces_.size();
        unsigned short ref = 0;
        for (unsigned int i = 0; i < nFaces; ++i)
        {
          objects_.at(id_obj).faces_.at(i).a = ref++;
          objects_.at(id_obj).faces_.at(i).b = ref++;
          objects_.at(id_obj).faces_.at(i).c = ref++;
        }

        /* // the dumb approach is to parse it as if it was a good index
           // but it seems it's just a more or less dumb going up index
           // try to compute one myself above...

        unsigned int nFaces = chunk_length / 6;
        for (unsigned int i = 0; i < nFaces; ++i)
        {
           _objects.at(id_obj).faces_.at(i).a =
               bin_io::getUShort(data, offset);
           _objects.at(id_obj).faces_.at(i).b =
               bin_io::getUShort(data, offset + 2);
           _objects.at(id_obj).faces_.at(i).c =
               bin_io::getUShort(data, offset + 4);
          offset += 6;
        }*/

        log_file << "MOIN parsed (" << nFaces << " entries)" << endl;
      }
        break;

      // MOLR - lights references
      case 0x4D4F4C52:
      {
        unsigned int nRefs = chunk_length / 2;
        for (unsigned int i = 0; i < nRefs; ++i)
        {
          objects_.at(id_obj)._WMO14_molr.push_back(
              bin_io::ReadInt16(data, offset));
          offset += 2;
        }
        log_file << "MOLR parsed" << endl;
      }
      break;

      // MODR - doodads references
      case 0x4D4F4452:
      {
        unsigned int nRefs = chunk_length / 2;
        for (unsigned int i = 0; i < nRefs; ++i)
        {
          objects_.at(id_obj)._WMO14_modr.push_back(
              bin_io::ReadInt16(data, offset));
          offset += 2;
        }
        log_file << "MODR parsed" << endl;
      }
        break;

      // MOCV - vertex shading
      case 0x4D4F4356:
      {
        unsigned int nColors = chunk_length / 4;
        for (unsigned int i = 0; i < nColors; ++i)
        {
          objects_.at(id_obj)._WMO14_mocv.push_back(
              bin_io::ReadInt32(data, offset));
          offset += 4;
        }
        log_file << "MOCV parsed" << endl;
      }
        break;

      // MLIQ - liquids
      case 0x4D4C4951:
      {
        for (unsigned int i = 0; i < chunk_length; ++i)
          objects_.at(id_obj)._WMO14_mliq.push_back(data.at(offset + i));
      }
        break;

      // unparsed chunks
      default:
        log_file << "Unparsed group chunk (ID " << std::hex << chunk_id
                 << ", length = " << std::dec << chunk_length
                 << ")" << endl;
    } // end of chunk ID switch

    // 8 + chunk_size = the whole chunk size
    chunk_offset += 8 + chunk_length;

  } // end of groups parsing

  log_file.close();
  return return_code;
}

// ------------------------------------------------------------



