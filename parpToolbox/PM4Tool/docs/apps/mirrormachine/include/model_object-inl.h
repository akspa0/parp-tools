#ifndef MODEL_OBJECTINL_H
#define MODEL_OBJECTINL_H

class ModelObject;

inline void ModelObject::addVertex(const Vertex &v)
{
  vertices_.push_back(v);
}
inline void ModelObject::addFace(const Face &f)
{
  faces_.push_back(f);
}
inline void ModelObject::addUVcoords(const UVcoords &uv)
{
  mapping_coords_.push_back(uv);
}

inline void ModelObject::setFaceMaterial(const unsigned short &f,
                                         const unsigned short &mat)
{
  // set the mat index for a given face
  faces_.at(f).mat = mat;
}



#endif // MODEL_OBJECTINL_H
