#ifndef _3D_TYPES_H
#define _3D_TYPES_H

#include <cmath>
#include <vector>
#include <QtGlobal>

class vector;


// ----------------------------------------
// 3D DATA

// a Normal is a 3D vector
// are the defined functions useless as the compiler can generate them ?
struct Normal
{
  float x, y, z;

  void set(float x_, float y_, float z_) { x = x_; y = y_; z = z_; }
  void copy(const Normal& n) { x = n.x; y = n.y; z = n.z; }
  void add(const Normal& n) { x += n.x; y += n.y; z += n.z; }
};

// a Vertex is a 3D position in space
// its Normal member is representing a vertex normal
struct Vertex
{
  float x, y, z;
  Normal n;

  void set(float x_, float y_, float z_) { x = x_; y = y_; z = z_; }
  void copy(const Vertex& v) {x = v.x; y = v.y; z = v.z;  }
};

// a bounding box is a box in space represented by two corners
struct BoundingBox
{
  Vertex a, b;
};

// a Face is pointing to 3 vertices through indices in some container
// the Normal n represents face normal, used for vertex normals.
// flags represents 3DS flags and is not used.
// mat represents a material index somewhere in a material container
struct Face
{
  quint16 a, b, c;
  Normal n;
  unsigned short flags_3DS;
  quint8 mat;
};

// a UVcoords struct is used in UV map, to apply textures on the model
struct UVcoords
{
  float u, v;

  void copy(const UVcoords& uv) { u = uv.u; v = uv.v; }
};

// A reference to a face in a face array.
// Used in MOBR (actually it's an uint16_t but whatever)
typedef unsigned int BspRef;

// Representation of a BSP tree node.
// Can be either a node or a leaf, depending on the plane_type.
// refs stores the references to the object.face array bind to the BSP tree.
// f_dist is unused.
struct BspNode
{
  enum PlaneType { YZ_PLANE, XZ_PLANE, XY_PLANE, LEAF };

  PlaneType plane_type;
  BspNode* child1;
  BspNode* child2;

  std::vector<BspRef> refs;

  float f_dist;
};




// normalizes the Normal *n
inline void normalize(Normal *n)
{
  const float length = sqrt((n->x * n->x) + (n->y * n->y) + (n->z * n->z));
  n->x /= length;
  n->y /= length;
  n->z /= length;
}

// calculate the cross product between two Normals n1 X n2
inline Normal crossProduct(const Normal &n1, const Normal &n2)
{
  Normal n;
  n.x = (n1.y * n2.z) - (n1.z * n2.y);
  n.y = (n1.z * n2.x) - (n1.x * n2.z);
  n.z = (n1.x * n2.y) - (n1.y * n2.x);
  return n;
}

// calculate the dot product between two 3D types
// Normal.Normal
inline float dotProduct(const Normal &n1, const Normal &n2)
{
  return n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
}
// Vertex.Normal
inline float dotProduct(const Vertex &v, const Normal &n)
{
  return v.x * n.x + v.y * n.y + v.z * n.z;
}

// computes the vector of the difference v1 - v2
// (returns a Normal type which isn't really the same thing...)
inline Normal vertexDifference(const Vertex &v1, const Vertex &v2)
{
  Normal n;
  n.set(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
  return n;
}





// ----------------------------------------
// COLORS

// color struct, 8bit RGB
struct Color24
{
  quint8 r, g, b;
};



#endif // 3D_TYPES_H
