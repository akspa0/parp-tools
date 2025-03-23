#include <cstdio>

#include <binary_io.h>
#include <model_object.h>






void ModelObject::computeBoundingBox()
{
  Vertex v;
  const unsigned int nVertices = vertices_.size();

  // can't use (0,0,0) (0,0,0) as a base unless there are no vertices
  if (nVertices > 0)
  {
    v = vertices_.at(0);
    bbox_.a.x = v.x;
    bbox_.a.y = v.y;
    bbox_.a.z = v.z;
    bbox_.b.x = v.x;
    bbox_.b.y = v.y;
    bbox_.b.z = v.z;
  }

  // searches for an actual bounding box
  for (unsigned int i = 0; i < nVertices; ++i)
  {
    v = vertices_.at(i);
    if (v.x < bbox_.a.x) bbox_.a.x = v.x;
    if (v.y < bbox_.a.y) bbox_.a.y = v.y;
    if (v.z < bbox_.a.z) bbox_.a.z = v.z;
    if (v.x > bbox_.b.x) bbox_.b.x = v.x;
    if (v.y > bbox_.b.y) bbox_.b.y = v.y;
    if (v.z > bbox_.b.z) bbox_.b.z = v.z;
  }
}


// build the face/vertex normals with math and brain (and cp)
void ModelObject::buildFaceNormals()
{
  Normal tmp_n, edge1, edge2;

  const unsigned int nFaces = faces_.size();
  for (unsigned int f = 0; f < nFaces; ++f)
  {
    edge1 = vertexDifference(vertices_.at(faces_.at(f).b),
                             vertices_.at(faces_.at(f).a));
    edge2 = vertexDifference(vertices_.at(faces_.at(f).c),
                             vertices_.at(faces_.at(f).a));
    tmp_n = crossProduct(edge1, edge2);
    normalize(&tmp_n);
    faces_.at(f).n.copy(tmp_n);
  }
}

void ModelObject::buildVertexNormals()
{
  // classic way
  // ok but wrong shadows in acute angles

  /*const unsigned int nVertex = getNVertex();
  const unsigned int nFaces = getNFace();
  for (unsigned int v = 0; v < nVertex; ++v) {
    for (unsigned int f = 0; f < nFaces; ++f) {
      if (isUsingVertex(getFace(f), getVertex(v))) {
        vertices.at(v).n.add(faces.at(f).n);
      }
    }
    utils::normalize(&(vertices.at(v).n));
  }*/

  // Assimp way
  // without sorting optimisations
  // but with all the stuff I don't get
  // so it's all good :) (wait)

  const unsigned int kFaces = faces_.size();
  const unsigned int kVertices = vertices_.size();

  // stores the face normals in the vertex normal slot
  for (unsigned int i = 0; i < kFaces; ++i)
  {
    vertices_.at(faces_.at(i).a).n.copy(faces_.at(i).n);
    vertices_.at(faces_.at(i).b).n.copy(faces_.at(i).n);
    vertices_.at(faces_.at(i).c).n.copy(faces_.at(i).n);
  }

  // epsilon is our distance checker
  float epsilon = 1e-5f;
  float epsilon_squared = epsilon * epsilon;

  // default_plane is a random default plane
  Normal default_plane;
  default_plane.set(0.8532, 0.34321, 0.5736);
  normalize(&default_plane);

  Normal tmp_norm;

  // stores distances from the default plane to our vertices
  std::vector<float> distances(kVertices);
  for (unsigned int i = 0; i < kVertices; ++i)
  {
    distances.at(i) = dotProduct(vertices_.at(i), default_plane);
  }

  // for each vertex we will store there its near friends
  unsigned int n_near_vertices = 0;
  std::vector<unsigned int> near_vertices(0);

  // computes the normals
  float distance, min_distance, max_distance, square_distance;
  std::vector<Normal> computed_normals(kVertices);
  std::vector<bool> okay_mate(kVertices, false);
  for (unsigned int i = 0; i < kVertices; ++i)
  {
    if (okay_mate.at(i)) continue;

    // finds the vertices near enough
    // gets faces indices related
    near_vertices.clear();
    distance = dotProduct(vertices_.at(i), default_plane);
    min_distance = distance - epsilon;
    max_distance = distance + epsilon;
    for (unsigned int j = 0; j < kVertices; ++j)
    {
      if (distances.at(j) > min_distance && distances.at(j) < max_distance)
      {
        tmp_norm.set(vertices_.at(j).n.x - vertices_.at(i).n.x,
                     vertices_.at(j).n.y - vertices_.at(i).n.y,
                     vertices_.at(j).n.z - vertices_.at(i).n.z);
        square_distance = tmp_norm.x * tmp_norm.x +
            tmp_norm.y * tmp_norm.y +
            tmp_norm.z * tmp_norm.z;
        if (square_distance < epsilon_squared)
          near_vertices.push_back(j);
      }
    }

    // actually computes the normal
    tmp_norm.set(0, 0, 0);
    n_near_vertices = near_vertices.size();
    for (unsigned int j = 0; j < n_near_vertices; ++j)
    {
      tmp_norm.add(vertices_.at(near_vertices.at(j)).n);
    }
    normalize(&tmp_norm);

    // stores the computed normal in our vector
    // and marks this vertex index as computed
    for (unsigned int j = 0; j < n_near_vertices; ++j)
    {
      computed_normals.at(near_vertices.at(j)).copy(tmp_norm);
      okay_mate.at(near_vertices.at(j)) = true;
    }
  }

  // replace the old normals by the computed
  for (unsigned int i = 0; i < kVertices; ++i)
  {
    vertices_.at(i).n.copy(computed_normals.at(i));
  }
}




