#ifndef BSPTREEGENERATOR_H
#define BSPTREEGENERATOR_H

#include <vector>

#include <QSettings>

#include <3D_types.h>
#include <model_object.h>

class string;



// Represents a bounding box with it's bottom-behind-left corner
// and the top-front-right one (or sth like that),
// with a bunch of useful functions.
// It's called CBBox because BBox is already a small structure.
class CBoundingBox
{

 public:

  enum SplittingMethod { YZ, XZ, XY };


  CBoundingBox() { }
  CBoundingBox(const BoundingBox& bbox) : a_(bbox.a), b_(bbox.b) { }
  CBoundingBox(const Vertex& a, const Vertex& b) : a_(a), b_(b) { }
  ~CBoundingBox() { }

  CBoundingBox& operator=(const CBoundingBox& bbox);

  Vertex a() const { return a_; }
  Vertex b() const { return b_; }
  /*size_t num_polys() const { return num_polys_; }*/
  void set_a(const Vertex& a) { a_ = a; }
  void set_b(const Vertex& b) { b_ = b; }
  /*void set_num_polys(size_t n) { num_polys_ = n; }*/

  // Returns true if the vertex v is inside the box.
  inline bool ContainsVertex(const Vertex& v) const
  {
    return ((v.x >= a_.x && v.x <= b_.x) &&
            (v.y >= a_.y && v.y <= b_.y) &&
            (v.z >= a_.z && v.z <= b_.z));
  }
  // Returns true if ONE of the 3 vertices of the face is inside the box.
  inline bool ContainsPoly(
      const Vertex& v1, const Vertex& v2, const Vertex& v3) const
  {
    return (ContainsVertex(v1) || ContainsVertex(v2) || ContainsVertex(v3));
  }

  // Split the box in two other boxes
  // Result : the 2 boxes in a vector of size 2.
  std::vector<CBoundingBox> SplitBox(SplittingMethod method,
                                     float* f_dist) const;



  // Returns the number of faces contained in this box.
  // This operations is quite heavy as it runs through the whole object.
  // You run it once and store this number in num_polys_.
  /*unsigned int GetNumFaces(const std::vector<Face>& faces,
                           const std::vector<Vertex>& vertices) const;*/


  // Because floats are dumb.
  void Enlarge(float f);

  // Shows the bounding box coordinates in a string, Java style !
  std::string ToString() const;

 private:

  // The corners. a_ is the LOWER, b_ the GREATER
  Vertex a_, b_;

  // Number of polys in the box.
  // Have to be set manually ; it will avoid redondant calculations.
  /*size_t num_polys_;*/

};



// Simple reference to the face list of an object,
// as stored in MOBR.
typedef unsigned int BspReference;








// This generator can take a ModelObject and create a BSP tree
// from its vertices/faces, to write correct MOBN/MOBR.
class BSPTreeGenerator
{

 public:

  BSPTreeGenerator(ModelObject* target) : object_(target)
  {
    QSettings opt("egamh9", "mirrormachine");
    max_leaf_size_ = opt.value("max_leaf_size", 300).toUInt();
  }

  // This is the function to run the generator and
  // fill a ModelObject with its values.
  void Process();

 private:

  // The recursive function applied to each node, starting from the root node.
  // Once the function call on the node returns, the tree is generated.
  void ProcessNode(const CBoundingBox& bbox, BspNode* node);

  // Searches for every polygon that is included in bbox :
  // every polygon here is added to node->refs.
  // Loops through the faces given in references (refs to object->faces)
  void GetFacesReferences(const CBoundingBox& bbox,
                          const std::vector<BspRef>& references, BspNode* node);

  // Split the couple bbox/father in the couples bbox_children/children
  // (both are vectors of size 2), choosing the most balanced way to do so.
  // Anyway, the children.refs vectors are ALWAYS built.
  CBoundingBox::SplittingMethod SplitBalanced(
      const CBoundingBox& bbox, BspNode* father,
      std::vector<CBoundingBox>* bbox_children, std::vector<BspNode>* children);

  // Returns the complete list of BspRef, just as it will be in MOBR.
  // In fact this is supposed to be directly assigned in object.bsp_refs.
  std::vector<BspRef> GetBspRefList() const;

  // Recursive sub-function of GetBspRefList who runs thru the tree.
  std::vector<BspRef> GetBspRefListR(const BspNode& node) const;

  // Finds the way of splitting a box that gives to more balanced results.
  // Ex : YZ gives (24,3), XZ gives (27,0), XY gives (15,12)
  // then the result will be XY.
  /*CBoundingBox::SplittingMethod GetSplitMethodBalanced(
      const CBoundingBox& bbox) const;*/

  // Calculates the balance between the two children of a box.
  // It reads the num_polys value, so compute it first.
  /*inline int GetBalance(const std::vector<CBoundingBox>& children) const
  {
    return abs(children.at(0).num_polys() - children.at(1).num_polys());
  }*/


  // The object the generator is working on.
  ModelObject* object_;

  // Pointer to the current BspRef list (linked to the node we're operating on).
  // It's at the object scope because it's useful for some separated functions
  // to know on what list operate and synchronize it simply.
  /*std::vector<BspRef>* references_ptr_;*/

  // Maximum number of faces that can be used for 1 leaf.
  // Refer to some documentation to know what's the best for you.
  unsigned int max_leaf_size_;

};



#endif // BSPTREEGENERATOR_H
