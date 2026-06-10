#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include <bsptreegenerator.h>



CBoundingBox& CBoundingBox::operator=(const CBoundingBox& bbox)
{
  a_.copy(bbox.a_);
  b_.copy(bbox.b_);
  return *this;
}


std::vector<CBoundingBox> CBoundingBox::SplitBox(SplittingMethod method,
                                                 float* f_dist) const
{
  std::vector<CBoundingBox> children(2, *this);
  // abs() shouldn't be needed as A is supposed to be always <= b ...
  float average;
  switch (method)
  {
    case YZ:
      average = round((b_.x + a_.x) / 2.0);
      children[0].b_.x = average;
      children[1].a_.x = average;
      break;
    case XZ:
      average = round((b_.y + a_.y) / 2.0);
      children[0].b_.y = average;
      children[1].a_.y = average;
      break;
    case XY:
      average = round((b_.z + a_.z) / 2.0);
      children[0].b_.z = average;
      children[1].a_.z = average;
      break;
  }
  *f_dist = average;
  return children;
}



/*unsigned int CBoundingBox::GetNumFaces(
    const std::vector<Face>& faces, const std::vector<Vertex>& vertices) const
{
  int num_faces = 0;
  for (size_t i = 0; i < faces.size(); ++i)
  {
    const Face& f = faces.at(i);
    if (ContainsPoly(vertices.at(f.a), vertices.at(f.b), vertices.at(f.c)))
      ++num_faces;
  }
  return num_faces;
}*/


void CBoundingBox::Enlarge(float f)
{
  a_.x -= f;
  a_.y -= f;
  a_.z -= f;
  b_.x += f;
  b_.y += f;
  b_.z += f;
}




std::string CBoundingBox::ToString() const
{
  std::stringstream ss;
  ss << "[ [" << a_.x << ", " << a_.y << ", " << a_.z << "], "
        " [" << b_.x << ", " << b_.y << ", " << b_.z << "] ]";
  return ss.str();
}




////////////////////////////////////////////////////////////////////////////////



void BSPTreeGenerator::Process()
{
  CBoundingBox root_box(object_->bbox_);
  /*root_box.Enlarge(0.001);*/

  // Let's reserve 2^15 nodes for our array.
  // I guess no tree will ever grow this large, and I have to reserve the place
  // as I'm push_backing the nodes to keep their relative positions.
  object_->bsp_nodes_.clear();
  object_->bsp_nodes_.reserve(32768);

  BspNode root_node;
  root_node.refs.resize(object_->faces_.size());
  for (unsigned int i = 0; i < object_->faces_.size(); ++i)
    root_node.refs.at(i) = i;
  object_->bsp_nodes_.push_back(root_node);

  ProcessNode(root_box, &(object_->bsp_nodes_.at(0)));

  object_->bsp_refs_ = GetBspRefList();

  return;
}



void BSPTreeGenerator::ProcessNode(const CBoundingBox& bbox, BspNode* node)
{
  // If the node has less than a certain amount of references,
  // we stop the processing there.
  if (node->refs.size() <= max_leaf_size_ /* for obvious reasons */)
  {
    node->plane_type = BspNode::LEAF;
    return;
  }

  // Else, prepares some vectors and split the node and the box.
  std::vector<CBoundingBox> bbox_children(2);
  std::vector<BspNode> node_children(2);
  node_children.at(0).refs.clear();
  node_children.at(1).refs.clear(); // is this useless ? shit
  CBoundingBox::SplittingMethod method_used =
      SplitBalanced(bbox, node, &bbox_children, &node_children);

  switch (method_used)
  {
    case CBoundingBox::YZ: node->plane_type = BspNode::YZ_PLANE; break;
    case CBoundingBox::XZ: node->plane_type = BspNode::XZ_PLANE; break;
    case CBoundingBox::XY: node->plane_type = BspNode::XY_PLANE; break;
    default: break;
  }

  // Once it's done, we store the nodes in our object.
  std::vector<BspNode>& nodes = object_->bsp_nodes_;

  // If a child has no refs (refs.size() == 0), there is no meaning in storing
  // it so I put a NULL ptr. The exporter will do the difference.

  if (node_children.at(1).refs.size() != 0)
  {
    // As this algorithm sucks a bit, it can happen that the box is reduced
    // to a simple plane, with more refs than the limit, and it loops until
    // it overflows the memory. This condition tests such case and force
    // the child node to be a leaf.
    if (node_children.at(1).refs.size() == node->refs.size())
    {
      node_children.at(1).plane_type = BspNode::LEAF;
      nodes.push_back(node_children.at(1));
      node->child2 = &(nodes.at(nodes.size() - 1));
    }
    else
    {
      nodes.push_back(node_children.at(1));
      node->child2 = &(nodes.at(nodes.size() - 1));
      ProcessNode(bbox_children.at(1), node->child2);
    }
  }
  else
  {
    node->child2 = NULL;
  }

  // Same for the other child. Yeah it's child2 then child1. Don't know why.
  // Don't know if it even matters.

  if (node_children.at(0).refs.size() != 0)
  {
    if (node_children.at(1).plane_type == node->plane_type &&
        node_children.at(1).f_dist == node->f_dist)
    {
      node_children.at(0).plane_type = BspNode::LEAF;
      nodes.push_back(node_children.at(0));
      node->child1 = &(nodes.at(nodes.size() - 1));
    }
    else
    {
      nodes.push_back(node_children.at(0));
      node->child1 = &(nodes.at(nodes.size() - 1));
      ProcessNode(bbox_children.at(0), node->child1);
    }
  }
  else
  {
    node->child1 = NULL;
  }


}





void BSPTreeGenerator::GetFacesReferences(
    const CBoundingBox& bbox,
    const std::vector<BspRef>& references,
    BspNode* node)
{
  node->refs.clear();
  std::vector<Face>& faces = object_->faces_;
  std::vector<Vertex>& vertices = object_->vertices_;

  for (size_t i = 0; i < references.size(); ++i)
  {
    /*std::cout << "ref " << i << ", ";*/
    const Face& f = faces.at(references.at(i));
    if (bbox.ContainsPoly(vertices.at(f.a), vertices.at(f.b), vertices.at(f.c)))
      node->refs.push_back(references.at(i));
  }
}





CBoundingBox::SplittingMethod BSPTreeGenerator::SplitBalanced(
    const CBoundingBox& bbox,
    BspNode* father,
    std::vector<CBoundingBox>* bbox_children,
    std::vector<BspNode>* children)
{
  CBoundingBox::SplittingMethod best_method = CBoundingBox::YZ;
  std::vector<BspNode> node_children_yz(2);
  std::vector<BspNode> node_children_xz(2);
  std::vector<BspNode> node_children_xy(2);
  int lower_balance, balance;

  // children (BspNodes) is filled for each test.
  // bbox_children isn't, and only split is used until the end of the function.

  // The lower_balance == 0 blocks are kinda verbose but I guess it may avoid
  // lots of useless calculations.
  float f_yz = 0.0;
  std::vector<CBoundingBox> split_yz = bbox.SplitBox(CBoundingBox::YZ, &f_yz);
  GetFacesReferences(split_yz.at(0), father->refs, &node_children_yz.at(0));
  GetFacesReferences(split_yz.at(1), father->refs, &node_children_yz.at(1));
  lower_balance = abs(node_children_yz.at(0).refs.size() -
                      node_children_yz.at(1).refs.size());
  /*if (lower_balance == 0)
  {
    *bbox_children = split_yz;
    father->plane_type = BspNode::YZ_PLANE;
    return;
  }*/

  /*int debug_balance_yz = lower_balance;*/

  float f_xz = 0.0;
  std::vector<CBoundingBox> split_xz = bbox.SplitBox(CBoundingBox::XZ, &f_xz);
  GetFacesReferences(split_xz.at(0), father->refs, &node_children_xz.at(0));
  GetFacesReferences(split_xz.at(1), father->refs, &node_children_xz.at(1));
  balance = abs(node_children_xz.at(0).refs.size() -
                node_children_xz.at(1).refs.size());
  /*if (balance == 0)
  {
    *bbox_children = split_xz;
    father->plane_type = BspNode::XZ_PLANE;
    return;
  }*/
  if (lower_balance > balance)
  {
    lower_balance = balance;
    best_method = CBoundingBox::XZ;
  }

  /*int debug_balance_xz = balance;*/

  float f_xy = 0.0;
  std::vector<CBoundingBox> split_xy = bbox.SplitBox(CBoundingBox::XY, &f_xy);
  GetFacesReferences(split_xy.at(0), father->refs, &node_children_xy.at(0));
  GetFacesReferences(split_xy.at(1), father->refs, &node_children_xy.at(1));
  balance = abs(node_children_xy.at(0).refs.size() -
                node_children_xy.at(1).refs.size());
  /*if (balance == 0)
  {
    *bbox_children = split_xy;
    father->plane_type = BspNode::XY_PLANE;
    return;
  }*/
  if (lower_balance > balance)
  {
    best_method = CBoundingBox::XY;
  }

  /*int debug_balance_xy = balance;*/

  switch (best_method)
  {
    case CBoundingBox::YZ:
      bbox_children->at(0) = split_yz.at(0);
      bbox_children->at(1) = split_yz.at(1);
      for (size_t i = 0; i < node_children_yz.at(0).refs.size(); ++i)
        children->at(0).refs.push_back(node_children_yz.at(0).refs.at(i));
      for (size_t i = 0; i < node_children_yz.at(1).refs.size(); ++i)
        children->at(1).refs.push_back(node_children_yz.at(1).refs.at(i));
      father->f_dist = f_yz;
      break;
    case CBoundingBox::XZ:
      bbox_children->at(0) = split_xz.at(0);
      bbox_children->at(1) = split_xz.at(1);
      for (size_t i = 0; i < node_children_xz.at(0).refs.size(); ++i)
        children->at(0).refs.push_back(node_children_xz.at(0).refs.at(i));
      for (size_t i = 0; i < node_children_xz.at(1).refs.size(); ++i)
        children->at(1).refs.push_back(node_children_xz.at(1).refs.at(i));
      father->f_dist = f_xz;
      break;
    case CBoundingBox::XY:
      bbox_children->at(0) = split_xy.at(0);
      bbox_children->at(1) = split_xy.at(1);
      for (size_t i = 0; i < node_children_xy.at(0).refs.size(); ++i)
        children->at(0).refs.push_back(node_children_xy.at(0).refs.at(i));
      for (size_t i = 0; i < node_children_xy.at(1).refs.size(); ++i)
        children->at(1).refs.push_back(node_children_xy.at(1).refs.at(i));
      father->f_dist = f_xy;
      break;
  }

  /*std::cout << "BALANCES :: yz = " << debug_balance_yz << ",  "
            << "xz = " << debug_balance_xz << ", "
            << "xy = " << debug_balance_xy << std::endl;*/

  return best_method;
}





std::vector<BspRef> BSPTreeGenerator::GetBspRefList() const
{
  std::vector<BspRef> references;
  /*std::vector<BspRef> children_references =
      GetBspRefListR(object_->bsp_nodes_.at(0));
  references.insert(
      references.end(), children_references.begin(), children_references.end());
*/
  for (size_t i = 0; i < object_->bsp_nodes_.size(); ++i)
  {
    const BspNode& node = object_->bsp_nodes_.at(i);
    if (node.plane_type == BspNode::LEAF)
    {
      references.insert(references.end(), node.refs.begin(), node.refs.end());
    }
  }

  return references;
}



std::vector<BspRef> BSPTreeGenerator::GetBspRefListR(const BspNode& node) const
{
  std::vector<BspRef> references;
  if (node.plane_type == BspNode::LEAF)
  {
    references.insert(references.end(), node.refs.begin(), node.refs.end());
  }
  else
  {
    std::vector<BspRef> child1_refs = GetBspRefListR(*node.child1);
    std::vector<BspRef> child2_refs = GetBspRefListR(*node.child2);
    references.insert(references.end(), child1_refs.begin(), child1_refs.end());
    references.insert(references.end(), child2_refs.begin(), child2_refs.end());
  }
  return references;
}






/*CBoundingBox::SplittingMethod BSPTreeGenerator::GetSplitMethodBalanced(
    const CBoundingBox& bbox) const
{
  // For each test, store the balance (ex : (15,12) has a balance of 3)
  // in the variable balance, and in lower_balance the lower balance found.
  int lower_balance, balance;
  CBoundingBox::SplittingMethod best_method = CBoundingBox::YZ;

  std::vector<Face>& faces = object_->faces_;
  std::vector<Vertex>& vertices = object_->vertices_;

  std::vector<CBoundingBox> children = bbox.SplitBox(CBoundingBox::YZ);
  children.at(0).set_num_polys( children.at(0).GetNumFaces(faces, vertices) );
  children.at(1).set_num_polys( children.at(1).GetNumFaces(faces, vertices) );
  lower_balance = GetBalance(children);
  if (lower_balance == 0)
    return CBoundingBox::YZ;

  children = bbox.SplitBox(CBoundingBox::XZ);
  children.at(0).set_num_polys( children.at(0).GetNumFaces(faces, vertices) );
  children.at(1).set_num_polys( children.at(1).GetNumFaces(faces, vertices) );
  balance = GetBalance(children);
  if (balance == 0)
    return CBoundingBox::XZ;
  if (lower_balance > balance)
  {
    lower_balance = balance;
    best_method = CBoundingBox::XZ;
  }

  children = bbox.SplitBox(CBoundingBox::XY);
  children.at(0).set_num_polys( children.at(0).GetNumFaces(faces, vertices) );
  children.at(1).set_num_polys( children.at(1).GetNumFaces(faces, vertices) );
  balance = GetBalance(children);
  if (balance == 0)
    return CBoundingBox::XY;
  if (lower_balance > balance)
  {
    lower_balance = balance;
    best_method = CBoundingBox::XY;
  }

  switch (best_method)
  {
    case CBoundingBox::YZ: std::cout << "bm : YZ" << std::endl; break;
    case CBoundingBox::XZ: std::cout << "bm : XZ" << std::endl; break;
    case CBoundingBox::XY: std::cout << "bm : XY" << std::endl; break;
  }

  return best_method;
}*/


