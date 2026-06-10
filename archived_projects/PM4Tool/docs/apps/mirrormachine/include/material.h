#ifndef MATERIAL_H
#define MATERIAL_H

#include <3D_types.h>


// Represents a material, with a lot of useless data.
// Colors are contained in vectors because they can have many values
// (like color + gamma corrected color)
class Material
{

public:

  std::string _name;
  std::string _path;

// these may be used later
//  unsigned short shininess;
//  unsigned short shininess_strength;
//  unsigned short transparency;
//  unsigned short transparency_falloff;
//  unsigned short reflection_blur;
//  unsigned short self_illum;
//  float wire_size;

  Material();

  // *** ACCESSORS ***

  Color24 getAmbientColor() const;
  Color24 getDiffuseColor() const;
  bool isTwoSided() const;

  // *** MUTATORS ***

  void setTwoSided(bool b);
  void setFlags(const unsigned short &fl);
  void setAmbientColor(const Color24 &color);
  void setDiffuseColor(const Color24 &color);
  void setSpecularColor(const Color24 &color);
  void setShadingValue(const unsigned short &sv);

private:

  // is the material two-sided ?
  // trigger the "disable backface-culling" flag in WMOs
  bool _two_sided;

  // flags ?
  // not really defined, as it should be cross-format flags
  unsigned short _flags;

  // colors - contain 2 colors each
  // 1st the used color, the second is some alpha stuff from 3DS
  std::vector<Color24> _ambient_color;
  std::vector<Color24> _diffuse_color;
  std::vector<Color24> _specular_color;

  // shading value - not used
  unsigned short _shading_value;

};

#include <material-inl.h>


#endif // MATERIAL_H
