#ifndef MATERIALINL_H
#define MATERIALINL_H

class Material;

// *** ACCESSORS ***

inline Color24 Material::getAmbientColor() const
{
  return _ambient_color.at(0);
}
inline Color24 Material::getDiffuseColor() const
{
  return _diffuse_color.at(0);
}
inline bool Material::isTwoSided() const
{
  return _two_sided;
}

// *** MUTATORS ***

inline void Material::setTwoSided(bool b)
{
  _two_sided = b;
}
inline void Material::setFlags(const unsigned short &fl)
{
  _flags = fl;
}
inline void Material::setAmbientColor(const Color24 &color)
{
  _ambient_color.at(0) = color;
}
inline void Material::setDiffuseColor(const Color24 &color)
{
  _diffuse_color.at(0) = color;
}
inline void Material::setSpecularColor(const Color24 &color)
{
  _specular_color.at(0) = color;
}
inline void Material::setShadingValue(const unsigned short &sv)
{
  _shading_value = sv;
}

#endif // MATERIALINL_H
