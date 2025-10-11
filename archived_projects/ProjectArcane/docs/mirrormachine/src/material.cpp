#include <string>
#include <vector>

#include <material.h>


Material::Material()
    : _name("")
    , _path("")
    , _two_sided(false)
    , _flags(0)
    , _ambient_color(2)
    , _diffuse_color(2)
    , _specular_color(2)
    , _shading_value(0)
{

}
