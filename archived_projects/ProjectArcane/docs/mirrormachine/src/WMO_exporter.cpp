#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <QSettings>

#include <3D_types.h>
#include <binary_io.h>
#include <model.h>
#include <model_object.h>

#include <WMO_exporter.h>


WMO_exporter::WMO_exporter()
    : _chunks(0)
    , _groups(0)
    , _nBatches(0)
{

}


// exporting

int WMO_exporter::process(const Model& model, const char* path)
{
  QSettings opt("egamh9", "mirrormachine");

  /*GiveTharoTheData(model);*/

  // root
  add_MVER(17);
  add_MOHD(model);
  add_MOTX(model);
  add_MOMT(model);
  add_MOGN(model);
  add_MOGI(model);
  add_MOSB(model);
  add_MOPV(model);
  add_MOPT(model);
  add_MOPR(model);
  add_MOVV(model);
  add_MOVB(model);
  add_MOLT(model);
  add_MODS(model);
  add_MODN(model);
  add_MODD(model);
  add_MFOG(model);

  // groups
  unsigned int nGroups = model.objects_.size();
  for (unsigned int i = 0; i < nGroups; ++i)
  {
    Group new_group;
    _groups.push_back(new_group);

    // required chunks
    add_MVER(17, i);
    add_MOPY(model, i);
    add_MOVI(model, i);
    add_MOVT(model, i);
    add_MONR(model, i);
    add_MOTV(model, i);
    add_MOBA(model, i);

    // not so required
    switch (model._import_type)
    {
      case Model::FROM_WMO14:
      {
        if (opt.value("alpha_doodads", false).toBool())
          if (model.objects_.at(i)._WMO14_mogp.flags & WMO14::HAS_DOODADS)
            add_MODR(model, i);

        add_MOBN(model, i);
        add_MOBR(model, i);

        if (opt.value("alpha_colors", false).toBool())
          if (model.objects_.at(i)._WMO14_mogp.flags & WMO14::HAS_COLORS)
            add_MOCV(model, i);
      }
        break;

      default:
        add_MOBN(model, i);
        add_MOBR(model, i);
    }

    // see ?
    add_MOGP(model, i);
  }

  int result = 0;
  result += write_root(path);
  result += write_groups(path);
  return result;
}





// build the file

void WMO_exporter::addChunk(const Chunk &ck)
{
  _chunks.push_back(ck);
}
void WMO_exporter::addChunkToGroup(const Chunk &ck, const unsigned int &gr)
{
  _groups.at(gr).chunks.push_back(ck);
}


// build root

void WMO_exporter::add_MVER(const unsigned int &version)
{
  std::vector<char> data = bin_io::vc_Int(version);
  Chunk c = {"MVER", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}


void WMO_exporter::add_MOHD(const Model &model)
{
  QSettings opt("egamh9", "mirrormachine");
  quint32 nMaterials = model.materials_.size();
  quint32 nGroups = model.objects_.size();
  quint32 nPortals = 0;
  quint32 nLights = 0;
  quint32 nModels = 0;
  quint32 nDoodads = 0;
  quint32 nSets = 1;
  quint32 wmo_ID = 0;
  quint32 liquid_type = 0;

  if (model._import_type == Model::FROM_WMO14)
  {
    if (opt.value("alpha_portals", false).toBool())
    {
      nPortals = model._WMO14_mohd.nPortals;
    }

    if (opt.value("alpha_doodads", false).toBool())
    {
      nModels = model._WMO14_mohd.nDoodad_names;
      nDoodads = model._WMO14_mohd.nDoodad_definitions;
      nSets = model._WMO14_mohd.nDoodad_sets;
    }

  }

  std::vector<char> data, temp_data;

  // nMaterials
  temp_data = bin_io::vc_Int(nMaterials);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  // nGroups
  temp_data = bin_io::vc_Int(nGroups);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  // nPortals
  temp_data = bin_io::vc_Int(nPortals);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  // nLights
  temp_data = bin_io::vc_Int(nLights);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  // nModels
  temp_data = bin_io::vc_Int(nModels);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  // nDoodads
  temp_data = bin_io::vc_Int(nDoodads);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  // nSets
  temp_data = bin_io::vc_Int(nSets);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  // ambient color
  data.push_back(model._ambient_color.at(0));
  data.push_back(model._ambient_color.at(1));
  data.push_back(model._ambient_color.at(2));
  data.push_back(0xff);
  // ID
  temp_data = bin_io::vc_Int(wmo_ID);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  // bounding box
  temp_data = bin_io::vc_Float(model.bbox_.a.x);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Float(model.bbox_.a.y);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Float(model.bbox_.a.z);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Float(model.bbox_.b.x);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Float(model.bbox_.b.y);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Float(model.bbox_.b.z);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  // liquid type
  temp_data = bin_io::vc_Int(liquid_type);
  data.insert(data.end(), temp_data.begin(), temp_data.end());

  Chunk c = {"MOHD", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}


void WMO_exporter::add_MOTX(const Model &model)
{
  QSettings opt("egamh9", "mirrormachine");
  std::vector<char> data;

  unsigned int nTextures = 0;
  for (unsigned int i = 0; i < model.materials_.size(); ++i)
    if (!model.materials_.at(i)._path.empty())
      ++nTextures;

  // if there is no texture, adds a gray BLP if asked
  if (nTextures == 0 && opt.value("gray_missing_texture", true).toBool())
  {
    std::string gray_blp = "DUNGEONS\\TEXTURES\\STORMWIND\\GRAY01.BLP";
    for (unsigned int l = 0; l < gray_blp.length(); ++l)
      data.push_back(gray_blp.at(l));
    data.push_back(0x00);
    while (data.size() % 4 != 0)
      data.push_back(0x00);
  }
  // else, writes down the textures paths with this elegant padding
  else
  {
    std::vector<std::string> textures = model.getTexturesPaths();
    for (unsigned int i = 0; i < textures.size(); ++i)
    {
      for (unsigned int j = 0; j < textures.at(i).length(); ++j)
        data.push_back(textures.at(i).at(j));
      data.push_back(0x00); // '\0'
      while (data.size() % 4 != 0) // complete the 4-bytes alignment
        data.push_back(0x00);
      data.push_back(0x00); // force a 4-byte padding
      while (data.size() % 4 != 0) // complete the padding
        data.push_back(0x00);
    }
  }

  Chunk c = {"MOTX", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}


void WMO_exporter::add_MOMT(const Model &model)
{
  QSettings opt("egamh9", "mirrormachine");
  std::vector<char> data, temp_data;
  Color24 color;

  const unsigned int nMaterials = model.materials_.size();
  for (unsigned int i = 0; i < nMaterials; ++i)
  {
    // flags1
    unsigned int flags = 0x00;
    if (model.materials_.at(i).isTwoSided() ||
        opt.value("disable_bfc", false).toBool())
      flags += 0x04; // disable backface culling
    temp_data = bin_io::vc_Int(flags);
    data.insert(data.end(), temp_data.begin(), temp_data.end());

    // shader
    temp_data = bin_io::vc_Int(0);
    data.insert(data.end(), temp_data.begin(), temp_data.end());

    // blend mode
    data.insert(data.end(), temp_data.begin(), temp_data.end());

    // texture1 path
    temp_data = bin_io::vc_Int(model.getTexturePathOffsets(i).at(0));
    data.insert(data.end(), temp_data.begin(), temp_data.end());

    // color1
    color = model.materials_.at(i).getAmbientColor();
    data.push_back(color.r);
    data.push_back(color.g);
    data.push_back(color.b);
    data.push_back(0xFF);

    // flags1
    temp_data = bin_io::vc_Int(0);
    data.insert(data.end(), temp_data.begin(), temp_data.end());

    // texture2
    temp_data = bin_io::vc_Int(model.getTexturePathOffsets(i).at(1));
    data.insert(data.end(), temp_data.begin(), temp_data.end());

    // color2
    color = model.materials_.at(i).getDiffuseColor();
    data.push_back(color.r);
    data.push_back(color.g);
    data.push_back(color.b);
    data.push_back(0xFF);

    // flags2 - 0x0C valid dummy flag ?
    temp_data = bin_io::vc_Int(0);
    data.insert(data.end(), temp_data.begin(), temp_data.end());

    // unk 1..7 (uint ? float ?)
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    data.insert(data.end(), temp_data.begin(), temp_data.end());

  }

  Chunk c = {"MOMT", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}


void WMO_exporter::add_MOGN(const Model &model)
{
  std::vector<char> data;
  data.push_back(0x00);
  data.push_back(0x00); // really needed ?
  std::vector<std::string> group_names = model.getGroupNames();
  std::string groupname;
  unsigned int ofs = 0;

  const unsigned int nGroupNames = group_names.size();
  for (unsigned int i = 0; i < nGroupNames; ++i)
  {
    groupname = group_names.at(i);
    while (ofs < groupname.length())
    {
      data.push_back(groupname.at(ofs));
      ++ofs;
    }
    data.push_back(0x00); // c_str '\0'
    ofs = 0;
  }
  while (data.size() % 4 != 0) // 4-byte padding
    data.push_back(0x00);

  Chunk c = {"MOGN", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}


void WMO_exporter::add_MOGI(const Model &model)
{
  QSettings opt("egamh9", "mirrormachine");

  std::vector<char> data, temp_data;

  const unsigned int nObjects = model.objects_.size();
  for (unsigned int id_obj = 0; id_obj < nObjects; ++id_obj)
  {
    // flags
    unsigned int flags = 0;
    if (opt.value("indoor", false).toBool())
      flags += 0x2000; // indoor
    if (opt.value("outdoor", true).toBool())
      flags += 0x8; // outdoor
    temp_data = bin_io::vc_Int(flags);
    data.insert(data.end(), temp_data.begin(), temp_data.end());

    // bounding box
    temp_data = bin_io::vc_Float(model.objects_.at(id_obj).bbox_.a.x);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    temp_data = bin_io::vc_Float(model.objects_.at(id_obj).bbox_.a.y);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    temp_data = bin_io::vc_Float(model.objects_.at(id_obj).bbox_.a.z);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    temp_data = bin_io::vc_Float(model.objects_.at(id_obj).bbox_.b.x);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    temp_data = bin_io::vc_Float(model.objects_.at(id_obj).bbox_.b.y);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    temp_data = bin_io::vc_Float(model.objects_.at(id_obj).bbox_.b.z);
    data.insert(data.end(), temp_data.begin(), temp_data.end());

    // index in MOGN - can be calculated from our i, but we put -1
    temp_data = bin_io::vc_Int(-1);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
  }
  Chunk c = {"MOGI", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}


void WMO_exporter::add_MOSB(const Model &/*model*/)
{
  std::vector<char> data, temp_data;
  temp_data = bin_io::vc_Int(0);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  Chunk c = {"MOSB", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}


void WMO_exporter::add_MOPV(const Model &model)
{
  QSettings opt("egamh9", "mirrormachine");

  std::vector<char> data, temp_data;

  if ((model._import_type == Model::FROM_WMO14) &&
      (opt.value("alpha_portals", false).toBool()))
  {
    WMO14::PortalVertices pv;
    unsigned int nPortals = model._WMO14_mopv.size();
    for (unsigned int i = 0; i < nPortals; ++i)
    {
      pv = model._WMO14_mopv.at(i);
      for (unsigned int vert = 0; vert < 4; ++vert)
      {
        temp_data = bin_io::vc_Float(pv.vertices[vert].x);
        data.insert(data.end(), temp_data.begin(), temp_data.end());
        temp_data = bin_io::vc_Float(pv.vertices[vert].y);
        data.insert(data.end(), temp_data.begin(), temp_data.end());
        temp_data = bin_io::vc_Float(pv.vertices[vert].z);
        data.insert(data.end(), temp_data.begin(), temp_data.end());
      }
    }
  }

  Chunk c = {"MOPV", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}


void WMO_exporter::add_MOPT(const Model &model)
{
  QSettings opt("egamh9", "mirrormachine");

  std::vector<char> data, temp_data;

  if ((model._import_type == Model::FROM_WMO14) &&
      (opt.value("alpha_portals", false).toBool()))
  {
    WMO14::PortalInfo info;
    unsigned int nPortals = model._WMO14_mopt.size();
    for (unsigned int i = 0; i < nPortals; ++i)
    {
      info = model._WMO14_mopt.at(i);
      temp_data = bin_io::vc_Short(info.base_index);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Short(info.nVertices);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(info.vector[0]);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(info.vector[1]);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(info.vector[2]);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(info.unknown);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
    }
  }

  Chunk c = {"MOPT", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}


void WMO_exporter::add_MOPR(const Model &model)
{
  QSettings opt("egamh9", "mirrormachine");

  std::vector<char> data, temp_data;

  if ((model._import_type == Model::FROM_WMO14) &&
      (opt.value("alpha_portals", false).toBool()))
  {
    WMO14::PortalRelation relation;
    unsigned int nRels = model._WMO14_mopr.size();
    for (unsigned int i = 0; i < nRels; ++i)
    {
      relation = model._WMO14_mopr.at(i);
      temp_data = bin_io::vc_Short(relation.portal_index);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Short(relation.group_index);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Short(relation.side);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Short(relation.filler);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
    }
  }

  Chunk c = {"MOPR", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}


void WMO_exporter::add_MOVV(const Model &/*model*/)
{
  std::vector<char> data;
  Chunk c = {"MOVV", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}


void WMO_exporter::add_MOVB(const Model &/*model*/)
{
  std::vector<char> data;
  Chunk c = {"MOVB", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}


void WMO_exporter::add_MOLT(const Model &/*model*/)
{
  std::vector<char> data;
  Chunk c = {"MOLT", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}


void WMO_exporter::add_MODS(const Model &model)
{
  QSettings opt("egamh9", "mirrormachine");

  std::vector<char> data, temp_data;

  if ((model._import_type == Model::FROM_WMO14) &&
      (opt.value("alpha_doodads", false).toBool()))
  {
    WMO14::DoodadSet set;
    unsigned int nSets = model._WMO14_mods.size();
    for (unsigned int i = 0; i < nSets; ++i)
    {
      set = model._WMO14_mods.at(i);
      for (unsigned char c = 0; c < 20; ++c)
        data.push_back(set.name[c]);
      temp_data = bin_io::vc_Int(set.first_instance);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Int(set.nDoodads);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Int(set.unk);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
    }
  }
  else
  {
    std::string default_set_name = "Set_$DefaultGlobal";
    data.insert(data.end(), &default_set_name[0], &default_set_name[18]);
    data.push_back(0x00);
    data.push_back(0x00); // to reach char[20]
    temp_data = bin_io::vc_Int(0);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    data.insert(data.end(), temp_data.begin(), temp_data.end());
  }

  Chunk c = {"MODS", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}


void WMO_exporter::add_MODN(const Model &model)
{
  QSettings opt("egamh9", "mirrormachine");

  std::vector<char> data;

  if ((model._import_type == Model::FROM_WMO14) &&
      (opt.value("alpha_doodads", false).toBool()))
  {
    for (unsigned int i = 0; i < model._WMO14_modn.model_paths.size(); ++i)
      data.push_back(model._WMO14_modn.model_paths.at(i));
  }

  Chunk c = {"MODN", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}


void WMO_exporter::add_MODD(const Model &model)
{
  QSettings opt("egamh9", "mirrormachine");

  std::vector<char> data, temp_data;

  if ((model._import_type == Model::FROM_WMO14) &&
      (opt.value("alpha_doodads", false).toBool()))
  {
    WMO14::DoodadDefinition def;
    const unsigned int nDefs = model._WMO14_modd.size();
    for (unsigned int i = 0; i < nDefs; ++i)
    {
      def = model._WMO14_modd.at(i);
      temp_data = bin_io::vc_Int(def.name_offset);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(def.position[0]);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(def.position[1]);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(def.position[2]);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(def.rotation[0]);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(def.rotation[1]);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(def.rotation[2]);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(def.rotation[3]);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(def.scale);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Int(def.color);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
    }
  }

  Chunk c = {"MODD", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}


void WMO_exporter::add_MFOG(const Model &/*model*/)
{
  std::vector<char> data, temp_data;
  // ***** create basic MFOG entry *****
  // flags
  temp_data = bin_io::vc_Int(0);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  // position
  temp_data = bin_io::vc_Float(0.0);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  // radius (small then large)
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  // fog end
  temp_data = bin_io::vc_Float(444.4445);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  // fog start multiplier
  temp_data = bin_io::vc_Float(0.25);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  // color 1 (white, full alpha)
  data.push_back(0xFF);
  data.push_back(0xFF);
  data.push_back(0xFF);
  data.push_back(0xFF);
  // unknown floats
  temp_data = bin_io::vc_Float(222.2222);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Float(-0.5);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  // color 2 (red, full alpha)
  data.push_back(0xFF);
  data.push_back(0x00);
  data.push_back(0x00);
  data.push_back(0xFF);
  Chunk c = {"MFOG", static_cast<unsigned int>(data.size()), data};
  addChunk(c);
}





// build groups

void WMO_exporter::add_MVER(const unsigned int &version,
                            const unsigned int &gr)
{
  std::vector<char> data = bin_io::vc_Int(version);
  Chunk c = {"MVER", static_cast<unsigned int>(data.size()), data};
  addChunkToGroup(c, gr);
}


void WMO_exporter::add_MOGP(const Model &model, const unsigned int &gr)
{
  QSettings opt("egamh9", "mirrormachine");

  unsigned int flags = 1;
  unsigned int mopr_index = 0;
  unsigned int portals_used = 0;
  if (model._import_type == Model::FROM_WMO14)
  {
    if (opt.value("alpha_portals", false).toBool())
    {
      mopr_index = model.objects_.at(gr)._WMO14_mogp.portals_index;
      portals_used = model.objects_.at(gr)._WMO14_mogp.portals_used;
    }
    if (opt.value("alpha_lights", false).toBool())
    {

    }
    if (opt.value("alpha_doodads", false).toBool())
    {
      if (model.objects_.at(gr)._WMO14_mogp.flags & WMO14::HAS_DOODADS)
        flags |= WMO14::HAS_DOODADS;
    }
    if (opt.value("alpha_colors", false).toBool())
    {
      if (model.objects_.at(gr)._WMO14_mogp.flags & WMO14::HAS_COLORS)
        flags |= WMO14::HAS_COLORS;
    }
    if (opt.value("alpha_liquids", false).toBool())
    {
      if (model.objects_.at(gr)._WMO14_mogp.flags & WMO14::HAS_WATER)
        flags |= WMO14::HAS_WATER;
    }
  }

  std::vector<char> data, temp_data;

  // mogn index - here position in our Group vector +2
  temp_data = bin_io::vc_Int(gr + 2);
  data.insert(data.end(), temp_data.begin(), temp_data.end());

  // descriptive name mogn index - 0 is ok ?
  temp_data = bin_io::vc_Int(0);
  data.insert(data.end(), temp_data.begin(), temp_data.end());

  // flags
  if (opt.value("indoor", false).toBool())
    flags += 0x2000; // indoor
  if (opt.value("outdoor", true).toBool())
    flags += 0x8; // outdoor
  temp_data = bin_io::vc_Int(flags);
  data.insert(data.end(), temp_data.begin(), temp_data.end());

  // bbox - same as MOGI ?
  temp_data = bin_io::vc_Float(model.objects_.at(gr).bbox_.a.x);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Float(model.objects_.at(gr).bbox_.a.y);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Float(model.objects_.at(gr).bbox_.a.z);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Float(model.objects_.at(gr).bbox_.b.x);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Float(model.objects_.at(gr).bbox_.b.y);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Float(model.objects_.at(gr).bbox_.b.z);
  data.insert(data.end(), temp_data.begin(), temp_data.end());

  // mopr index
  temp_data = bin_io::vc_Short(mopr_index);
  data.insert(data.end(), temp_data.begin(), temp_data.end());

  // mopr portals used
  temp_data = bin_io::vc_Short(portals_used);
  data.insert(data.end(), temp_data.begin(), temp_data.end());

  // nBatches - 2 shorts A B and an int C, stores the nb in C
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Int(_nBatches);
  data.insert(data.end(), temp_data.begin(), temp_data.end());

  // indices in WMO fog list
  temp_data = bin_io::vc_Int(0);
  data.insert(data.end(), temp_data.begin(), temp_data.end());

  // liquid type
  temp_data = bin_io::vc_Int(15);
  data.insert(data.end(), temp_data.begin(), temp_data.end());

  // WMO group id
  temp_data = bin_io::vc_Int(0);
  data.insert(data.end(), temp_data.begin(), temp_data.end());

  // padding
  temp_data = bin_io::vc_Int(0);
  data.insert(data.end(), temp_data.begin(), temp_data.end());
  data.insert(data.end(), temp_data.begin(), temp_data.end());

  Chunk c = {"MOGP", getMOGPsize(gr), data};
  // MOGP has to be build at the end, but is the 2nd chunk
  _groups[gr].chunks.insert(_groups.at(gr).chunks.begin() + 1, c);
}
unsigned int WMO_exporter::getMOGPsize(const unsigned int &gr) const
{
  unsigned int that_size = 0;
  for (unsigned int ch = 0; ch < _groups.at(gr).chunks.size(); ++ch)
  {
    that_size += 0x8;
    that_size += _groups.at(gr).chunks.at(ch).data.size();
  }
  that_size -= 0xC; // remove MVER
  that_size += 0x44; // add MOGP header
  return that_size;
}


void WMO_exporter::add_MOPY(const Model &model, const unsigned int &gr)
{
  std::vector<char> data;
  unsigned int nFaces = model.objects_.at(gr).faces_.size();
  for (unsigned int i = 0; i < nFaces; ++i)
  {
    data.push_back(0x20); // flags
    data.push_back(model.objects_.at(gr).faces_.at(i).mat);
  }
  Chunk c = {"MOPY", static_cast<unsigned int>(data.size()), data};
  addChunkToGroup(c, gr);
}


void WMO_exporter::add_MOVI(const Model &model, const unsigned int &gr)
{
  std::vector<char> data, temp_data;
  Face f;
  unsigned int nFaces = model.objects_.at(gr).faces_.size();
  for (unsigned int i = 0; i < nFaces; ++i)
  {
    f = model.objects_.at(gr).faces_.at(i);
    temp_data = bin_io::vc_Short(f.a);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    temp_data = bin_io::vc_Short(f.b);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    temp_data = bin_io::vc_Short(f.c);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
  }
  Chunk c = {"MOVI", static_cast<unsigned int>(data.size()), data};
  addChunkToGroup(c, gr);
}


void WMO_exporter::add_MOVT(const Model &model, const unsigned int &gr)
{
  std::vector<char> data, temp_data;
  Vertex v;

  unsigned int nVertex = model.objects_.at(gr).vertices_.size();
  /*if (model._import_type == Model::FROM_OBJ)
  {
    for (unsigned int i = 0; i < nVertex; ++i)
    {
      v = model.objects_.at(gr).vertices_.at(i);
      temp_data = bin_io::vc_Float(v.x);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(v.y);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(v.z);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
    }
  }
  else
  {*/
    for (unsigned int i = 0; i < nVertex; ++i)
    {
      v = model.objects_.at(gr).vertices_.at(i);
      temp_data = bin_io::vc_Float(v.x);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(v.y);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(v.z);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
    }
  /*}*/

  Chunk c = {"MOVT", static_cast<unsigned int>(data.size()), data};
  addChunkToGroup(c, gr);
}


void WMO_exporter::add_MONR(const Model &model, const unsigned int &gr)
{
  QSettings opt("egamh9", "mirrormachine");
  std::vector<char> data, temp_data;
  Normal n;

  unsigned int nVertex = model.objects_.at(gr).vertices_.size();
  if (model._import_type == Model::FROM_OBJ)
  {
    for (unsigned int i = 0; i < nVertex; ++i)
    {
      n = model.objects_.at(gr).vertices_.at(i).n;
      temp_data = bin_io::vc_Float(n.x);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(n.y);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(n.z);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
    }
  }
  else
  {
    for (unsigned int i = 0; i < nVertex; ++i)
    {
      if (opt.value("create_normals", true).toBool())
      {
        n = model.objects_.at(gr).vertices_.at(i).n;
        temp_data = bin_io::vc_Float(n.x);
        data.insert(data.end(), temp_data.begin(), temp_data.end());
        temp_data = bin_io::vc_Float(n.y);
        data.insert(data.end(), temp_data.begin(), temp_data.end());
        temp_data = bin_io::vc_Float(n.z);
        data.insert(data.end(), temp_data.begin(), temp_data.end());
      }
      else
      {
        temp_data = bin_io::vc_Float(0);
        data.insert(data.end(), temp_data.begin(), temp_data.end());
        data.insert(data.end(), temp_data.begin(), temp_data.end());
        data.insert(data.end(), temp_data.begin(), temp_data.end());
      }
    }
  }

  Chunk c = {"MONR", static_cast<unsigned int>(data.size()), data};
  addChunkToGroup(c, gr);
}


void WMO_exporter::add_MOTV(const Model &model, const unsigned int &gr)
{
  QSettings opt("egamh9", "mirrormachine");
  std::vector<char> data, temp_data;
  UVcoords uv;

  unsigned int nVertex = model.objects_.at(gr).mapping_coords_.size();
  for (unsigned int i = 0; i < nVertex; ++i)
  {
    uv = model.objects_.at(gr).mapping_coords_.at(i);
    temp_data = bin_io::vc_Float(uv.u);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    temp_data = bin_io::vc_Float(uv.v);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
  }

  // this is a dirty workaround to make unmapped models "working"
  // this should be use for DEBUGGING ONLY
  if (nVertex == 0)
  {
    nVertex = model.objects_.at(gr).vertices_.size();
    for (unsigned int i = 0; i < nVertex; ++i)
    {
      temp_data = bin_io::vc_Float(0);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      temp_data = bin_io::vc_Float(1);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
    }
  }

  Chunk c = {"MOTV", static_cast<unsigned int>(data.size()), data};
  addChunkToGroup(c, gr);
}


void WMO_exporter::add_MOBA(const Model &model, const unsigned int &gr)
{
  // creates one batch per material used, according to polygons
  std::vector<char> data, temp_data, bbox;
  temp_data = bin_io::vc_Short(
      static_cast<qint16>(model.objects_.at(gr).bbox_.a.x));
  bbox.insert(bbox.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Short(
      static_cast<qint16>(model.objects_.at(gr).bbox_.a.y));
  bbox.insert(bbox.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Short(
      static_cast<qint16>(model.objects_.at(gr).bbox_.a.z));
  bbox.insert(bbox.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Short(
      static_cast<qint16>(model.objects_.at(gr).bbox_.b.x));
  bbox.insert(bbox.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Short(
      static_cast<qint16>(model.objects_.at(gr).bbox_.b.y));
  bbox.insert(bbox.end(), temp_data.begin(), temp_data.end());
  temp_data = bin_io::vc_Short(
      static_cast<qint16>(model.objects_.at(gr).bbox_.b.z));
  bbox.insert(bbox.end(), temp_data.begin(), temp_data.end());

  Face f;

  unsigned int current_face_count = 0, full_face_count = 0;
  unsigned int vertex_count = 0;
  quint8 current_mat = model.objects_.at(gr).faces_.at(0).mat;

  // to remember how many vertices we parsed
  // stores it in a bool vector, representing the vertices indices
  const unsigned int nVertices = model.objects_.at(gr).vertices_.size();
  std::vector<bool> parsed_vertices(nVertices, false);

  const unsigned int nFaces = model.objects_.at(gr).faces_.size();
  for (unsigned int id_f = 0; id_f < nFaces; ++id_f)
  {
    f = model.objects_.at(gr).faces_.at(id_f);

    if ((f.mat != current_mat) ||
        (f.mat == 0xFF) ||
        ((id_f+1) == nFaces))
    {
      // writes a new batch when all the faces with the same
      // material (!0xFF) has been parsed
      // (or if it's the last face)

      if ((id_f+1) == nFaces)
      {
        parsed_vertices.at(f.a) = true;
        parsed_vertices.at(f.b) = true;
        parsed_vertices.at(f.c) = true;
        current_face_count += 3;
      }

      // short bounding box - lazy way, same bb for every batch
      data.insert(data.end(), bbox.begin(), bbox.end());
      // first index in face list
      temp_data = bin_io::vc_Int(full_face_count);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      // nb of indices read in face list
      temp_data = bin_io::vc_Short(current_face_count);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      // first index in vertex list
      temp_data = bin_io::vc_Short(0);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      // last index in vertex list
      temp_data = bin_io::vc_Short(nVertices - 1);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
      // unknown - 0
      data.push_back(0x00);
      // material id
      data.push_back(current_mat);

      full_face_count += current_face_count;
      current_face_count = 0;
      ++vertex_count;
      current_mat = f.mat;
    }

    // sets the face vertices as parsed
    parsed_vertices.at(f.a) = true;
    parsed_vertices.at(f.b) = true;
    parsed_vertices.at(f.c) = true;

    // the face_count goes +3 because it's the index in MOVI
    // and a face is 3 MOVI indices
    current_face_count += 3;

    // ends the loop if it reaches collision-only polygons
    if (f.mat == 0xFF)
      break;
  }

  // a MOBA entry takes 24 bytes
  _nBatches = data.size() / 24;

  Chunk c = {"MOBA", static_cast<unsigned int>(data.size()), data};
  addChunkToGroup(c, gr);
}



void WMO_exporter::add_MOLR(const Model& /*model*/, const unsigned int& /*gr*/)
{

}


void WMO_exporter::add_MODR(const Model &model, const unsigned int &gr)
{
  std::vector<char> data, temp_data;

  const unsigned int nDefs = model.objects_.at(gr)._WMO14_modr.size();
  for (unsigned int i = 0; i < nDefs; ++i)
  {
    temp_data = bin_io::vc_Short(model.objects_.at(gr)._WMO14_modr.at(i));
    data.insert(data.end(), temp_data.begin(), temp_data.end());
  }

  Chunk c = {"MODR", static_cast<unsigned int>(data.size()), data};
  addChunkToGroup(c, gr);
}




void WMO_exporter::add_MOBN(const Model &model, const unsigned int &gr)
{
  // adds a dummy entry
  QSettings opt("egamh9", "mirrormachine");

  std::vector<char> data, temp_data;

  if (opt.value("generate_bsp", true).toBool())
  {
    unsigned int face_counter = 0;
    const unsigned int num_nodes = model.objects_.at(gr).bsp_nodes_.size();
    const std::vector<BspNode>& nodes = model.objects_.at(gr).bsp_nodes_;
    for (unsigned int i = 0; i < num_nodes; ++i)
    {
      const BspNode& node = nodes.at(i);
      if (node.plane_type == BspNode::LEAF)
      {
        // planetype
        temp_data = bin_io::vc_Short(4);
        data.insert(data.end(), temp_data.begin(), temp_data.end());
        // children (nope)
        temp_data = bin_io::vc_Short(0xFFFF);
        data.insert(data.end(), temp_data.begin(), temp_data.end());
        data.insert(data.end(), temp_data.begin(), temp_data.end());
        // nb of faces
        temp_data = bin_io::vc_Short(node.refs.size());
        data.insert(data.end(), temp_data.begin(), temp_data.end());
        // first face index
        temp_data = bin_io::vc_Int(face_counter);
        data.insert(data.end(), temp_data.begin(), temp_data.end());
        // fDist
        temp_data = bin_io::vc_Float(0);
        data.insert(data.end(), temp_data.begin(), temp_data.end());

        face_counter += node.refs.size();
      }
      else
      {
        // planetype
        switch (node.plane_type)
        {
          case BspNode::YZ_PLANE: temp_data = bin_io::vc_Short(0); break;
          case BspNode::XZ_PLANE: temp_data = bin_io::vc_Short(1); break;
          case BspNode::XY_PLANE: temp_data = bin_io::vc_Short(2); break;
          default: break;
        }
        data.insert(data.end(), temp_data.begin(), temp_data.end());
        // children
        // holy shit did i really wrote that
        if (node.child1 != NULL)
        {
          unsigned long long child1_addr =
              reinterpret_cast<unsigned long long>(node.child1);
          unsigned int child1_index = (child1_addr -
              reinterpret_cast<unsigned long long>(
                  model.objects_.at(gr).bsp_nodes_.data())) / sizeof(BspNode);
          temp_data = bin_io::vc_Short(child1_index);

        }
        else
        {
          temp_data = bin_io::vc_Short(0xFFFF);
        }
        data.insert(data.end(), temp_data.begin(), temp_data.end());

        if (node.child2 != NULL)
        {
          unsigned long long child2_addr =
              reinterpret_cast<unsigned long long>(node.child2);
          unsigned int child2_index = (child2_addr -
              reinterpret_cast<unsigned long long>(
                  model.objects_.at(gr).bsp_nodes_.data())) / sizeof(BspNode);
          temp_data = bin_io::vc_Short(child2_index);
        }
        else
        {
          temp_data = bin_io::vc_Short(0xFFFF);
        }
        data.insert(data.end(), temp_data.begin(), temp_data.end());
        // nb of faces
        temp_data = bin_io::vc_Short(0);
        data.insert(data.end(), temp_data.begin(), temp_data.end());
        // first face index
        temp_data = bin_io::vc_Int(0);
        data.insert(data.end(), temp_data.begin(), temp_data.end());
        // fDist
        temp_data = bin_io::vc_Float(node.f_dist);
        data.insert(data.end(), temp_data.begin(), temp_data.end());
      }

    }



    /*add_MOBN_WriteBspNode(model, gr, model.objects_.at(gr).bsp_nodes_.at(0),
                          &data, &face_counter);*/
  }

  else
  {
    // planetype (4 is a leaf)
    temp_data = bin_io::vc_Short(4);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    // childrens (nope)
    temp_data = bin_io::vc_Short(0xFFFF);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    // nb of faces
    if(opt.value("add_collisions", true).toBool())
      temp_data = bin_io::vc_Short(model.objects_.at(gr).faces_.size());
    else
      temp_data = bin_io::vc_Short(0);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    // first face index
    temp_data = bin_io::vc_Int(0);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
    // fDist
    temp_data = bin_io::vc_Float(0);
    data.insert(data.end(), temp_data.begin(), temp_data.end());
  }

  Chunk c = {"MOBN", static_cast<unsigned int>(data.size()), data};
  addChunkToGroup(c, gr);
}

/*
void WMO_exporter::add_MOBN_WriteBspNode(const Model& model,
                                         const unsigned int &gr,
                                         const BspNode& node,
                                         std::vector<char>* data,
                                         unsigned int* face_counter)
{
  std::vector<char> temp_data;
  if (node.plane_type == BspNode::LEAF)
  {
    // planetype
    temp_data = bin_io::vc_Short(4);
    data->insert(data->end(), temp_data.begin(), temp_data.end());
    // children (nope)
    temp_data = bin_io::vc_Short(0xFFFF);
    data->insert(data->end(), temp_data.begin(), temp_data.end());
    data->insert(data->end(), temp_data.begin(), temp_data.end());
    // nb of faces
    temp_data = bin_io::vc_Short(node.refs.size());
    data->insert(data->end(), temp_data.begin(), temp_data.end());
    // first face index
    temp_data = bin_io::vc_Int(*face_counter);
    data->insert(data->end(), temp_data.begin(), temp_data.end());
    // fDist
    temp_data = bin_io::vc_Float(0);
    data->insert(data->end(), temp_data.begin(), temp_data.end());

    *face_counter += node.refs.size();
  }
  else
  {
    // planetype
    switch (node.plane_type)
    {
      case BspNode::YZ_PLANE: temp_data = bin_io::vc_Short(0); break;
      case BspNode::XZ_PLANE: temp_data = bin_io::vc_Short(1); break;
      case BspNode::XY_PLANE: temp_data = bin_io::vc_Short(2); break;
      default: break;
    }
    data->insert(data->end(), temp_data.begin(), temp_data.end());
    // children
    // holy shit did i really wrote that
    unsigned long long child1_addr =
        reinterpret_cast<unsigned long long>(node.child1);
    unsigned long long child2_addr =
        reinterpret_cast<unsigned long long>(node.child2);
    unsigned int child1_index = (child1_addr -
        reinterpret_cast<unsigned long long>(
            model.objects_.at(gr).bsp_nodes_.data())) / sizeof(BspNode);
    unsigned int child2_index = (child2_addr -
        reinterpret_cast<unsigned long long>(
            model.objects_.at(gr).bsp_nodes_.data())) / sizeof(BspNode);
    temp_data = bin_io::vc_Short(child1_index);
    data->insert(data->end(), temp_data.begin(), temp_data.end());
    temp_data = bin_io::vc_Short(child2_index);
    data->insert(data->end(), temp_data.begin(), temp_data.end());
    // nb of faces
    temp_data = bin_io::vc_Short(0);
    data->insert(data->end(), temp_data.begin(), temp_data.end());
    // first face index
    temp_data = bin_io::vc_Int(0);
    data->insert(data->end(), temp_data.begin(), temp_data.end());
    // fDist
    temp_data = bin_io::vc_Float(0);
    data->insert(data->end(), temp_data.begin(), temp_data.end());

    add_MOBN_WriteBspNode(model, gr, *node.child1, data, face_counter);
    add_MOBN_WriteBspNode(model, gr, *node.child2, data, face_counter);

  }
}
*/


void WMO_exporter::add_MOBR(const Model &model, const unsigned int &gr)
{
  // include every poly in the list
  QSettings opt("egamh9", "mirrormachine");

  std::vector<char> data, temp_data;

  if (opt.value("generate_bsp", true).toBool())
  {
    const std::vector<BspRef>& refs = model.objects_.at(gr).bsp_refs_;
    for (unsigned short f = 0; f < refs.size(); ++f)
    {
      temp_data = bin_io::vc_Short(refs.at(f));
      data.insert(data.end(), temp_data.begin(), temp_data.end());
    }
  }

  else if (opt.value("add_collisions", true).toBool())
  {
    const unsigned int nFace = model.objects_.at(gr).faces_.size();
    for (unsigned short f = 0; f < nFace; ++f)
    {
      temp_data = bin_io::vc_Short(f);
      data.insert(data.end(), temp_data.begin(), temp_data.end());
    }
  }

  Chunk c = {"MOBR", static_cast<unsigned int>(data.size()), data};
  addChunkToGroup(c, gr);
}


void WMO_exporter::add_MOCV(const Model &model, const unsigned int &gr)
{
  std::vector<char> data, temp_data;

  const unsigned int nColors = model.objects_.at(gr)._WMO14_mocv.size();
  for (unsigned int i = 0; i < nColors; ++i)
  {
    temp_data = bin_io::vc_Int(model.objects_.at(gr)._WMO14_mocv.at(i));
    data.insert(data.end(), temp_data.begin(), temp_data.end());
  }

  Chunk c = {"MOCV", static_cast<unsigned int>(data.size()), data};
  addChunkToGroup(c, gr);
}


void WMO_exporter::add_MLIQ(const Model &model, const unsigned int &gr)
{
  Chunk c = {"MLIQ",
             static_cast<unsigned int>(
                 model.objects_.at(gr)._WMO14_mliq.size()),
             model.objects_.at(gr)._WMO14_mliq};
  addChunkToGroup(c, gr);
}







// write the file

std::string WMO_exporter::createGroupPath(const char *path,
                                          const unsigned int &gr)
{
  using std::ostringstream;

  std::string group_path(path);
  group_path.append("_");

  // stores the group_index under a string shape with awesome C++ stuff
  std::string index_string =
      static_cast<ostringstream*>(&(ostringstream() << gr))->str();

  if (gr < 10)
    group_path.append("00");
  else if (gr < 100)
    group_path.append("0");

  group_path.append(index_string);
  group_path.append(".wmo");

  return group_path;
}

void WMO_exporter::chunks_to_bytes(const std::vector<Chunk> &chunks,
                          std::vector<char> &output)
{
  std::vector<char> temp_data;

  const unsigned int nChunks = chunks.size();
  for (unsigned int i = 0; i < nChunks; ++i)
  {
    // ident
    for (int j = 3; j >= 0; --j) // reads the 4-letter from the end
      output.push_back(chunks.at(i).ident.at(j));
    // size
    temp_data = bin_io::vc_Int(chunks.at(i).given_size);
    output.insert(output.end(), temp_data.begin(), temp_data.end());
    // data
    output.insert(output.end(),
                  chunks.at(i).data.begin(),
                  chunks.at(i).data.end());
  }
}

int WMO_exporter::write_root(const char *path)
{
  std::string complete_path(path);
  complete_path.append(".wmo");

  std::ofstream ofile(complete_path.c_str(), std::ios::binary);
  if (!ofile.is_open())
    return 1;

  std::vector<char> chunks_bytes;
  chunks_to_bytes(_chunks, chunks_bytes);
  ofile.write((char*) &chunks_bytes[0], chunks_bytes.size());
  ofile.close();
  return 0;
}

int WMO_exporter::write_groups(const char *path)
{
  std::string group_path;
  std::vector<char> chunks_bytes;
  std::ofstream ofile;

  const unsigned int nGroups = _groups.size();
  for (unsigned int i = 0; i < nGroups; ++i)
  {
    group_path = createGroupPath(path, i);
    ofile.open(group_path.c_str(), std::ios::binary);
    if (!ofile.is_open())
      return 1;

    chunks_bytes.clear();
    chunks_to_bytes(_groups.at(i).chunks, chunks_bytes);
    ofile.write((char*) &chunks_bytes[0], chunks_bytes.size());
    ofile.close();
  }
  return 0;
}





/*void WMO_exporter::GiveTharoTheData(const Model& model) const
{
  std::ofstream out_file("vertices.txt");
  if (!out_file.is_open())
  {
    std::cerr << "Can't open vertices.txt to output text data" << std::endl;
    return;
  }

  for (size_t i = 0; i < model.objects_.size(); ++i)
  {
    out_file << "o " << i << std::endl;
    const ModelObject& object = model.objects_.at(i);
    for (size_t j = 0; j < object.vertices_.size(); ++j)
    {
      out_file << "v " << object.vertices_.at(j).x << " "
                       << object.vertices_.at(j).y << " "
                       << object.vertices_.at(j).z << std::endl;
    }
    for (size_t j = 0; j < object.faces_.size(); ++j)
    {
      out_file << "f " << object.faces_.at(j).a << " "
                       << object.faces_.at(j).b << " "
                       << object.faces_.at(j).c << std::endl;
    }
  }

  out_file.close();
}*/
