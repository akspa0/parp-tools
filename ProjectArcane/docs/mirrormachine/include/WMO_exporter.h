#ifndef WMO_H
#define WMO_H

#include <string>
#include <vector>


// chunk of data according to WoW spec (4-byte name, uint32 data-size and data)
struct Chunk
{
  std::string ident;
  quint32 given_size;
  std::vector<char> data;
};

// represents a group file with its list of chunks (poor design)
struct Group
{
  std::vector<Chunk> chunks;
};


// A WMO_exporter is a class computing WMO files and writing them
class WMO_exporter
{

public:

  // empty constructor
  WMO_exporter();

  // compute the WMO chunks and files from the given Model
  // using the private members of the class
  int process(const Model &model, const char *path);

private:

  // stores the chunks of the root file
  std::vector<Chunk> _chunks;
  // stores the Groups (chunks) for each group file
  std::vector<Group> _groups;

  // stores the nb of batches after computing a good number
  unsigned int _nBatches;


  // add a chunk to root/group vector<Chunk>
  void addChunk(const Chunk &ck);
  void addChunkToGroup(const Chunk &ck, const unsigned int &gr);

  // create the root file chunks
  void add_MVER(const unsigned int &version);
  void add_MOHD(const Model &model);
  void add_MOTX(const Model &model);
  void add_MOMT(const Model &model);
  void add_MOGN(const Model &model);
  void add_MOGI(const Model &model);
  void add_MOSB(const Model &/*model*/);
  void add_MOPV(const Model &model);
  void add_MOPT(const Model &model);
  void add_MOPR(const Model &model);
  void add_MOVV(const Model &/*model*/);
  void add_MOVB(const Model &/*model*/);
  void add_MOLT(const Model &/*model*/);
  void add_MODS(const Model &model);
  void add_MODN(const Model &model);
  void add_MODD(const Model &model);
  void add_MFOG(const Model &/*model*/);

  // create the group file chunks
  void add_MVER(const unsigned int &version, const unsigned int &gr);
  void add_MOGP(const Model &model, const unsigned int &gr);
  void add_MOPY(const Model &model, const unsigned int &gr);
  void add_MOVI(const Model &model, const unsigned int &gr);
  void add_MOVT(const Model &model, const unsigned int &gr);
  void add_MONR(const Model &model, const unsigned int &gr);
  void add_MOTV(const Model &model, const unsigned int &gr);
  void add_MOBA(const Model &model, const unsigned int &gr);
  void add_MOLR(const Model& /*model*/, const unsigned int& /*gr*/);
  void add_MODR(const Model &model, const unsigned int &gr);
  void add_MOBN(const Model &/*model*/, const unsigned int &gr);
  void add_MOBR(const Model &/*model*/, const unsigned int &gr);
  void add_MOCV(const Model &model, const unsigned int &gr);
  void add_MLIQ(const Model &/*model*/, const unsigned int &gr);

  // Writes a given BSP node in the *data.
  // It is recursive and works and the ModelObject.bsp_nodes and refs.
  void add_MOBN_WriteBspNode(const Model& model,
                             const unsigned int &gr,
                             const BspNode& node,
                             std::vector<char>* data,
                             unsigned int* face_counter);

  // calculate the MOGP size for an object
  // useful as MOGP size can't be known before we computed every other chunk
  unsigned int getMOGPsize(const unsigned int &gr) const;

  // create a group path (like "_04" for group 4)
  std::string createGroupPath(const char *path, const unsigned int &gr);

  // time to write down our files !
  // writes the chunks bytes in the output vector
  void chunks_to_bytes(const std::vector<Chunk> &chunks,
                       std::vector<char> &output);
  // write the root file
  int write_root(const char *path);
  // write the group files
  int write_groups(const char *path);

  // for Tharo's plans on doing a BSP generator
  // be brave !
  /*void GiveTharoTheData(const Model& model) const;*/
};


#endif
