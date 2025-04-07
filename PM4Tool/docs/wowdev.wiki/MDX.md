{{Template:SectionBox/VersionRange|max_expansionlevel=1|max_exclusive=1}}
[[MDX|MDX]] files are [[chunk|chunked]] binary files that contain model objects. They are the predecessor of the [[M2]] format.

First used in Warcraft 3, [[MDX|MDX]] was actively developed and used in WoW as the primary model format until patch {{Template:Sandbox/PrettyVersion|expansionlevel=0|build=0.11.0.3925}}. Although obsolete, some [[DBC|DBC]]s still contain filenames with a <tt>.mdx</tt> extension.

<b>Note:</b> The majority of the below information has been taken from the {{Template:Sandbox/PrettyVersion|expansionlevel=0|build=0.5.3.3368}} client and is only truly compliant for version 1300 of the format.

__TOC__

==Structure==
The complete structure for a [[MDX|MDX]] file. <b>Note:</b> Chunks after <tt>MODL</tt> don't have to conform to a specific order and can be unimplemented on a per-file basis - particularly the <tt>K***</tt> sub-chunks.
 struct MDLBASE
 {
   char magic[4];           // MDLX
   [[#VERS|VERS]] version;
   [[#MODL|MODL]] model; 
   [[#SEQS|SEQS]] sequences;
   [[#GLBS|GLBS]] globalSeqs;
   [[#MTLS|MTLS]] materials;
   [[#TEXS|TEXS]] textures;
   [[#TXAN|TXAN]] textureanims;
   [[#GEOS|GEOS]] geosets;
   [[#GEOA|GEOA]] geosetAnims;
   [[#BONE|BONE]] bones;
   [[#LITE|LITE]] lights;
   [[#HELP|HELP]] helpers;
   [[#ATCH|ATCH]] attachments;
   [[#PIVT|PIVT]] pivotPoints;
   [[#PREM|PREM]] particleEmitters;
   [[#CAMS|CAMS]] cameras;
   [[#EVTS|EVTS]] events;
   [[#PRE2|PRE2]] particleEmitters2;
   [[#CORN|CORN]] popcornFxEmitters;
   [[#HTST|HTST]] hitTestShapes;
   [[#RIBB|RIBB]] ribbonEmitters;
   [[#CLID|CLID]] collision;
 };
 

==Common Types==

===C3Color===
 struct C3Color
 {
   float b;
   float g;
   float r;
 };

===C4QuaternionCompressed===
For the WoW variant of [[MDX|MDX]], all {{Template:Type|C4Quaternion}}s are packed in int64_ts.
<syntaxhighlight lang="cpp">
struct C4QuaternionCompressed
{
  int64_t m_data;
 
  C4Quaternion Get()
  {
     C4Quaternion result;	  
     result.X = ( m_data >> 42 ) * 0.00000047683716;
     result.Y = (( m_data << 22 ) >> 43 ) * 0.00000095367432;
     result.Z = ((int32_t)( m_data << 11 ) >> 11) * 0.00000095367432;
     result.W = GetW( result.X, result.Y, result.Z );      
     return result;
  }
  
  double GetW(float x, float y, float z)
  {
     double len = 1.0 - (x * x + y * y + z * z);      
     if ( len >= 0.00000095367432 )
        return sqrt( len );
     
     return 0.0;
  }
  
  void Set(C4Quaternion source)
  {
     int32_t sign = ( source->W >= 0.0 ? 1 : -1 );            
     int64_t x = (int64_t)( sign * source->X * 2097152.0 ) << 21;
     int64_t y = (int64_t)( sign * source->Y * 1048576.0 ) & 0x1FFFFF;
     int64_t z = (int64_t)( sign * source->Z * 1048576.0 ) & 0x1FFFFF;
     
     m_data = z | (( y | x ) << 21 );
  }    
};
</syntaxhighlight>

===CMdlBounds===
 struct CMdlBounds
 {
   {{Template:Type|CAaBox}} extent;
   float radius;
 };

===MDLKEYTRACK===
The <tt>MDLKEYTRACK</tt> is a simpler equivalent of the [[M2#Types|M2Track]]. It stores a list of <tt>MDLKEYFRAMES</tt> which are essentially tuples of a time and T type values.
If the type is <tt>TRACK_HERMITE</tt> or <tt>TRACK_BEZIER</tt> then the frame also contains <tt>inTan</tt> and <tt>outTan</tt> information. See the relevant [[M2#Interpolation|M2 interpolation section]] for more information.

Under certain conditions the client overrides the track type e.g. when <tt>MDLMODELSECTION.flags & 4</tt> (always animate) is set <tt>TRACK_LINEAR</tt> is used.

 template<typename T>
 struct MDLKEYTRACK<T>
 {
   uint32_t count;
   MDLTRACKTYPE type;
   uint32_t globalSeqId;        // [[#GLBS|GLBS]] index or 0xFFFFFFFF if none
   MDLKEYFRAME<T> keys[count];  
 };
 
 template<typename T>
 struct MDLKEYFRAME<T>
 {
   int32_t time;
   T value;
  #if MDLKEYTRACK.type > TRACK_LINEAR
   T inTan;
   T outTan;
  #endif
 };
 
 enum MDLTRACKTYPE : uint32_t
 {
   TRACK_NO_INTERP = 0x0,
   TRACK_LINEAR = 0x1,
   TRACK_HERMITE = 0x2,
   TRACK_BEZIER = 0x3,
   NUM_TRACK_TYPES = 0x4,
 };

===MDLSIMPLEKEYTRACK===
The <tt>MDLSIMPLEKEYTRACK</tt> is used in place of the <tt>MDLKEYTRACK</tt> when only linear integer values are required. Types used by this track are <tt>MDLINTKEY</tt> and <tt>MDLEVENTKEY</tt>.
 template<typename T>
 struct MDLSIMPLEKEYTRACK<T>
 {
   uint32_t count;
   uint32_t globalSeqId;  // [[#GLBS|GLBS]] index or 0xFFFFFFFF if none
   T keys[count];
 };
 
 struct MDLINTKEY         // default type
 {
   uint32_t time;
   uint32_t value;
 };
 
 struct MDLEVENTKEY       // only used for the [[#EVTS|EVTS]] [[#KEVT|KEVT]] sub-chunk
 {
   int32_t time;
 };

===MDLGENOBJECT===
<tt>MDLGENOBJECT</tt> is a base class inherited by several chunks. This is not just for common data but is also used to build an object hierarchy.

The hierarchy is usually organised as: <code>Bones (root bones first) → Lights → Helpers → Attachments → ParticleEmitters → RibbonEmitters → Events → HitTestShapes</code>. The client will throw an exception if the objectIds are not sequential.
 struct MDLGENOBJECT
 {
   uint32_t size;
   char name[0x50];
   uint32_t objectId; // globally unique id, used as the index in the hierarchy. index into [[#PIVT|PIVT]]
   uint32_t parentId; // parent MDLGENOBJECT's objectId or 0xFFFFFFFF if none
   uint32_t flags;
   
   [[#KGTR|KGTR]] transkeys;
   [[#KGRT|KGRT]] rotkeys;
   [[#KGSC|KGSC]] scalekeys;
 };

====Flags====
<b>Notes:</b> Certain flag combinations are invalid and will throw exceptions. Flags ≥ 0x20000 are only applicable to [[#PRE2|PRE2]] objects. GENOBJECT flags are also set in the class constructor.
{| class="wikitable"
|-
! width="50" | Flag
! width="450" | Meaning
! width="650" | Notes
|-
| 0x00000001 || DONT_INHERIT_TRANSLATION || 
|- || || 
| 0x00000002 || DONT_INHERIT_SCALING || 
|- || || 
| 0x00000004 || DONT_INHERIT_ROTATION || 
|- || || 
| 0x00000008 || BILLBOARD || 
|- || || 
| 0x00000010 || BILLBOARD_LOCK_X || 
|- || || 
| 0x00000020 || BILLBOARD_LOCK_Y || 
|- || || 
| 0x00000040 || BILLBOARD_LOCK_Z || 
|- || || 
| 0x00000080 || GENOBJECT_MDLBONESECTION || not explicitly set in the files however all other GENOBJECT flags are
|- || || 
| 0x00000100 || GENOBJECT_MDLLIGHTSECTION || 
|- || || 
| 0x00000200 || GENOBJECT_MDLEVENTSECTION || 
|- || || 
| 0x00000400 || GENOBJECT_MDLATTACHMENTSECTION || 
|- || || 
| 0x00000800 || GENOBJECT_MDLPARTICLEEMITTER2 || 
|- || || 
| 0x00001000 || GENOBJECT_MDLHITTESTSHAPE || 
|- || || 
| 0x00002000 || GENOBJECT_MDLRIBBONEMITTER || 
|- || || 
| 0x00004000 || PROJECT || 
|- || || 
| 0x00008000 || EMITTER_USES_TGA ([[#PREM|PREM]]), UNSHADED ([[#PRE2|PRE2]]) || UNSHADED disables lighting on [[M2/Rendering#CParticleMat|particle materials]]
|- || || 
| 0x00010000 || EMITTER_USES_MDL ([[#PREM|PREM]]), SORT_PRIMITIVES_FAR_Z ([[#PRE2|PRE2]]) || 
|- || || 
| 0x00020000 || LINE_EMITTER || 
|- || || 
| 0x00040000 || PARTICLE_UNFOGGED || disables fog on [[M2/Rendering#CParticleMat|particle materials]]
|- || || 
| 0x00080000 || PARTICLE_USE_MODEL_SPACE || uses model space instead of world space
|- || || 
| 0x00100000 || PARTICLE_INHERIT_SCALE || 
|- || || 
| 0x00200000 || PARTICLE_INSTANT_VELOCITY_LINEAR || a cumulative velocity is applied to new particles
|- || || 
| 0x00400000 || PARTICLE_0XKILL || particles are destroyed after their first update tick
|- || || 
| 0x00800000 || PARTICLE_Z_VELOCITY_ONLY || particle X and Y velocities are set to 0.0 at instantiation
|- || || 
| 0x01000000 || PARTICLE_TUMBLER || 50% chance to invert each particle position component at instantiation (PET_BASE)
|- || || 
| 0x02000000 || PARTICLE_TAIL_GROWS || 
|- || || 
| 0x04000000 || PARTICLE_EXTRUDE || extrudes between the previous and current translation
|- || || 
| 0x08000000 || PARTICLE_XYQUADS || particles align to the XY axis facing the Z axis
|- || || 
| 0x10000000 || PARTICLE_PROJECT || 
|- || || 
| 0x20000000 || PARTICLE_FOLLOW || particles follow each other
|- || || 
|}

====KGTR====
Geoset translation track
 struct KGTR
 {
   char tag[4]; // KGTR
   [[#MDLKEYTRACK|MDLKEYTRACK]]<{{Template:Type|C3Vector}}> transkeys;
 };
====KGRT====
Geoset rotation track
 struct KGRT
 {
   char tag[4]; // KGRT
   [[#MDLKEYTRACK|MDLKEYTRACK]]<[[#C4QuaternionCompressed|C4QuaternionCompressed]]> rotkeys;
 };
====KGSC====
Geoset scale track
 struct KGSC
 {
   char tag[4]; // KGSC
   [[#MDLKEYTRACK|MDLKEYTRACK]]<{{Template:Type|C3Vector}}> scalekeys;
 };

==VERS==
Version. Equivalent to the <tt>MVER</tt> chunk.

File analysis of v1400 shows no structural differences to v1300, the only apparent change is that referenced file paths are now normalized.

v1500 sees two structural changes from the previous iterations namely; new flags in the [[#MTLS|MTLS]] chunk and a complete redesign of the [[#GEOS_.28v1500.29|GEOS]] chunk.

The WC3 and WC3 Reforged structure is documented [https://www.hiveworkshop.com/threads/mdx-specifications.240487 here].

 uint32_t version; // 800 WC3, 900, 1000 WC3 Reforged, 1300 {{Template:Sandbox/VersionRange|max_expansionlevel=0|max_build=0.9.1.3810|max_exclusive=1}}, 1400 & 1500 {{Template:Sandbox/VersionRange|min_expansionlevel=0|min_build=0.9.1.3810}}

==MODL==
Global model information.
 struct MDLMODELSECTION
 {
   char name[0x50];
   char animationFile[0x104];  // always 0 filled
   [[#CMdlBounds|CMdlBounds]] bounds;          // for reforged: seems to be the radius first then the box 
   uint32_t blendTime;
   #if !WC3Reforged            // Reforged doesn't have those flags
      uint8_t flags;           // deprecated, always 0. &1, 2: GROUND_TRACK, &4: always animate
   #endif
 };
 
 enum GROUND_TRACK
 {
   TRACK_YAW_ONLY = 0x0,
   TRACK_PITCH_YAW = 0x1,
   TRACK_PITCH_YAW_ROLL = 0x2,
   GROUND_TRACK_MASK = 0x3,
 };

==SEQS==
Sequences. [[MDX|MDX]] uses a single track for all animations meaning start times and end times between each animation are consecutive.
 struct SEQS
 {
    uint32_t numSeqs;   // limited to 0xFF
    MDLSEQUENCESSECTION sequences [numSeqs];
 };
  
 struct MDLSEQUENCESSECTION
 {
   char name[0x50];
   {{Template:Type|CiRange}} time;       // start time, end time
   float movespeed;     // movement speed of the entity while playing this animation
   uint32_t flags;      // &1: non looping
   #if WC3 Reforged
      float rarity;
      int syncPoint;    // probably for syncing audio with the animation, only seen 0 so far
   #endif
   [[#CMdlBounds|CMdlBounds]] bounds;
   #if !WC3 Reforged
      float frequency;  // determines chance of this animation playing. for all animations of the same type this must add to 1.0
      {{Template:Type|CiRange}} replay;     // the client will pick a random number of repetitions within bounds
      uint32_t blendTime;
   #endif
 };

==GLBS==
Maximum lengths for sequence ranges. This chunk has no count, the client reads uint32_ts until chunk.size bytes have been read.
 struct MDLGLOBALSEQSECTION
 {
   uint32_t length[chunk.size / 0x4];
 };

==MTLS==
Materials.
 struct MTLS
 {
   uint32_t numMaterials;    // limited to 0xFF
   uint32_t unused;          // has values but is ignored by the client
   MDLMATERIALSECTION materials[numMaterials];
 };
 
 struct MDLMATERIALSECTION
 {
   uint32_t size;
   int32_t priorityPlane;    // priority is sorted lowest to highest
   uint32_t numLayers;	     
   MDLTEXLAYER texLayers[numLayers];	
 };
 
 struct MDLTEXLAYER
 { 
   uint32_t size;
   MDLTEXOP blendMode;
   MDLGEO flags;
   uint32_t textureId;        // [[#TEXS|TEXS]] index or 0xFFFFFFFF for none
   uint32_t transformId;      // [[#TXAN|TXAN]] index or 0xFFFFFFFF for none
   int32_t coordId;           // [[#UAVS|UAVS]] index or -1 for none, defines vertex buffer format <code>coordId == -1 ? GxVBF_PN : GxVBF_PNT0</code>
   float staticAlpha;         // 0 for transparent, 1 for opaque
   
   [[#KMTA|KMTA]] alphaKeys;
   [[#KMTF|KMTF]] flipKeys;
 };
 
 enum MDLTEXOP : uint32_t
 {
   TEXOP_LOAD = 0x0,
   TEXOP_TRANSPARENT = 0x1,
   TEXOP_BLEND = 0x2,
   TEXOP_ADD = 0x3,
   TEXOP_ADD_ALPHA = 0x4,
   TEXOP_MODULATE = 0x5,
   TEXOP_MODULATE2X = 0x6,
   NUMTEXOPS = 0x7,
 };
 
 enum MDLGEO : uint32_t
 {
   MODEL_GEO_UNSHADED = 0x1,
   MODEL_GEO_SPHERE_ENV_MAP = 0x2,  // unused until v1500
   MODEL_GEO_WRAPWIDTH = 0x4,       // unused until v1500
   MODEL_GEO_WRAPHEIGHT = 0x8,      // unused until v1500
   MODEL_GEO_TWOSIDED = 0x10,
   MODEL_GEO_UNFOGGED = 0x20,
   MODEL_GEO_NO_DEPTH_TEST = 0x40,
   MODEL_GEO_NO_DEPTH_SET = 0x80,
   MODEL_GEO_NO_FALLBACK = 0x100,   // added in v1500. seen in <tt>ElwynnTallWaterfall01.mdx</tt>, <tt>FelwoodTallWaterfall01.mdx</tt> and <tt>LavaFallsBlackRock*.mdx</tt>
 };

===MTLS(Reforged)===
 In reforged we don't know the number of materials, and sizeof(MTLS) isn't constant so read till end of the chunk.

 struct {
   SHADERREF shaderRef;        // reference to the shader
   LAYS lays;                  // layers
 } MTLS;

 struct {
   int32 sizeOfLayer;        // including the int
   int32 priorityPlane;
   int32 flags;
   char shaderName[80];    // shader name
 } SHADERREF;

 // Layers
 // 0 - Diffuse
 // 1 - Normal Map
 // 2 - ORM = Occlusion, Roughness, Metalic (RGB)
 // 3 - Emissive
 // 4 - Team Color
 // 5 - Environment Map
 struct {
    char LAYS[4];
    int32 numberOfLAYS;
    TEXTURELAYER textureLayers[numberOfLAYS]
 } LAYS;

 struct {
   int32 textureLayerSize;   // including this int
   MDLTEXOP blendMode;
   MDLGEO shadingFlags;
   int32 textureID;
   int32 textureAnimationID;
   int32 coordID;
   float alpha;
   float emissiveGain;
   #if version == 1000
        float unk1[3];
        int32 unk2[2];       // 0 filled
   #endif
   [[#KMTE|KMTE]] emissiveKeys;
   [[#KMTA|KMTA]] alphaKeys;
   [[#KMTF|KMTF]] textureLayerKeys;
 } TEXTURELAYER;

===KMTE===
Material alpha track
 struct KMTE
 {
   char tag[4]; // KMTE
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> emissiveKeys;
 };

===KMTA===
Material alpha track
 struct KMTA
 {
   char tag[4]; // KMTA
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> alphaKeys;
 };

===KMTF===
Material flipbook texture track
 struct KMTF
 {
   char tag[4]; // KMTF
   [[#MDLSIMPLEKEYTRACK|MDLSIMPLEKEYTRACK]]<MDLINTKEY> flipKeys;
 };

==TEXS==
Textures. The client reads <tt>MDLTEXTURESECTION</tt>s until chunk.size bytes have been read.
 struct TEXS
 {
   MDLTEXTURESECTION textures[chunk.size / sizeof(MDLTEXTURESECTION)];
 };
 
 struct MDLTEXTURESECTION
 {
   #if WC3 Reforged
      char texturePath[268];   // string followed by 0's till 268
   #else
      REPLACEABLE_MATERIAL_IDS replaceableId;   // used for texture variations or 0 for none
      char image[0x104];                        // 0 filled when replaceableId is set
      uint32_t flags;                           // &1: wrap width, &2: wrap height
   #endif
 };
 
 enum REPLACEABLE_MATERIAL_IDS : uint32_t
 {
   TEX_COMPONENT_SKIN = 0x1,
   TEX_COMPONENT_OBJECT_SKIN = 0x2,
   TEX_COMPONENT_WEAPON_BLADE = 0x3,
   TEX_COMPONENT_WEAPON_HANDLE = 0x4,
   TEX_COMPONENT_ENVIRONMENT = 0x5,
   TEX_COMPONENT_CHAR_HAIR = 0x6,
   TEX_COMPONENT_CHAR_FACIAL_HAIR = 0x7,
   TEX_COMPONENT_SKIN_EXTRA = 0x8,
   TEX_COMPONENT_UI_SKIN = 0x9,
   TEX_COMPONENT_TAUREN_MANE = 0xA,
   TEX_COMPONENT_MONSTER_1 = 0xB,
   TEX_COMPONENT_MONSTER_2 = 0xC,
   TEX_COMPONENT_MONSTER_3 = 0xD,
   TEX_COMPONENT_ITEM_ICON = 0xE,
   NUM_REPLACEABLE_MATERIAL_IDS = 0xF,
 };

==TXAN==
Texture Animations.
 struct TXAN
 {
   uint32_t numTexAnims;
   MDLTEXANIMSECTION textureAnims[numTexAnims];
 };
 
 struct MDLTEXANIMSECTION
 {
   uint32_t size;
   
   [[#KTAT|KTAT]] transkeys;
   [[#KTAR|KTAR]] rotkeys;
   [[#KTAS|KTAS]] scalekeys;
 };

===KTAT===
Texture animation translation track
 struct KTAT
 {
   char tag[4]; // KTAT
   [[#MDLKEYTRACK|MDLKEYTRACK]]<{{Template:Type|C3Vector}}> transkeys;
 };
===KTAR===
Texture animation rotation track
 struct KTAR
 {
   char tag[4]; // KTAR
   [[#MDLKEYTRACK|MDLKEYTRACK]]<[[#C4QuaternionCompressed|C4QuaternionCompressed]]> rotkeys;
 };
===KTAS===
Texture animation scale track
 struct KTAS
 {
   char tag[4]; // KTAS
   [[#MDLKEYTRACK|MDLKEYTRACK]]<{{Template:Type|C3Vector}}> scalekeys;
 };

==GEOS==
Geosets.

===GEOS (≤ v1400)===
 struct GEOS
 {
   uint32_t numGeosets;                  // limited to 0xFF
   MDLGEOSETSECTION geosets[numGeosets];
 };
 
 struct MDLGEOSETSECTION
 {
   uint32_t size;
   
   [[#VRTX|VRTX]] vertices;
   [[#NRMS|NRMS]] normals;
   [[#UAVS|UAVS]] texCoords;
   MDLPRIMITIVES primitives;
   [[#GNDX|GNDX]] vertGroupIndices;
   [[#MTGC|MTGC]] groupMatrixCounts;
   [[#MATS|MATS]] matrices;
   [[#BIDX|BIDX]] boneIndices;
   [[#BWGT|BWGT]] boneWeights;
   
   uint32_t materialId;                  // [[#MTLS|MTLS]] index
   uint32_t selectionGroup;              // when formatted as four digits, the first two digits map to CHARACTER_GEOSET_SECTIONS, the second two digits are an associated sub-group
                                         // see the related [[M2/.skin#Mesh_part_ID|M2 skin section]] for more information
   uint32_t flags;                       // &1: unselectable
   [[#CMdlBounds|CMdlBounds]] bounds;
   uint32_t numSeqBounds;
   [[#CMdlBounds|CMdlBounds]] seqBounds[numSeqBounds];
 };
 
 struct MDLPRIMITIVES
 {
   [[#PTYP|PTYP]] types;
   [[#PCNT|PCNT]] counts;
   [[#PVTX|PVTX]] vertices;
 };
 
 enum CHARACTER_GEOSET_SECTIONS
 {
   CHARGEOSET_HAIR = 0x0,
   CHARGEOSET_BEARD = 0x1,
   CHARGEOSET_SIDEBURN = 0x2,
   CHARGEOSET_MOUSTACHE = 0x3,
   CHARGEOSET_GLOVE = 0x4,
   CHARGEOSET_BOOT = 0x5,
   CHARGEOSET_OBSOLETEDONTUSEME = 0x6,
   CHARGEOSET_EAR = 0x7,
   CHARGEOSET_SLEEVES = 0x8,
   CHARGEOSET_PANTS = 0x9,
   CHARGEOSET_DOUBLET = 0xA,
   CHARGEOSET_PANTDOUBLET = 0xB,
   CHARGEOSET_TABARD = 0xC,
   CHARGEOSET_ROBE = 0xD,
   CHARGEOSET_LOINCLOTH = 0xE,
   NUM_CHARGEOSETS = 0xF,
   CHARGEOSET_NONE = 0xFFFFFFFF,
 };

===GEOS (v1500)===
{{Template:SectionBox|This section only applies to version 1500.}}
 struct GEOS
 {
   uint32_t numGeosets;                  // limited to 0xFF
   MDLGEOSETSECTION geosets[numGeosets];
   MDLBATCH batches[numGeosets];
 };
 
 struct MDLGEOSETSECTION
 {
    uint32_t materialId;
    {{Template:Type|C3Vector}} boundsCentre;
    float boundsRadius;
    uint32_t selectionGroup;
    uint32_t geosetIndex;
    uint32_t flags;                  // &1: unselectable, &0x10: project2D, &0x20: shaderSkin, other flags are unimplemented
    
    char vertexTag[4];               // PVTX
    uint32_t vertexCount;
    char primTypeTag[4];             // PTYP
    uint32_t primitiveTypesCount;
    char primVertexTag[4];           // PVTX (duplicated tag name, client doesn't validate them)
    uint32_t primitiveVerticesCount;
 
    uint64_t unused;                 // explicitly 0, ignored by client
 }
 
 struct MDLBATCH
 {
    const MDLGEOSETSECTION geoset = GEOS.geosets[index];   // GEOS geoset of matching index
    
    MDLVERTEX vertices[geoset.vertexCount];
    uint32_t primitiveType;  				  // always 0x3 (Triangle)
    uint32_t unknown;  				          // always 0
    
    uint16_t numPrimVertices;  				  // matches geoset.primitiveVerticesCount
    uint16_t minVertex;   
    uint16_t maxVertex;
    uint16_t unused;  				          // explicitly 0, ignored by client
    
    uint16_t primitiveVertices[numPrimVertices];   
    
  #if numPrimVertices % 8 != 0
    uint16_t padding[x];     // alignment padding, calculated as <code>x = (8 - numPrimVertices % 8)</code>
  #endif
 }
 
 struct MDLVERTEX            // same structure as [[M2#Vertices|M2Vertex]]
 {
    {{Template:Type|C3Vector}} position;
    uint8_t boneWeights[4];
    uint8_t boneIndices[4];
    {{Template:Type|C3Vector}} normal;
    {{Template:Type|C2Vector}} texCoords[2];  // second is always (0,0) in all beta files however use of both is supported
 }

===GEOS (Reforged)===
Reforged doesn't have a number of geosets, instead we read SUBMESHes till end of chunk.
 struct GEOS
 {
   SUBMESH submeshes[];
 };
 
 struct SUBMESH
 {
   int32 submeshSize;    // including current int
   [[#VRTX|VRTX]] vertices;
   [[#NRMS|NRMS]] normals;
   [[#PTYP|PTYP]] types;
   [[#PVTX|PVTX]] vertices;
   [[#GNDX|GNDX]] vertGroupIndices;
   [[#MTGC|MTGC]] groupMatrixCounts;
   [[#MATS|MATS]] matrices;
   [[#TANG|TANG]] tangents;
   [[#SKIN|SKIN]] boneWeights;
   [[#UVAS|UVAS]] numberOfUVBS;               // Used to be texture coordinates in the original WC3, now just an int counting the number of UV layers
   [[#UVBS|UVBS]] texCoords;         // Chunk can be present twice when the mesh uses two uv layers UV0, UV1
 }
 
===VRTX===
Vertices. Also used by [[#CLID|CLID]].
 struct VRTX
 {
   char tag[4]; // VRTX
   uint32_t count; // limited to 0xFFFF
   {{Template:Type|C3Vector}} vertices[count]; 
 };
===NRMS===
Normals. Also used by [[#CLID|CLID]].
 struct NRMS
 {
   char tag[4]; // NRMS
   uint32_t count;
   {{Template:Type|C3Vector}} normals[count];
 };
===UVAS===
Texture coordinates. The client uses UVAS.count * [[#VRTX|VRTX]].count to calculate how many C2Vectors to read
 struct UVAS
 {
   #if WC3 Reforged
      int32 numberOfUVBS;
   #else
      char tag[4]; // UVAS
      uint32_t count;
      {{Template:Type|C2Vector}} texCoords[count * vertices.count];
   #endif
 };

===UVBS===
Texture coordinates. The client uses UVAS.count * [[#VRTX|VRTX]].count to calculate how many C2Vectors to read
 struct UVAS
 {
   char tag[4]; // UVBS
   uint32_t count;
   {{Template:Type|C2Vector}} texCoords[count];
 };

===PTYP===
Primitive types. This is always 0x4 (Triangle) although the client appears to support all <tt>FACETYPE</tt>s
 struct PTYP
 {
   char tag[4]; // PTYP
   uint32_t count;
   FACETYPE primitiveTypes[count];
 };
 
 enum FACETYPE : uint8_t
 {
   FACETYPE_POINTS = 0x0,
   FACETYPE_LINES = 0x1,
   FACETYPE_LINE_LOOP = 0x2,
   FACETYPE_LINE_STRIP = 0x3,
   FACETYPE_TRIANGLES = 0x4,
   FACETYPE_TRIANGLE_STRIP = 0x5,
   FACETYPE_TRIANGLE_FAN = 0x6,
   FACETYPE_QUADS = 0x7,
   FACETYPE_QUAD_STRIP = 0x8,
   FACETYPE_POLYGON = 0x9
 };
===PCNT===
Primitive counts. The number of uint16_ts used by [[#PVTX|PVTX]] in each group
 struct PCNT
 {
   char tag[4]; // PCNT
   uint32_t count;
   uint32_t primitiveCounts[count];
 };
===PVTX===
Primitive vertices
 struct PVTX
 {
   char tag[4]; // PVTX
   uint32_t count;
   uint16_t primitiveVertices[count];
 };
===GNDX===
Vertex group indices
 struct GNDX
 {
   char tag[4]; // GNDX
   uint32_t count;
   uint8_t vertGroupIndices[count];
 };
===MTGC===
Group matrix counts
 struct MTGC
 {
   char tag[4]; // MTGC
   uint32_t count;
   uint32_t groupMatrixCounts[count];
 };
===MATS===
Matrices
 struct MATS
 {
   char tag[4]; // MATS
   uint32_t count;
   uint32_t matrices[count];
 };
===TANG===
Vertex tangents
 struct TANG
 {
   char tag[4]; // TANG
   uint32_t count;
   C4Vector tangents[count];
 };
===SKIN===
Vertex weights. Divide weight values by 255f to normalize.
 struct SKIN
 {
   char tag[4]; // SKIN
   uint32_t count;
   BONEWEIGHT weights[count / 8];
 };
 struct BONEWEIGHT
 {
   byte boneIndex0;
   byte boneIndex1;
   byte boneIndex2;
   byte boneIndex3;
   byte weight0;
   byte weight1;
   byte weight2;
   byte weight3;
 };
===BIDX===
Bone indices
 struct BIDX
 {
   char tag[4]; // BIDX
   uint32_t count;
   uint32_t boneIndices[count];
 };
===BWGT===
Bone weights
 struct BWGT
 {
   char tag[4]; // BWGT
   uint32_t count;
   uint32_t boneWeights[count];
 };

==GEOA==
Geoset animations
 struct GEOA
 {
   uint32_t numGeoAnims;
   MDLGEOSETANIMSECTION geosetAnims[numGeoAnims];
 };
  
 struct MDLGEOSETANIMSECTION
 {
   uint32_t size;
   uint32_t geosetId;        // [[#GEOS|GEOS]] index or 0xFFFFFFFF if none
   float staticAlpha;        // 0 is transparent, 1 is opaque
   [[#C3Color|C3Color]] staticColor;
   uint32_t flags;           // &1: color
   
   [[#KGAO|KGAO]] alphaKeys;
   [[#KGAC|KGAC]] colorKeys;
 };

===KGAO===
Animated geoset alpha track
 struct KGAO
 {
   char tag[4]; // KGAO
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> alphaKeys;
 };
===KGAC===
Animated geoset color track
 struct KGAC
 {
   char tag[4]; // KGAC
   [[#MDLKEYTRACK|MDLKEYTRACK]]<[[#C3Color|C3Color]]> colorKeys;
 };
        
==BONE==
Bones
 struct BONE
 {
   #if WC3 Reforged
      MDLBONESECTION bones[];   // Reforged doesn't have a bone count, read till end of chunk
   #else
      uint32_t numBones;
      MDLBONESECTION bones[numBones];
   #end
 };
 
 struct MDLBONESECTION : MDLGENOBJECT
 {
   [[#MDLGENOBJECT|MDLGENOBJECT]] object;
   
   uint32_t geosetId;       // [[#GEOS|GEOS]] index or 0xFFFFFFFF if none
   uint32_t geosetAnimId;   // [[#GEOA|GEOA]] index or 0xFFFFFFFF if none
 };

==LITE==
Lights.
 struct LITE
 {
   uint32_t numLights;
   MDLLIGHTSECTION lights[numLights];
 };
 
 struct MDLLIGHTSECTION : MDLGENOBJECT
 {
   uint32_t size;
   [[#MDLGENOBJECT|MDLGENOBJECT]] object;
   
   LIGHT_TYPE type;
   float staticAttenStart;
   float staticAttenEnd;
   [[#C3Color|C3Color]] staticColor;
   float staticIntensity;  
   [[#C3Color|C3Color]] staticAmbColor;
   float staticAmbIntensity;
   
   [[#KLAS|KLAS]] attenstartkeys;
   [[#KLAE|KLAE]] attenendkeys;
   [[#KLAC|KLAC]] colorkeys;
   [[#KLAI|KLAI]] intensitykeys;
   [[#KLBC|KLBC]] ambcolorkeys;
   [[#KLBI|KLBI]] ambintensitykeys;
   [[#KVIS|KVIS]] visibilityKeys;
 };
 
 enum LIGHT_TYPE : uint32_t
 {
   LIGHTTYPE_OMNI = 0x0,
   LIGHTTYPE_DIRECT = 0x1,
   LIGHTTYPE_AMBIENT = 0x2,
   NUM_MDL_LIGHT_TYPES = 0x3,
 };

===KLAS===
Light attenuation start track
 struct KLAS
 {
   char tag[4]; // KLAS
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> attenstartkeys;
 };
===KLAE===
Light attenuation end track
 struct KLAE
 {
   char tag[4]; // KLAE
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> attenendkeys;
 };
===KLAC===
Light color track
 struct KLAC
 {
   char tag[4]; // KLAC
   [[#MDLKEYTRACK|MDLKEYTRACK]]<[[#C3Color|C3Color]]> colorkeys;
 };
===KLAI===
Light intensity track
 struct KLAI
 {
   char tag[4]; // KLAI
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> intensitykeys;
 };
===KLBC===
Light ambience color track
 struct KLBC
 {
   char tag[4]; // KLBC
   [[#MDLKEYTRACK|MDLKEYTRACK]]<[[#C3Color|C3Color]]> ambcolorkeys
 };
===KLBI===
Light ambient intensity track
 struct KLBI
 {
   char tag[4]; // KLBI
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> ambintensitykeys;
 };
===KVIS===
Visiblity track. <b>Note:</b> Unlike other tracks this one is used globally. Values are boolean floats of 0.0 and 1.0 
 struct KVIS
 {
   char tag[4]; // KVIS
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> visibilityKeys;
 };

===KATV===
Visiblity track. Values are boolean floats of 0.0 and 1.0 
 struct KATV
 {
   char tag[4]; // KATV
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> visibilityKeys;
 };

==HELP==
Helpers.
 struct HELP
 {
   uint32_t count;
   MDLGENOBJECT helpers[count];
 };

==ATCH==
Attachment Points.
 struct ATCH
 {
   #if WC3 Reforged
      MDLATTACHMENTSECTIONREFORGED attachments[];   // Reforged doesn't have an attachment count, read till end of chunk
   #else
      uint32_t numAttachments;
      uint32_t unused;                 // has values but is ignored by the client
      MDLATTACHMENTSECTION attachments[numAttachments];
   #end
 };
 
 struct MDLATTACHMENTSECTION : MDLGENOBJECT
 {
   uint32_t size;
   [[#MDLGENOBJECT|MDLGENOBJECT]] object;             // MDLGENOBJECT.name is loosely the GEOCOMPONENTLINKS enum
   
   GEOCOMPONENTLINKS attachmentId;
   uint8_t padding;
   char path[0x104];                // 0 filled in all alpha files
   
   [[#KVIS|KVIS]] visibilityKeys;
 };

 struct MDLATTACHMENTSECTIONREFORGED : MDLGENOBJECT
 {
   uint32_t size;
   [[#MDLGENOBJECT|MDLGENOBJECT]] object;             // MDLGENOBJECT.name is loosely the GEOCOMPONENTLINKS enum

   [[#KATV|KATV]] visibilityKeys;
 };
 
 enum GEOCOMPONENTLINKS : uint32_t
 {
   ATTACH_SHIELD = 0x0,
   ATTACH_HANDR = 0x1,
   ATTACH_HANDL = 0x2,
   ATTACH_ELBOWR = 0x3,
   ATTACH_ELBOWL = 0x4,
   ATTACH_SHOULDERR = 0x5,
   ATTACH_SHOULDERL = 0x6,
   ATTACH_KNEER = 0x7,
   ATTACH_KNEEL = 0x8,
   ATTACH_HIPR = 0x9,
   ATTACH_HIPL = 0xA,
   ATTACH_HELM = 0xB,
   ATTACH_BACK = 0xC,
   ATTACH_SHOULDERFLAPR = 0xD,
   ATTACH_SHOULDERFLAPL = 0xE,
   ATTACH_TORSOBLOODFRONT = 0xF,
   ATTACH_TORSOBLOODBACK = 0x10,
   ATTACH_BREATH = 0x11,
   ATTACH_PLAYERNAME = 0x12,
   ATTACH_UNITEFFECT_BASE = 0x13,
   ATTACH_UNITEFFECT_HEAD = 0x14,
   ATTACH_UNITEFFECT_SPELLLEFTHAND = 0x15,
   ATTACH_UNITEFFECT_SPELLRIGHTHAND = 0x16,
   ATTACH_UNITEFFECT_SPECIAL1 = 0x17,
   ATTACH_UNITEFFECT_SPECIAL2 = 0x18,
   ATTACH_UNITEFFECT_SPECIAL3 = 0x19,
   ATTACH_SHEATH_MAINHAND = 0x1A,
   ATTACH_SHEATH_OFFHAND = 0x1B,
   ATTACH_SHEATH_SHIELD = 0x1C,
   ATTACH_PLAYERNAMEMOUNTED = 0x1D,
   ATTACH_LARGEWEAPONLEFT = 0x1E,
   ATTACH_LARGEWEAPONRIGHT = 0x1F,
   ATTACH_HIPWEAPONLEFT = 0x20,
   ATTACH_HIPWEAPONRIGHT = 0x21,
   ATTACH_TORSOSPELL = 0x22,
   ATTACH_HANDARROW = 0x23,
   NUM_ATTACH_SLOTS = 0x24,
   ATTACH_NONE = 0xFFFFFFFF,
 };

==PIVT==
Pivot points. The client reads C3Vectors until chunk.size bytes have been read. PivotPoints are paired with <tt>MDLGENOBJECT</tt>s by matching indices.
 struct PIVT
 {
   {{Template:Type|C3Vector}} pivotPoints[chunk.size / 0xC];	 
 };

==PREM==
Particle emitters. <b>Note:</b> This is deprecated use [[#PRE2|PRE2]] instead.
 struct PREM
 { 
   uint32_t numEmitters;
   MDLPARTICLEEMITTER emitters[numEmitters];
 };
 
 struct MDLPARTICLEEMITTER : MDLGENOBJECT
 {
   uint32_t size;
   [[#MDLGENOBJECT|MDLGENOBJECT]] object;
   
   float staticEmissionRate;  
   float staticGravity;  
   float staticLongitude;
   float staticLatitude;
   MDLPARTICLE particle;
   
   [[#KPEE|KPEE]] emissionRate;
   [[#KPEG|KPEG]] gravity;
   [[#KPLN|KPLN]] longitude;
   [[#KPLT|KPLT]] latitude;
   [[#KVIS|KVIS]] visibilityKeys;
 };
 
 struct MDLPARTICLE
 {
   char path[0x104];       // model path
   float staticLife;
   float staticSpeed;
   
   [[#KPEL|KPEL]] life;
   [[#KPES|KPES]] speed;
 };

===KPEE===
Particle emitter emission rate track
 struct KPEE
 {
   char tag[4]; // KPEE
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> emissionRate;
 };
===KPEG===
Particle emitter particle gravity track
 struct KPEG
 {
   char tag[4]; // KPEG
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> gravity;
 };
===KPLT===
Particle emitter particle latitude track
 struct KPLT
 {
   char tag[4]; // KPLT
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> latitude;
 };
===KPEL===
Particle emitter particle life track
 struct KPEL
 {
   char tag[4]; // KPEL
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> life;
 };
===KPES===
Particle emitter particle speed track
 struct KPES
 {
   char tag[4]; // KPES
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> speed;
 };

==CAMS==
Cameras.
 struct CAMS
 {
   uint32_t numCameras;
   MDLCAMERASECTION cameras[numCameras];   
 };
 
 struct MDLCAMERASECTION
 {
   uint32_t size;
   char name[0x50];        // common names are CameraPortrait, Portrait and Paperdoll
   {{Template:Type|C3Vector}} pivot;
   float fieldOfView;      // default is 0.9500215
   float farClip;          // default is 27.7777786
   float nearClip;         // default is 0.222222224
   {{Template:Type|C3Vector}} targetPivot;
   
   [[#KCTR|KCTR]] transkeys;
   [[#KCRL|KCRL]] rollkeys;
   [[#KVIS|KVIS]] visibilityKeys;
   [[#KTTR|KTTR]] targettranskeys;
 };

===KCTR===
Camera translation track
 struct KCTR
 {
   char tag[4]; // KCTR
   [[#MDLKEYTRACK|MDLKEYTRACK]]<{{Template:Type|C3Vector}}> transkeys;
 };
===KCRL===
Camera roll track
 struct KCRL
 {
   char tag[4]; // KCRL
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> rollkeys;
 };
===KTTR===
Camera target translation track
 struct KTTR
 {
   char tag[4]; // KTTR
   [[#MDLKEYTRACK|MDLKEYTRACK]]<{{Template:Type|C3Vector}}> targettranskeys;
 };

==EVTS==
Events. For a complete list see the [[M2#Possible_Events|M2 events section]].
 struct EVTS
 {
   uint32_t numEventObjs;
   MDLEVENTSECTION events[numEventObjs];
 };
 
 struct MDLEVENTSECTION : MDLGENOBJECT
 {
   uint32_t size;
   [[#MDLGENOBJECT|MDLGENOBJECT]] object;
   
   [[#KEVT|KEVT]] eventKeys;
 };

===KEVT===
Event time track
 struct KEVT
 {
   char tag[4]; // KEVT
   [[#MDLSIMPLEKEYTRACK|MDLSIMPLEKEYTRACK]]<MDLEVENTKEY> eventKeys;
 };

==PRE2==
Particle Emitter 2, the successor of the [[#PREM|PREM]] chunk.
 struct PRE2
 {
   #if WC3Reforged
      MDLPARTICLEEMITTER2 emitters[];    // Reforged doesn't have a number of emitters, read till end of chunk
   #else
      uint32_t numEmitters;
      MDLPARTICLEEMITTER2 emitters[numEmitters];
   #end
 };
 
 struct MDLPARTICLEEMITTER2 : MDLGENOBJECT
 {
   uint32_t size;
   [[#MDLGENOBJECT|MDLGENOBJECT]] object;
   
   #if !WC3Reforged
      uint32_t emitterSize;
      PARTICLE_EMITTER_TYPE emitterType;
   #end
   float staticSpeed;          // particleVelocity
   float staticVariation;      // particleVelocityVariation, velocity multiplier. client adds 1.0 and multiplies by random multiplier
   float staticLatitude;
   #if !WC3Reforged
      float staticLongitude;
   #end
   float staticGravity;        // particleAcceleration, only applied to the z axis
   #if !WC3Reforged
      float staticZsource;        // deducted from the particle starting z position. must be ≥ 0.0
   #end
   float staticLife;           // base particle lifespan in seconds
   float staticEmissionRate;   // base amount of particles per second. client treats negatives as 0.0
   float staticLength;         // height, for <tt>PET_SPLINE endAngle (multiplied by emissionRate)</tt>, for <tt>PET_SPHERE outerRadius</tt>
   float staticWidth;          // width, for <tt>PET_SPLINE startAngle</tt>, for <tt>PET_SPHERE innerRadius</tt>
   #if WC3Reforged
      PARTICLE_BLEND_MODE blendMode;
   #end
   uint32_t rows;
   uint32_t cols;
   PARTICLE_TYPE type; 
   float tailLength;
   float middleTime;
   [[#C3Color|C3Color]] startColor;
   [[#C3Color|C3Color]] middleColor;
   [[#C3Color|C3Color]] endColor;
   uint8_t startAlpha;
   uint8_t middleAlpha;
   uint8_t endAlpha;   
   float startScale;
   float middleScale;
   float endScale;
   uint32_t lifespanUVAnimStart;
   uint32_t lifespanUVAnimEnd;
   uint32_t lifespanUVAnimRepeat;
   uint32_t decayUVAnimStart;
   uint32_t decayUVAnimEnd;
   uint32_t decayUVAnimRepeat;
   uint32_t tailUVAnimStart;
   uint32_t tailUVAnimEnd;
   uint32_t tailUVAnimRepeat;
   uint32_t tailDecayUVAnimStart;
   uint32_t tailDecayUVAnimEnd;
   uint32_t tailDecayUVAnimRepeat;
   #if !WC3Reforged
      PARTICLE_BLEND_MODE blendMode;
   #end
   uint32_t textureId;         // [[#TEXS|TEXS]] index or 0xFFFFFFFF if none
   #if WC3Reforged
      int32_t squirts;
   #end
   int32_t priorityPlane;      // priority is sorted lowest to highest
   uint32_t replaceableId;     // only seen in <tt>Wisp.mdx</tt>
   #if !WC3Reforged
      char geometryMdl[0x104];    // particle model
      char recursionMdl[0x104];  
      float twinkleFPS;           // default is 10.0
      float twinkleOnOff;         // boolean, twinkle applies additional scaling to make a shrink and grow effect
      float twinkleScaleMin;      // twinkle is not applied if <code>twinkleScaleMax - twinkleScaleMin == 0.0</code>
      float twinkleScaleMax;
      float ivelScale;            // instant velocity scale, multiplier for each particle's intial velocity
      float tumblexMin;           // tumble adds a randomised rotation to each particle
      float tumblexMax;
      float tumbleyMin;
      float tumbleyMax;
      float tumblezMin;
      float tumblezMax;
      float drag;                 // decreases particle velocity over time
      float spin;
      {{Template:Type|C3Vector}} windVector;       // simulates being blown
      float windTime;             // how long windVector is to be applied
      float followSpeed1;
      float followScale1;
      float followSpeed2;
      float followScale2;
      uint32_t numSplines;
      {{Template:Type|C3Vector}} spline[numSplines];
      uint32_t squirts;           // boolean
   #end   
   #if WC3Reforged
      [[#KP2V|KP2V]] visibilityKeys;
   #else
      [[#KVIS|KVIS]] visibilityKeys;
   #end
   [[#KP2S|KP2S]] speed;
   [[#KP2R|KP2R]] variation;
   [[#KP2L|KP2L]] latitude;
   [[#KPLN|KPLN]] longitude;
   [[#KP2G|KP2G]] gravity;
   [[#KLIF|KLIF]] life;
   [[#KP2E|KP2E]] emissionRate;
   [[#KP2W|KP2W]] width;
   [[#KP2N|KP2N]] length;
   [[#KP2Z|KP2Z]] zsource;
 };
 
 enum PARTICLE_BLEND_MODE : uint32_t
 {
   PBM_BLEND = 0x0,
   PBM_ADD = 0x1,
   PBM_MODULATE = 0x2,
   PBM_MODULATE_2X = 0x3,
   PBM_ALPHA_KEY = 0x4,
   NUM_PARTICLE_BLEND_MODES = 0x5,
 };
 
 enum PARTICLE_TYPE : uint32_t
 {
   PT_HEAD = 0x0,
   PT_TAIL = 0x1,
   PT_BOTH = 0x2,
   NUM_PARTICLE_TYPES = 0x3,
 };
 
 enum PARTICLE_EMITTER_TYPE : uint32_t
 {
   PET_BASE = 0x0,
   PET_PLANE = 0x1,
   PET_SPHERE = 0x2,
   PET_SPLINE = 0x3,
   NUM_PARTICLE_EMITTER_TYPES = 0x4,
 };

===KP2V===
Particle emitter 2 visibility track
 struct KP2V
 {
   char tag[4]; // KP2V
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> visibility;
 };
===KP2S===
Particle emitter 2 speed track
 struct KP2S
 {
   char tag[4]; // KP2S
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> speed;
 };
===KP2R===
Particle emitter 2 variation track
 struct KP2R
 {
   char tag[4]; // KP2R
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> variation;
 };
===KP2L===
Particle emitter 2 latitude track
 struct KP2L
 {
   char tag[4]; // KP2L
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> latitude;
 };
===KPLN===
Particle emitter 2 longitude track
 struct KPLN
 {
   char tag[4]; // KPLN
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> longitude;
 };
===KP2G===
Particle emitter 2 gravity track
 struct KP2G
 {
   char tag[4]; // KP2G
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> gravity;
 };
===KLIF===
Particle emitter 2 life track
 struct KLIF
 {
   char tag[4]; // KLIF
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> life;
 };
===KP2E===
Particle emitter 2 emission rate track
 struct KP2E
 {
   char tag[4]; // KP2E
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> emissionRate; 
 };
===KP2W===
Particle emitter 2 width track
 struct KP2W
 {
   char tag[4]; // KP2W
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> width;
 };
===KP2N===
Particle emitter 2 length track
 struct KP2N
 {
   char tag[4]; // KP2N
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> length;
 };
===KP2Z===
Particle emitter 2 zsource track
 struct KP2Z
 {
   char tag[4]; // KP2Z
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> zsource;
 };

==CORN==
Reforged uses a 3rd party particle emitter as well, named PopcornFX version 2. The chunk references the *.pkb (popcornfx baked files) as well as the attributes exposed in the pkb files.

 struct CORN
 {
   CORNEMITTER cornEmitter[];   // count is unknown, read to end.
 };

 struct CORNEMITTER
 {
   uint32 emitterSize;         // including this int
   [[#MDLGENOBJECT|MDLGENOBJECT]] object;
   C4Color colorMultiplier;   // the color is multiplied with the overall color of the particles
   C4Color teamColor;         // default is (1,1,1,0) turned off, changes the color of some of the particles of some emitters (e.g: Wisp)
   char filePath[260];        // the path to the .pkb file
   char popcornFlags[260];    // comma separated flags (e.g: "Always=on" for emitters that are always on and aren't activated by animation tracks)
   [[#KPPA|KPPA]] alphaMultiplier;
   [[#KPPC|KPPC]] colorMultiplier;
   [[#KPPE|KPPE]] emissionRateMultiplier;
   [[#KPPL|KPPL]] lifespanMultiplier;
   [[#KPPS|KPPS]] speedMultiplier;
   [[#KPPV|KPPV]] visibility;  // on/off
 };

===KPPA===
Popcorn emitter alpha multiplier
 struct KPPA
 {
   char tag[4]; // KPPA
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> alphaMultiplier;
 };

===KPPC===
Popcorn emitter color multiplier
 struct KPPC
 {
   char tag[4]; // KPPC
   [[#MDLKEYTRACK|MDLKEYTRACK]]<C3Color> colorMultiplier;
 };

===KPPE===
Popcorn emitter emission rate multiplier
 struct KPPE
 {
   char tag[4]; // KPPE
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> emissionRateMultiplier;
 };

===KPPL===
Popcorn emitter lifespan multiplier
 struct KPPL
 {
   char tag[4]; // KPPL
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> lifespanMultiplier;
 };

===KPPS===
Popcorn emitter speed multiplier
 struct KPPS
 {
   char tag[4]; // KPPS
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> speedMultiplier;
 };

===KPPV===
Popcorn emitter visibility
 struct KPPV
 {
   char tag[4]; // KPPV
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> visibility; // if 1 emitter is activated, if 0 emitter is deactivated
 };

==HTST==
Hit test shapes.
 struct HTST
 {
   uint32_t numHitTestShapes;
   MDLHITTESTSHAPE hittestshapes[numHitTestShapes];
 };
 
 struct MDLHITTESTSHAPE : MDLGENOBJECT
 {
   uint32_t size;
   [[#MDLGENOBJECT|MDLGENOBJECT]] object;
   
   GEOM_SHAPE type;
   
  #if type == SHAPE_BOX:
   MDLBOX box;
  #elseif type == SHAPE_CYLINDER:
   MDLCYLINDER cylinder;
  #elseif type == SHAPE_SPHERE:
   MDLSPHERE sphere;
  #elseif type == SHAPE_PLANE:
   MDLPLANE plane;
  #endif
 };
 
 enum GEOM_SHAPE : uint8_t
 {
   SHAPE_BOX = 0x0,
   SHAPE_CYLINDER = 0x1,
   SHAPE_SPHERE = 0x2,
   SHAPE_PLANE = 0x3,
   NUM_SHAPES = 0x4,
 };
 
 struct MDLBOX
 {
   {{Template:Type|C3Vector}} minimum;
   {{Template:Type|C3Vector}} maximum;
 };
 
 struct MDLCYLINDER
 {
   {{Template:Type|C3Vector}} base;
   float height;
   float radius;
 };
 
 struct MDLSPHERE
 {
   {{Template:Type|C3Vector}} center;
   float radius;
 };
 
 struct MDLPLANE
 {
   float length;
   float width;
 };

==RIBB==
Ribbon emitter.
 struct RIBB
 {
   uint32_t numEmitters;
   MDLRIBBONEMITTER emitters[numEmitters];
 };
 
 struct MDLRIBBONEMITTER : MDLGENOBJECT
 {
   uint32_t size;
   [[#MDLGENOBJECT|MDLGENOBJECT]] object;
   
   uint32_t emitterSize;
   float staticHeightAbove;        // must be ≥ 0.0
   float staticHeightBelow;        // must be ≥ 0.0
   float staticAlpha;              // 0 is transparent, 1 is opaque
   [[#C3Color|C3Color]] staticColor;
   float edgeLifetime;             // in seconds. must be > 0.0, client forces a minimum of 0.25s
   uint32_t staticTextureSlot;
   uint32_t edgesPerSecond;        // must be ≥ 1.0
   uint32_t textureRows;
   uint32_t textureCols;  
   uint32_t materialId;            // [[#MTLS|MTLS]] index
   float gravity;  
 
   [[#KRHA|KRHA]] heightAbove;
   [[#KRHB|KRHB]] heightBelow;
   [[#KRAL|KRAL]] alphaKeys;
   [[#KRCO|KRCO]] colorKeys;
   [[#KRTX|KRTX]] textureSlot;               // unused by alpha files
   [[#KVIS|KVIS]] visibilityKeys;
 };
 
===KRHA===
Ribbon emitter height above track
 struct KRHA
 {
   char tag[4]; // KRHA
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> heightAbove;
 };
===KRHB===
Ribbon emitter height below track
 struct KRHB
 {
   char tag[4]; // KRHB
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> heightBelow;
 };
===KRAL===
Ribbon emitter alpha track
 struct KRAL
 {
   char tag[4]; // KRAL
   [[#MDLKEYTRACK|MDLKEYTRACK]]<float> alphaKeys;
 };
===KRCO===
Ribbon emitter color track
 struct KRCO
 {
   char tag[4]; // KRCO
   [[#MDLKEYTRACK|MDLKEYTRACK]]<[[#C3Color|C3Color]]> colorKeys;
 };
===KRTX===
Ribbon emitter texture slot track
 struct KRTX
 {
   char tag[4]; // KRTX
   [[#MDLSIMPLEKEYTRACK|MDLSIMPLEKEYTRACK]]<MDLINTKEY> textureSlot;
 };
 
==CLID==
Collision.
 struct MDLCOLLISION
 {
   [[#VRTX|VRTX]] vertices;
   [[#TRI|TRI]] triIndices; 
   [[#NRMS|NRMS]] facetNormals;
 };

===TRI ===
Triangles
 struct TRI
 {
   char tag[4]; // 'TRI ' the space (ASCII char 32) is intentional
   uint32_t count;
   uint16_t triIndices[count];
 };

[[Category:Format]]
