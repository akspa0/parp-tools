// docs/AlphaWDTReader/snippets/mh2o_struct_layout.cs
// Purpose: Annotated MH2O header/instance/vertex layout for WotLK path (3.x).
// References: MapUpconverter MH2O*.cs, gillijimproject Mh2o.h

namespace Snippets
{
    // Note: sizes/pack are indicative; verify against MapUpconverter before use.
    public struct MH2OHeader
    {
        public uint LayerCount;   // number of layers in chunk
        public uint OfsLayers;    // offset to layers array
        public uint OfsAttribs;   // optional attributes
    }

    public struct MH2OLayer
    {
        public ushort MinX, MinY, MaxX, MaxY; // mask bbox
        public ushort Flags;                  // type flags
        public uint OfsHeightmap;             // per-cell heights
        public uint OfsMask;                  // packed mask bits
        public float MinHeight;               // stats
        public float MaxHeight;
    }
}
