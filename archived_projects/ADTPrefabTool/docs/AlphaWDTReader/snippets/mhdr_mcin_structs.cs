// docs/AlphaWDTReader/snippets/mhdr_mcin_structs.cs
// Purpose: Annotated 3.x (WotLK) MHDR/MCIN/MCNK struct layouts for reference.
// Verify exact field order/sizes against WoWFormatLib ADT.Struct.cs before use.

namespace Snippets
{
    // MHDR: ADT header
    public struct MHDR
    {
        public uint Flags;
        public uint OfsMCIN;
        public uint OfsMTEX;
        public uint OfsMMDX;
        public uint OfsMMID;
        public uint OfsMWMO;
        public uint OfsMWID;
        public uint OfsMDDF;
        public uint OfsMODF;
        public uint OfsMFBO;
        public uint OfsMH2O; // WotLK water table (top-level)
        // ... other offsets as per spec
    }

    // MCIN: 256 entries, one per MCNK
    public struct MCINEntry
    {
        public uint OfsMCNK; // offset to MCNK block
        public uint Size;    // size of MCNK block
        public uint Flags;
        public uint AsyncId; // unused/reserved
    }

    // MCNK: map chunk header (followed by subchunks like MCVT/MCNR/etc.)
    public struct MCNK
    {
        public uint Flags;
        public uint IndexX;
        public uint IndexY;
        public uint NLayers;
        public uint NDoodadRefs;
        public uint OfsMCVT; // vertices
        public uint OfsMCNR; // normals
        public uint OfsMCLY; // layers
        public uint OfsMCRF; // refs
        public uint OfsMCSH; // holes
        public uint OfsMCAL; // alpha maps
        public uint OfsMCLQ; // legacy water (should be 0 in our outputs)
        public uint SizeMCLQ;
        // ... other fields per 3.x spec
    }
}
