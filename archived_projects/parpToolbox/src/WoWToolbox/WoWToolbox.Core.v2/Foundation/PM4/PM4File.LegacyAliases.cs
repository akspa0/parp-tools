using WoWToolbox.Core.v2.Foundation.PM4.Chunks;
using Warcraft.NET.Attribute;

namespace WoWToolbox.Core.v2.Foundation.PM4
{
    // Provides legacy-friendly property aliases so that ported v1 code (expecting VertexPositionsChunk etc.)
    // compiles without heavy edits. These simply forward to the new chunk names used in Core.v2.
    public partial class PM4File
    {
        [ChunkIgnore]
        public MSPVChunk? VertexPositionsChunk
        {
            get => MSPV;
            // Dummy setter to satisfy reflection; sets underlying MSPV if called
            set => MSPV = value;
        }
        [ChunkIgnore]
        public MSVIChunk? VertexIndicesChunk
        {
            get => MSVI;
            set => MSVI = value;
        }
        [ChunkIgnore]
        public MPRLChunk? PositionDataChunk
        {
            get => MPRL;
            set => MPRL = value;
        }
    }
}
