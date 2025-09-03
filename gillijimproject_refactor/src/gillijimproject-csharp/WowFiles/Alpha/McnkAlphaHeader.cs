using System.Runtime.InteropServices;
using GillijimProject.WowFiles;

namespace GillijimProject.WowFiles.Alpha
{
    [StructLayout(LayoutKind.Sequential)]
    public struct McnkAlphaHeader
    {
        public int Flags;
        public int IndexX;
        public int IndexY;
        public float Unknown1;
        public int NLayers;
        public int M2Number;
        public int McvtOffset;
        public int McnrOffset;
        public int MclyOffset;
        public int McrfOffset;
        public int McalOffset;
        public int McalSize;
        public int McshOffset;
        public int McshSize;
        public int Unknown3;
        public int WmoNumber;
        public int Holes;
        public int GroundEffectsMap1;
        public int GroundEffectsMap2;
        public int GroundEffectsMap3;
        public int GroundEffectsMap4;
        public int Unknown6;
        public int Unknown7;
        public int McnkChunksSize;
        public int Unknown8;
        public int MclqOffset;
        public int Unused1;
        public int Unused2;
        public int Unused3;
        public int Unused4;
        public int Unused5;
        public int Unused6;

        public McnkHeader ToMcnkHeader()
        {
            return new McnkHeader
            {
                Flags = this.Flags,
                IndexX = this.IndexX,
                IndexY = this.IndexY,
                NLayers = this.NLayers,
                M2Number = this.M2Number,
                McvtOffset = this.McvtOffset,
                McnrOffset = this.McnrOffset,
                MclyOffset = this.MclyOffset,
                McrfOffset = this.McrfOffset,
                McalOffset = this.McalOffset,
                McalSize = this.McalSize,
                McshOffset = this.McshOffset,
                McshSize = this.McshSize,
                AreaId = this.Unknown3,
                WmoNumber = this.WmoNumber,
                Holes = this.Holes,
                GroundEffectsMap1 = this.GroundEffectsMap1,
                GroundEffectsMap2 = this.GroundEffectsMap2,
                GroundEffectsMap3 = this.GroundEffectsMap3,
                GroundEffectsMap4 = this.GroundEffectsMap4,
                MclqOffset = this.MclqOffset,
            };
        }
    }
}
