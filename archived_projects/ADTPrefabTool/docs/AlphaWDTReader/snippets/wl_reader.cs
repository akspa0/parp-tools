// docs/AlphaWDTReader/snippets/wl_reader.cs
// Purpose: Minimal readers for WLW/WLQ/WLM legacy liquid files to inform analysis.
// These formats are not loaded by retail clients; we use them to annotate tiles with named waterways.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Snippets
{
    public enum WlLiquidType
    {
        Still = 0,
        Ocean = 1,
        Unknown2 = 2,
        RiverSlow = 4,
        Magma = 6,
        Fast = 8
    }

    public sealed class WlRegion
    {
        public string Source;           // WLW/WLQ/WLM
        public string Name;             // if available from path/metadata
        public int Version;
        public int BlockIndex;
        public WlLiquidType LiquidType;
        public int MinX, MinY, MaxX, MaxY; // bbox in internal coords if derivable
    }

    public static class WlReader
    {
        public static List<WlRegion> ReadWLW(string path)
        {
            var regions = new List<WlRegion>();
            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs);

            uint magic = br.ReadUInt32(); // 'LIQ*' per wowdev
            ushort version = br.ReadUInt16();
            ushort unk06 = br.ReadUInt16();
            ushort liquidType16 = br.ReadUInt16();
            br.ReadUInt16(); // padding
            uint blockCount = br.ReadUInt32();

            for (int i = 0; i < blockCount; i++)
            {
                // Each block: vertices[16] + coord + data[0x50]
                int minx = int.MaxValue, miny = int.MaxValue, maxx = int.MinValue, maxy = int.MinValue;
                // 16 vertices (C3Vectori) -> 16 * (3*4 bytes)
                for (int v = 0; v < 16; v++)
                {
                    int x = br.ReadInt32();
                    int y = br.ReadInt32();
                    int z = br.ReadInt32();
                    minx = Math.Min(minx, x); miny = Math.Min(miny, y);
                    maxx = Math.Max(maxx, x); maxy = Math.Max(maxy, y);
                }
                // internal coord C2Vectori
                int cx = br.ReadInt32();
                int cy = br.ReadInt32();
                // uint16[0x50] data
                for (int d = 0; d < 0x50; d++) br.ReadUInt16();

                regions.Add(new WlRegion
                {
                    Source = "WLW",
                    Name = Path.GetFileNameWithoutExtension(path),
                    Version = version,
                    BlockIndex = i,
                    LiquidType = (WlLiquidType)liquidType16,
                    MinX = minx, MinY = miny, MaxX = maxx, MaxY = maxy
                });
            }

            // Optional second block table (rare); skip conservatively if present
            if (fs.Position + 4 <= fs.Length)
            {
                uint block2Count = br.ReadUInt32();
                long expected = fs.Position + block2Count * (3*4 + 2*4 + 0x38);
                if (expected <= fs.Length)
                {
                    fs.Position = expected; // skip
                }
            }

            return regions;
        }

        public static List<WlRegion> ReadWLQ(string path)
        {
            var regions = new List<WlRegion>();
            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs);

            uint magic = br.ReadUInt32(); // '2QIL'
            ushort version = br.ReadUInt16();
            ushort unk06 = br.ReadUInt16();
            br.ReadBytes(4); // unk08
            uint liquidType = br.ReadUInt32();
            // uint16[9] unk10
            for (int i = 0; i < 9; i++) br.ReadUInt16();
            uint blockCount = br.ReadUInt32();

            for (int i = 0; i < blockCount; i++)
            {
                // same block format as WLW per doc (360 bytes); we conservatively scan to compute bbox
                int minx = int.MaxValue, miny = int.MaxValue, maxx = int.MinValue, maxy = int.MinValue;
                for (int v = 0; v < 16; v++)
                {
                    int x = br.ReadInt32();
                    int y = br.ReadInt32();
                    int z = br.ReadInt32();
                    minx = Math.Min(minx, x); miny = Math.Min(miny, y);
                    maxx = Math.Max(maxx, x); maxy = Math.Max(maxy, y);
                }
                int cx = br.ReadInt32();
                int cy = br.ReadInt32();
                for (int d = 0; d < 0x38 / 2; d++) br.ReadUInt16(); // consume rest of 360 bytes block

                regions.Add(new WlRegion
                {
                    Source = "WLQ",
                    Name = Path.GetFileNameWithoutExtension(path),
                    Version = version,
                    BlockIndex = i,
                    LiquidType = (WlLiquidType)liquidType,
                    MinX = minx, MinY = miny, MaxX = maxx, MaxY = maxy
                });
            }
            return regions;
        }

        public static List<WlRegion> ReadWLM(string path)
        {
            // WLM is magma variant of WLW; parse as WLW but force type Magma
            var regs = ReadWLW(path);
            foreach (var r in regs) r.LiquidType = WlLiquidType.Magma;
            return regs;
        }
    }
}
