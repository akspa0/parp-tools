using System;
using System.IO;
using System.Text;
using GillijimProject.Next.Core.Domain;
using GillijimProject.Next.Core.Domain.Liquids;
using GillijimProject.Next.Core.IO;
using Xunit;

namespace GillijimProject.Next.Tests;

public class AlphaMclqExtractorTests
{
    [Fact]
    public void WaterLayout_HeaderEndOrigin_ParsesVerticesAndTiles()
    {
        // Arrange
        var heights = CreateSeqFloatGrid(MclqData.VertexGrid, MclqData.VertexGrid, baseVal: 100f, step: 0.5f);
        var depth = CreateSeqByteGrid(MclqData.VertexGrid, MclqData.VertexGrid, baseVal: 1);
        var types = CreateFilledTypeGrid(MclqLiquidType.River);
        var flags = CreateFilledFlagsGrid(MclqTileFlags.ForcedSwim);
        var (path, dispose) = TestAdtBuilder.BuildAdtWithSingleMcnk(
            layout: TestAdtBuilder.Layout.Water,
            origin: TestAdtBuilder.OffsetOrigin.HeaderEnd,
            heights: heights,
            depths: depth,
            types: types,
            flags: flags,
            heightMin: 95f,
            heightMax: 150f
        );

        try
        {
            var extractor = new AlphaMclqExtractor();
            var res = extractor.Extract(new AdtAlpha(path));

            // Assert
            Assert.Equal(256, res.Length);
            Assert.NotNull(res[0]);
            Assert.False(res[0] is null);
            var m = res[0];
            Assert.Equal(heights, m.Heights);
            Assert.Equal(depth, m.Depth);
            // spot-check a few tiles
            Assert.Equal(MclqLiquidType.River, m.Types[0, 0]);
            Assert.Equal(MclqTileFlags.ForcedSwim, m.Flags[0, 0] & MclqTileFlags.ForcedSwim);
            Assert.Equal(MclqLiquidType.River, m.Types[7, 7]);
            Assert.Equal(MclqTileFlags.ForcedSwim, m.Flags[7, 7] & MclqTileFlags.ForcedSwim);
        }
        finally
        {
            dispose();
        }
    }

    [Fact]
    public void OceanLayout_DataStartOrigin_InfersHeightsFromMin()
    {
        // Arrange
        var depth = CreateFilledByteGrid(MclqData.VertexGrid, MclqData.VertexGrid, 7);
        var types = CreateFilledTypeGrid(MclqLiquidType.Ocean);
        var flags = CreateFilledFlagsGrid(MclqTileFlags.Fatigue);
        float heightMin = 42.25f;
        float heightMax = 99.75f; // ignored for ocean heights
        var (path, dispose) = TestAdtBuilder.BuildAdtWithSingleMcnk(
            layout: TestAdtBuilder.Layout.Ocean,
            origin: TestAdtBuilder.OffsetOrigin.DataStart,
            heights: null, // not used by ocean
            depths: depth,
            types: types,
            flags: flags,
            heightMin: heightMin,
            heightMax: heightMax
        );

        try
        {
            var extractor = new AlphaMclqExtractor();
            var res = extractor.Extract(new AdtAlpha(path));

            // Assert
            Assert.NotNull(res[0]);
            Assert.False(res[0] is null);
            var m = res[0];
            // All heights should equal heightMin for ocean layout
            for (int i = 0; i < m.Heights.Length; i++)
                Assert.Equal(heightMin, m.Heights[i]);
            // Depths preserved
            Assert.Equal(depth, m.Depth);
            // Tiles preserved
            Assert.Equal(MclqLiquidType.Ocean, m.Types[3, 4]);
            Assert.True((m.Flags[3, 4] & MclqTileFlags.Fatigue) != 0);
        }
        finally
        {
            dispose();
        }
    }

    [Fact]
    public void MagmaLayout_ChunkBeginOrigin_DepthZeroAndHeightsParsed()
    {
        // Arrange
        var heights = CreateFilledFloatGrid(MclqData.VertexGrid, MclqData.VertexGrid, 123.456f);
        var types = CreateFilledTypeGrid(MclqLiquidType.Magma);
        var flags = CreateFilledFlagsGrid(MclqTileFlags.Unknown20);
        var (path, dispose) = TestAdtBuilder.BuildAdtWithSingleMcnk(
            layout: TestAdtBuilder.Layout.Magma,
            origin: TestAdtBuilder.OffsetOrigin.ChunkBegin,
            heights: heights,
            depths: null, // magma doesn't carry depth per vertex
            types: types,
            flags: flags,
            heightMin: 0f,
            heightMax: 0f
        );

        try
        {
            var extractor = new AlphaMclqExtractor();
            var res = extractor.Extract(new AdtAlpha(path));

            // Assert
            Assert.NotNull(res[0]);
            Assert.False(res[0] is null);
            var m = res[0];
            Assert.Equal(heights, m.Heights);
            // Depth should be zeroed for magma
            foreach (var d in m.Depth) Assert.Equal((byte)0, d);
            // Tile flags preserved
            Assert.True((m.Flags[5, 6] & MclqTileFlags.Unknown20) != 0);
        }
        finally
        {
            dispose();
        }
    }

    [Fact]
    public void WaterLayout_AllOffsetOrigins_ParseSuccessfully()
    {
        foreach (var origin in new[]
        {
            TestAdtBuilder.OffsetOrigin.HeaderEnd,
            TestAdtBuilder.OffsetOrigin.DataStart,
            TestAdtBuilder.OffsetOrigin.ChunkBegin
        })
        {
            var heights = CreateFilledFloatGrid(MclqData.VertexGrid, MclqData.VertexGrid, 10f);
            var depth = CreateFilledByteGrid(MclqData.VertexGrid, MclqData.VertexGrid, 2);
            var types = CreateFilledTypeGrid(MclqLiquidType.Slime);
            var flags = CreateFilledFlagsGrid(MclqTileFlags.None);
            var (path, dispose) = TestAdtBuilder.BuildAdtWithSingleMcnk(
                layout: TestAdtBuilder.Layout.Water,
                origin: origin,
                heights: heights,
                depths: depth,
                types: types,
                flags: flags,
                heightMin: 10f,
                heightMax: 10f
            );

            try
            {
                var extractor = new AlphaMclqExtractor();
                var res = extractor.Extract(new AdtAlpha(path));
                Assert.NotNull(res[0]);
                Assert.False(res[0] is null);
                var m = res[0];
                Assert.Equal(MclqLiquidType.Slime, m.Types[0, 0]);
            }
            finally
            {
                dispose();
            }
        }
    }

    [Fact]
    public void OfsLiquidZero_SkipsChunk_ReturnsNull()
    {
        // Arrange
        var heights = CreateFilledFloatGrid(MclqData.VertexGrid, MclqData.VertexGrid, 5f);
        var depths = CreateFilledByteGrid(MclqData.VertexGrid, MclqData.VertexGrid, 3);
        var types = CreateFilledTypeGrid(MclqLiquidType.River);
        var flags = CreateFilledFlagsGrid(MclqTileFlags.None);
        var (path, dispose) = TestAdtBuilder.BuildAdtWithSingleMcnk(
            layout: TestAdtBuilder.Layout.Water,
            origin: TestAdtBuilder.OffsetOrigin.HeaderEnd,
            heights: heights,
            depths: depths,
            types: types,
            flags: flags,
            heightMin: 5f,
            heightMax: 5f
        );
        try
        {
            PatchMcnkOfsLiquid(path, 0u);
            var extractor = new AlphaMclqExtractor();
            var res = extractor.Extract(new AdtAlpha(path));
            Assert.Null(res[0]);
        }
        finally { dispose(); }
    }

    [Fact]
    public void SizeLiquidTooSmall_SkipsChunk_ReturnsNull()
    {
        var heights = CreateFilledFloatGrid(MclqData.VertexGrid, MclqData.VertexGrid, 8f);
        var depths = CreateFilledByteGrid(MclqData.VertexGrid, MclqData.VertexGrid, 1);
        var types = CreateFilledTypeGrid(MclqLiquidType.River);
        var flags = CreateFilledFlagsGrid(MclqTileFlags.None);
        var (path, dispose) = TestAdtBuilder.BuildAdtWithSingleMcnk(
            layout: TestAdtBuilder.Layout.Water,
            origin: TestAdtBuilder.OffsetOrigin.HeaderEnd,
            heights: heights,
            depths: depths,
            types: types,
            flags: flags,
            heightMin: 8f,
            heightMax: 8f
        );
        try
        {
            // Force sizeLiquid to a too-small non-zero value
            PatchMcnkSizeLiquid(path, 8u);
            var extractor = new AlphaMclqExtractor();
            var res = extractor.Extract(new AdtAlpha(path));
            Assert.Null(res[0]);
        }
        finally { dispose(); }
    }

    [Fact]
    public void TruncatedPayload_ReturnsNull_NoThrow()
    {
        var heights = CreateFilledFloatGrid(MclqData.VertexGrid, MclqData.VertexGrid, 10f);
        var depths = CreateFilledByteGrid(MclqData.VertexGrid, MclqData.VertexGrid, 2);
        var types = CreateFilledTypeGrid(MclqLiquidType.River);
        var flags = CreateFilledFlagsGrid(MclqTileFlags.None);
        var (path, dispose) = TestAdtBuilder.BuildAdtWithSingleMcnk(
            layout: TestAdtBuilder.Layout.Water,
            origin: TestAdtBuilder.OffsetOrigin.HeaderEnd,
            heights: heights,
            depths: depths,
            types: types,
            flags: flags,
            heightMin: 10f,
            heightMax: 10f
        );
        try
        {
            // Remove a few bytes from the end to simulate truncation
            TruncateFile(path, 4);
            var extractor = new AlphaMclqExtractor();
            var res = extractor.Extract(new AdtAlpha(path));
            Assert.Null(res[0]);
        }
        finally { dispose(); }
    }

    private static float[] CreateSeqFloatGrid(int cols, int rows, float baseVal, float step)
    {
        var arr = new float[cols * rows];
        float v = baseVal;
        for (int i = 0; i < arr.Length; i++) { arr[i] = v; v += step; }
        return arr;
    }

    private static float[] CreateFilledFloatGrid(int cols, int rows, float val)
    {
        var arr = new float[cols * rows];
        for (int i = 0; i < arr.Length; i++) arr[i] = val;
        return arr;
    }

    private static byte[] CreateSeqByteGrid(int cols, int rows, byte baseVal)
    {
        var arr = new byte[cols * rows];
        byte v = baseVal;
        for (int i = 0; i < arr.Length; i++) { arr[i] = v; unchecked { v++; } }
        return arr;
    }

    private static byte[] CreateFilledByteGrid(int cols, int rows, byte val)
    {
        var arr = new byte[cols * rows];
        for (int i = 0; i < arr.Length; i++) arr[i] = val;
        return arr;
    }

    private static MclqLiquidType[,] CreateFilledTypeGrid(MclqLiquidType t)
    {
        var arr = new MclqLiquidType[MclqData.TileGrid, MclqData.TileGrid];
        for (int y = 0; y < MclqData.TileGrid; y++)
        for (int x = 0; x < MclqData.TileGrid; x++)
            arr[x, y] = t;
        return arr;
    }

    private static MclqTileFlags[,] CreateFilledFlagsGrid(MclqTileFlags f)
    {
        var arr = new MclqTileFlags[MclqData.TileGrid, MclqData.TileGrid];
        for (int y = 0; y < MclqData.TileGrid; y++)
        for (int x = 0; x < MclqData.TileGrid; x++)
            arr[x, y] = f;
        return arr;
    }

    private static class TestAdtBuilder
    {
        public enum Layout { Water, Ocean, Magma }
        public enum OffsetOrigin { HeaderEnd, DataStart, ChunkBegin }

        public static (string path, Action dispose) BuildAdtWithSingleMcnk(
            Layout layout,
            OffsetOrigin origin,
            float[]? heights,
            byte[]? depths,
            MclqLiquidType[,] types,
            MclqTileFlags[,] flags,
            float heightMin,
            float heightMax)
        {
            string path = Path.Combine(Path.GetTempPath(), $"alpha_mclq_test_{Guid.NewGuid():N}.adt");
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

            // Reserve MCIN (we'll backpatch the entry after writing MCNK)
            WriteFourCC(bw, "MCIN");
            bw.Write((uint)16); // one entry
            long mcinEntryPos = ms.Position; // start of the 16-byte entry
            bw.Write(0u); // offset (backpatch)
            bw.Write(0u); // size (backpatch)
            bw.Write(0u); // flags
            bw.Write(0u); // id

            // Build MCNK content in memory
            var (mcnkBytes, mcnkSize) = BuildMcnkChunk(layout, origin, heights, depths, types, flags, heightMin, heightMax);

            // Remember MCNK absolute offset
            long mcnkOffset = ms.Position;
            ms.Write(mcnkBytes, 0, mcnkBytes.Length);

            // Backpatch MCIN entry
            long cur = ms.Position;
            ms.Position = mcinEntryPos;
            bw.Write((uint)mcnkOffset);
            bw.Write((uint)mcnkSize);
            bw.Write(0u);
            bw.Write(0u);
            ms.Position = cur;

            File.WriteAllBytes(path, ms.ToArray());
            return (path, () => { try { File.Delete(path); } catch { /* ignore */ } });
        }

        private static (byte[] bytes, uint chunkSize) BuildMcnkChunk(
            Layout layout,
            OffsetOrigin origin,
            float[]? heights,
            byte[]? depths,
            MclqLiquidType[,] types,
            MclqTileFlags[,] flags,
            float heightMin,
            float heightMax)
        {
            // Build header (128 bytes) + MCLQ payload
            var header = BuildMcnkHeader(layout, origin);
            var mclq = BuildMclqPayload(layout, heights, depths, types, flags, heightMin, heightMax);

            // Update header's OfsLiquid and SizeLiquid according to origin (MCLQ placed immediately after header)
            const int pad = 16; // ensure non-zero ofsLiquid
            uint ofs = origin switch
            {
                OffsetOrigin.HeaderEnd => (uint)pad,
                OffsetOrigin.DataStart => 128u + (uint)pad,
                OffsetOrigin.ChunkBegin => 136u + (uint)pad, // includes 8 bytes for fourcc+size
                _ => (uint)pad
            };
            PatchMcnkHeaderOfsSize(header, ofs, (uint)mclq.Length);

            // Compute chunk size: 128 header + mclq payload
            uint chunkSize = (uint)(128 + pad + mclq.Length);

            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);
            WriteFourCC(bw, "MCNK");
            bw.Write(chunkSize);
            bw.Write(header, 0, header.Length);
            // write padding to place MCLQ after header by 'pad' bytes
            bw.Write(new byte[pad]);
            bw.Write(mclq, 0, mclq.Length);

            return (ms.ToArray(), chunkSize);
        }

        private static byte[] BuildMcnkHeader(Layout layout, OffsetOrigin origin)
        {
            // Flags
            uint flags = layout switch
            {
                Layout.Magma => 0x10u, // FLAG_LQ_MAGMA
                Layout.Ocean => 0x08u, // FLAG_LQ_OCEAN
                _ => 0x04u,            // FLAG_LQ_RIVER (treated as Water layout)
            };

            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

            // Write header fields in the same order ReadMcnkHeader() reads them
            bw.Write(flags);                   // Flags
            bw.Write(0u);                      // indexX
            bw.Write(0u);                      // indexY
            bw.Write(0f);                      // radius
            bw.Write(0u);                      // nLayers
            bw.Write(0u);                      // nDoodadRefs
            bw.Write(0u);                      // ofsHeight
            bw.Write(0u);                      // ofsNormal
            bw.Write(0u);                      // ofsLayer
            bw.Write(0u);                      // ofsRefs
            bw.Write(0u);                      // ofsAlpha
            bw.Write(0u);                      // sizeAlpha
            bw.Write(0u);                      // ofsShadow
            bw.Write(0u);                      // sizeShadow
            bw.Write(0u);                      // areaid
            bw.Write(0u);                      // nMapObjRefs
            bw.Write((ushort)0);               // holes
            bw.Write((ushort)0);               // pad0
            for (int i = 0; i < 8; i++) bw.Write((ushort)0); // predTex[8]
            for (int i = 0; i < 8; i++) bw.Write((byte)0);   // noEffectDoodad[8]
            bw.Write(0u);                      // ofsSndEmitters
            bw.Write(0u);                      // nSndEmitters

            // Record position for ofsLiquid and sizeLiquid (this matches our reader)
            long ofsPos = ms.Position;
            bw.Write(0u);                      // OfsLiquid (patch later)
            bw.Write(0u);                      // SizeLiquid (patch later)

            // Pad to 128 bytes
            while (ms.Length < 128) bw.Write((byte)0);

            return ms.ToArray();
        }

        private static void PatchMcnkHeaderOfsSize(byte[] header, uint ofsLiquid, uint sizeLiquid)
        {
            // In BuildMcnkHeader we placed OfsLiquid at byte offset 100 and SizeLiquid at 104
            const int ofsPos = 100;
            BitConverter.GetBytes(ofsLiquid).CopyTo(header, ofsPos);
            BitConverter.GetBytes(sizeLiquid).CopyTo(header, ofsPos + 4);
        }

        private static byte[] BuildMclqPayload(
            Layout layout,
            float[]? heights,
            byte[]? depths,
            MclqLiquidType[,] types,
            MclqTileFlags[,] flags,
            float heightMin,
            float heightMax)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

            // CRange
            bw.Write(heightMin);
            bw.Write(heightMax);

            const int V = MclqData.VertexGrid * MclqData.VertexGrid; // 81

            switch (layout)
            {
                case Layout.Water:
                    if (heights is null || depths is null) throw new ArgumentNullException();
                    for (int i = 0; i < V; i++)
                    {
                        bw.Write(depths[i]);      // depth
                        bw.Write((byte)0);        // flow0
                        bw.Write((byte)0);        // flow1
                        bw.Write((byte)0);        // filler
                        bw.Write(heights[i]);     // height
                    }
                    break;
                case Layout.Ocean:
                    if (depths is null) throw new ArgumentNullException();
                    for (int i = 0; i < V; i++)
                    {
                        bw.Write(depths[i]);      // depth
                        bw.Write((byte)0);        // foam
                        bw.Write((byte)0);        // wet
                        bw.Write((byte)0);        // filler
                    }
                    break;
                case Layout.Magma:
                    if (heights is null) throw new ArgumentNullException();
                    for (int i = 0; i < V; i++)
                    {
                        bw.Write((ushort)0);      // u16 s
                        bw.Write((ushort)0);      // u16 t
                        bw.Write(heights[i]);     // height
                    }
                    break;
            }

            // Tiles 8x8
            for (int ty = 0; ty < MclqData.TileGrid; ty++)
            {
                for (int tx = 0; tx < MclqData.TileGrid; tx++)
                {
                    byte b = (byte)(((byte)flags[tx, ty] & 0xF0) | ((byte)types[tx, ty] & 0x0F));
                    bw.Write(b);
                }
            }

            return ms.ToArray();
        }

        private static void WriteFourCC(BinaryWriter bw, string cc)
        {
            var bytes = Encoding.ASCII.GetBytes(cc);
            if (bytes.Length != 4) throw new ArgumentException("fourcc must be 4 chars");
            bw.Write(bytes);
        }
    }

    // Helpers to patch the synthetic ADT on disk
    private static void PatchMcnkOfsLiquid(string path, uint newOfsLiquid)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: true);
        using var bw = new BinaryWriter(fs, Encoding.ASCII, leaveOpen: true);
        // MCIN header is at start: 4cc + size = 8 bytes, then entry
        fs.Position = 8; // start of first MCIN entry
        uint mcnkOff = br.ReadUInt32();
        fs.Position = mcnkOff + 8 + 100; // data start + 100 = ofsLiquid position
        bw.Write(newOfsLiquid);
    }

    private static void PatchMcnkSizeLiquid(string path, uint newSizeLiquid)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: true);
        using var bw = new BinaryWriter(fs, Encoding.ASCII, leaveOpen: true);
        fs.Position = 8;
        uint mcnkOff = br.ReadUInt32();
        fs.Position = mcnkOff + 8 + 104; // data start + 104 = sizeLiquid position
        bw.Write(newSizeLiquid);
    }

    private static void TruncateFile(string path, int bytesToRemove)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        long newLen = Math.Max(0, fs.Length - bytesToRemove);
        fs.SetLength(newLen);
    }
}
