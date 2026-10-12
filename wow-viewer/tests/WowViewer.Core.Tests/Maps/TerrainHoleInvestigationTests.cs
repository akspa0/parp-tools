using System;
using System.Buffers.Binary;
using System.IO;
using System.Linq;
using System.Numerics;
using WowViewer.Core.IO.Casc;
using WowViewer.Core.IO.Lk;
using WowViewer.Core.Maps;
using Xunit;
using Xunit.Abstractions;

namespace WowViewer.Core.Tests.Maps;

public sealed class TerrainHoleInvestigationTests
{
    private const float MapOrigin = 17066.666666f;
    private const float ChunkSize = 533.333333f;
    private readonly ITestOutputHelper _output;

    public TerrainHoleInvestigationTests(ITestOutputHelper output)
    {
        _output = output;
    }

    // Search all tiles for WMO 111538
    [Fact]
    public void FindUndeadChurchTile()
    {
        const string installDir = @"I:\wow12\World of Warcraft";
        if (!Directory.Exists(installDir)) return;
        var storage = CascStorage.OpenLocal(installDir, "wow_classic_beta", @"output\cache\casc", allowCdnFill: false);
        byte[]? Read(uint id) => id != 0 && storage.TryReadFile(id, out byte[]? b) == CascReadStatus.Ok ? b : null;
        byte[]? wdt = Read(775971);
        Assert.NotNull(wdt);

        int maidOffset = -1;
        for (int i = 0; i + 8 <= wdt.Length; i++)
        {
            if ((wdt[i] == 'D' && wdt[i + 1] == 'I' && wdt[i + 2] == 'A' && wdt[i + 3] == 'M') ||
                (wdt[i] == 'M' && wdt[i + 1] == 'A' && wdt[i + 2] == 'I' && wdt[i + 3] == 'D'))
            {
                maidOffset = i;
                break;
            }
        }
        int maidPayload = maidOffset + 8;
        for (int slot = 0; slot < 4096; slot++)
        {
            uint obj0Id = BitConverter.ToUInt32(wdt, maidPayload + slot * 32 + 4);
            if (obj0Id == 0) continue;
            byte[]? o0 = Read(obj0Id);
            if (o0 == null) continue;
            int offset = 0;
            while (offset + 8 <= o0.Length)
            {
                uint id = BitConverter.ToUInt32(o0, offset);
                uint size = BitConverter.ToUInt32(o0, offset + 4);
                int dataStart = offset + 8;
                if ((id == 0x46444F4D || id == 0x4D4F4446) && dataStart + 64 <= o0.Length)
                {
                    int count = (int)size / 64;
                    for (int i = 0; i < count; i++)
                    {
                        uint nameId = BitConverter.ToUInt32(o0, dataStart + i * 64);
                        if (nameId == 111538)
                        {
                            int tx = slot / 64;
                            int ty = slot % 64;
                            _output.WriteLine($"Found Church (nameId=111538) in Slot {slot} (tx={tx}, ty={ty}, obj0Id={obj0Id})");
                        }
                    }
                }
                offset = dataStart + (int)size;
            }
        }
    }

    [Fact]
    public void Investigate_AzerothTile48_59_HolesAndPlacements()
    {
        const string installDir = @"I:\wow12\World of Warcraft";
        if (!Directory.Exists(installDir))
        {
            _output.WriteLine("Install dir not found, skipping.");
            return;
        }

        var storage = CascStorage.OpenLocal(installDir, "wow_classic_beta", @"output\cache\casc", allowCdnFill: false);
        byte[]? Read(uint id) => id != 0 && storage.TryReadFile(id, out byte[]? b) == CascReadStatus.Ok ? b : null;

        // Azeroth WDT FileDataID: 775971
        byte[]? wdt = Read(775971);
        Assert.NotNull(wdt);

        int maidOffset = -1;
        for (int i = 0; i + 8 <= wdt.Length; i++)
        {
            if ((wdt[i] == 'D' && wdt[i + 1] == 'I' && wdt[i + 2] == 'A' && wdt[i + 3] == 'M') ||
                (wdt[i] == 'M' && wdt[i + 1] == 'A' && wdt[i + 2] == 'I' && wdt[i + 3] == 'D'))
            {
                maidOffset = i;
                break;
            }
        }
        Assert.True(maidOffset >= 0);

        int maidPayload = maidOffset + 8;
        SolveHoleMapping(wdt, maidPayload, Read);
    }

    [Fact]
    public void Investigate_DeathknellTile29_28_ChurchHoles()
    {
        const string installDir = @"I:\wow12\World of Warcraft";
        if (!Directory.Exists(installDir)) return;

        var storage = CascStorage.OpenLocal(installDir, "wow_classic_beta", @"output\cache\casc", allowCdnFill: false);
        byte[]? Read(uint id) => id != 0 && storage.TryReadFile(id, out byte[]? b) == CascReadStatus.Ok ? b : null;

        byte[]? wdt = Read(775971);
        Assert.NotNull(wdt);

        int maidOffset = -1;
        for (int i = 0; i + 8 <= wdt.Length; i++)
        {
            if ((wdt[i] == 'D' && wdt[i + 1] == 'I' && wdt[i + 2] == 'A' && wdt[i + 3] == 'M') ||
                (wdt[i] == 'M' && wdt[i + 1] == 'A' && wdt[i + 2] == 'I' && wdt[i + 3] == 'D'))
            {
                maidOffset = i;
                break;
            }
        }
        Assert.True(maidOffset >= 0);

        int maidPayload = maidOffset + 8;
        int slot = 29 * 64 + 28;
        uint rootId = BitConverter.ToUInt32(wdt, maidPayload + slot * 32);
        uint obj0Id = BitConverter.ToUInt32(wdt, maidPayload + slot * 32 + 4);
        uint obj1Id = BitConverter.ToUInt32(wdt, maidPayload + slot * 32 + 8);

        byte[]? rootBytes = Read(rootId);
        byte[]? obj0Bytes = Read(obj0Id);
        byte[]? obj1Bytes = Read(obj1Id);

        _output.WriteLine($"Tile [29, 28]: rootId={rootId}, obj0Id={obj0Id}, obj1Id={obj1Id}");
        Assert.NotNull(rootBytes);
        AnalyzeTileHoles(rootBytes, obj0Bytes, 29, 28);
        FindPlacementsNearChunks(rootBytes, obj0Bytes, obj1Bytes, 29, 28);
    }

    [Fact]
    public void Investigate_RavenHill_ChurchHoles()
    {
        const string installDir = @"I:\wow12\World of Warcraft";
        if (!Directory.Exists(installDir)) return;

        var storage = CascStorage.OpenLocal(installDir, "wow_classic_beta", @"output\cache\casc", allowCdnFill: false);
        byte[]? Read(uint id) => id != 0 && storage.TryReadFile(id, out byte[]? b) == CascReadStatus.Ok ? b : null;

        byte[]? wdt = Read(775971);
        Assert.NotNull(wdt);

        int maidOffset = -1;
        for (int i = 0; i + 8 <= wdt.Length; i++)
        {
            if ((wdt[i] == 'D' && wdt[i + 1] == 'I' && wdt[i + 2] == 'A' && wdt[i + 3] == 'M') ||
                (wdt[i] == 'M' && wdt[i + 1] == 'A' && wdt[i + 2] == 'I' && wdt[i + 3] == 'D'))
            {
                maidOffset = i;
                break;
            }
        }
        Assert.True(maidOffset >= 0);

        int maidPayload = maidOffset + 8;
        for (int tx = 45; tx <= 49; tx++)
        {
            for (int ty = 32; ty <= 36; ty++)
            {
                int slot = tx * 64 + ty;
                uint rootId = BitConverter.ToUInt32(wdt, maidPayload + slot * 32);
                uint obj0Id = BitConverter.ToUInt32(wdt, maidPayload + slot * 32 + 4);
                uint obj1Id = BitConverter.ToUInt32(wdt, maidPayload + slot * 32 + 8);
                byte[]? rootBytes = Read(rootId);
                byte[]? obj0Bytes = Read(obj0Id);
                byte[]? obj1Bytes = Read(obj1Id);
                if (rootBytes != null)
                {
                    // Scan root for holes
                    int offset = 0;
                    while (offset + 8 <= rootBytes.Length)
                    {
                        uint id = BitConverter.ToUInt32(rootBytes, offset);
                        uint size = BitConverter.ToUInt32(rootBytes, offset + 4);
                        int dataStart = offset + 8;
                        if ((id == 0x4D434E4B || id == 0x4B4E434D) && dataStart + 128 <= rootBytes.Length)
                        {
                            uint ix = BitConverter.ToUInt32(rootBytes, dataStart + 4);
                            uint iy = BitConverter.ToUInt32(rootBytes, dataStart + 8);
                            ulong holes64 = BitConverter.ToUInt64(rootBytes, dataStart + 0x14);
                            ushort holes16 = BitConverter.ToUInt16(rootBytes, dataStart + 0x3C);
                            if (holes64 != 0 || holes16 != 0)
                            {
                                byte[] hb = new byte[8];
                                Array.Copy(rootBytes, dataStart + 0x14, hb, 0, 8);
                                string hex = string.Join(" ", hb.Select(b => $"{b:X2}"));
                                _output.WriteLine($"*** Tile [{tx}, {ty}] Chunk [{ix}, {iy}]: holes64=0x{holes64:X16}, bytes={hex}");
                                for (int r = 0; r < 8; r++)
                                {
                                    string rowBits = "";
                                    for (int c = 0; c < 8; c++) rowBits += ((hb[r] >> c) & 1) != 0 ? "X" : ".";
                                    _output.WriteLine($"      r{r}: {rowBits}");
                                }
                            }
                        }
                        offset = dataStart + (int)size;
                    }

                    // Check for church in obj0 or obj1
                    void CheckWmo(byte[]? b, string tag)
                    {
                        if (b == null) return;
                        int o = 0;
                        while (o + 8 <= b.Length)
                        {
                            uint cid = BitConverter.ToUInt32(b, o);
                            uint csz = BitConverter.ToUInt32(b, o + 4);
                            int ds = o + 8;
                            if ((cid == 0x46444F4D || cid == 0x4D4F4446) && ds + 64 <= b.Length)
                            {
                                int cnt = (int)csz / 64;
                                for (int i = 0; i < cnt; i++)
                                {
                                    uint nid = BitConverter.ToUInt32(b, ds + i * 64);
                                    if (nid == 111538 || nid == 106800)
                                    {
                                        float rawX = BitConverter.ToSingle(b, ds + i * 64 + 8);
                                        float rawY = BitConverter.ToSingle(b, ds + i * 64 + 16);
                                        float rawZ = BitConverter.ToSingle(b, ds + i * 64 + 12);
                                        _output.WriteLine($"*** Tile [{tx}, {ty}] {tag} MODF WMO nid={nid}, rend=({MapOrigin - rawY:F2}, {MapOrigin - rawX:F2}, {rawZ:F2})");
                                    }
                                }
                            }
                            o = ds + (int)csz;
                        }
                    }
                    CheckWmo(obj0Bytes, "obj0");
                    CheckWmo(obj1Bytes, "obj1");

                    // Check M2s near the church
                    void CheckM2(byte[]? b)
                    {
                        if (b == null) return;
                        int o = 0;
                        while (o + 8 <= b.Length)
                        {
                            uint cid = BitConverter.ToUInt32(b, o);
                            uint csz = BitConverter.ToUInt32(b, o + 4);
                            int ds = o + 8;
                            if ((cid == 0x4644444D || cid == 0x4D444446) && ds + 36 <= b.Length)
                            {
                                int cnt = (int)csz / 36;
                                for (int i = 0; i < cnt; i++)
                                {
                                    uint nid = BitConverter.ToUInt32(b, ds + i * 36);
                                    uint uid = BitConverter.ToUInt32(b, ds + i * 36 + 4);
                                    float rawX = BitConverter.ToSingle(b, ds + i * 36 + 8);
                                    float rawZ = BitConverter.ToSingle(b, ds + i * 36 + 12);
                                    float rawY = BitConverter.ToSingle(b, ds + i * 36 + 16);
                                    float rendX = MapOrigin - rawY;
                                    float rendY = MapOrigin - rawX;
                                    if (MathF.Abs(rendX - (-9154.65f)) < 150 && MathF.Abs(rendY - (-594.39f)) < 150)
                                    {
                                        _output.WriteLine($"  M2 near church #{i}: nid={nid}, uid={uid}, rend=({rendX:F2}, {rendY:F2}, {rawZ:F2})");
                                    }
                                }
                            }
                            o = ds + (int)csz;
                        }
                    }
                    CheckM2(obj0Bytes);
                    CheckM2(obj1Bytes);
                }
            }
        }
    }

    [Fact]
    public void TestHoleAlignmentAgainstModels_Deathknell()
    {
        // Chunk [5, 8] bounds: X=[1300.00, 1333.33], Y=[1933.33, 1966.67]
        // Chunk [5, 9] bounds: X=[1266.67, 1300.00], Y=[1933.33, 1966.67]
        //
        // Model placements in Deathknell:
        // Church WMO: rend=(1318.11, 1964.44, 19.04) -> inside Chunk [5, 8]
        // Church crypt stairs lead down towards X=1300..1290 (into Chunk [5, 9])
        // Tent M2: rend=(1280.19, 1961.92, 16.69) -> inside Chunk [5, 9]
        // Open Grave M2s: rend=(1280.32, 1961.35), (1278.83, 1963.54) -> inside Chunk [5, 9]

        // Chunk [5, 8] raw hole bytes: 00 00 00 00 00 00 0F 0F
        // Chunk [5, 9] raw hole bytes: 3F 3F 00 00 00 00 00 00

        byte[] h8 = new byte[] { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0F, 0x0F };
        byte[] h9 = new byte[] { 0x3F, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

        float subCell = (533.333333f / 16f) / 8f; // 4.166667 yards

        for (int invR = 0; invR <= 1; invR++)
        {
            for (int invC = 0; invC <= 1; invC++)
            {
                _output.WriteLine($"=== Testing invR={invR}, invC={invC} ===");

                // Calculate bounding box of holed cells in Chunk [5, 8]
                float minX8 = float.MaxValue, maxX8 = float.MinValue;
                float minY8 = float.MaxValue, maxY8 = float.MinValue;
                for (int r = 0; r < 8; r++)
                {
                    for (int c = 0; c < 8; c++)
                    {
                        if (((h8[r] >> c) & 1) != 0)
                        {
                            int cellY = invR == 1 ? (7 - r) : r;
                            int cellX = invC == 1 ? (7 - c) : c;

                            // In TerrainTileMeshBuilder:
                            // wx = chunk.WorldPosition.X - cellY * subCell;
                            // wy = chunk.WorldPosition.Y - cellX * subCell;
                            // Chunk [5, 8] WorldPosition: X=1333.33, Y=1966.67
                            float wx = 1333.3333f - cellY * subCell;
                            float wy = 1966.6667f - cellX * subCell;
                            minX8 = MathF.Min(minX8, wx - subCell); maxX8 = MathF.Max(maxX8, wx);
                            minY8 = MathF.Min(minY8, wy - subCell); maxY8 = MathF.Max(maxY8, wy);
                        }
                    }
                }
                _output.WriteLine($"  Chunk [5, 8] Hole Box: X=[{minX8:F2}, {maxX8:F2}], Y=[{minY8:F2}, {maxY8:F2}]");

                // Calculate bounding box of holed cells in Chunk [5, 9]
                float minX9 = float.MaxValue, maxX9 = float.MinValue;
                float minY9 = float.MaxValue, maxY9 = float.MinValue;
                for (int r = 0; r < 8; r++)
                {
                    for (int c = 0; c < 8; c++)
                    {
                        if (((h9[r] >> c) & 1) != 0)
                        {
                            int cellY = invR == 1 ? (7 - r) : r;
                            int cellX = invC == 1 ? (7 - c) : c;

                            // Chunk [5, 9] WorldPosition: X=1300.00, Y=1966.67
                            float wx = 1300.0000f - cellY * subCell;
                            float wy = 1966.6667f - cellX * subCell;
                            minX9 = MathF.Min(minX9, wx - subCell); maxX9 = MathF.Max(maxX9, wx);
                            minY9 = MathF.Min(minY9, wy - subCell); maxY9 = MathF.Max(maxY9, wy);
                        }
                    }
                }
                _output.WriteLine($"  Chunk [5, 9] Hole Box: X=[{minX9:F2}, {maxX9:F2}], Y=[{minY9:F2}, {maxY9:F2}]");

                // Check continuity at boundary X=1300:
                bool continuousAtBoundary = MathF.Abs(minX8 - maxX9) < 0.1f;
                _output.WriteLine($"  Continuous across X=1300 boundary? {continuousAtBoundary} (minX8={minX8:F2}, maxX9={maxX9:F2})");
            }
        }
    }

    [Fact]
    public void ScanDeathknellSurroundingTiles()
    {
        const string installDir = @"I:\wow12\World of Warcraft";
        if (!Directory.Exists(installDir)) return;

        var storage = CascStorage.OpenLocal(installDir, "wow_classic_beta", @"output\cache\casc", allowCdnFill: false);
        byte[]? Read(uint id) => id != 0 && storage.TryReadFile(id, out byte[]? b) == CascReadStatus.Ok ? b : null;

        byte[]? wdt = Read(775971);
        Assert.NotNull(wdt);

        int maidOffset = -1;
        for (int i = 0; i + 8 <= wdt.Length; i++)
        {
            if ((wdt[i] == 'D' && wdt[i + 1] == 'I' && wdt[i + 2] == 'A' && wdt[i + 3] == 'M') ||
                (wdt[i] == 'M' && wdt[i + 1] == 'A' && wdt[i + 2] == 'I' && wdt[i + 3] == 'D'))
            {
                maidOffset = i;
                break;
            }
        }
        Assert.True(maidOffset >= 0);

        int maidPayload = maidOffset + 8;

        for (int tx = 28; tx <= 30; tx++)
        {
            for (int ty = 27; ty <= 29; ty++)
            {
                int slot = tx * 64 + ty;
                uint rootId = BitConverter.ToUInt32(wdt, maidPayload + slot * 32);
                if (rootId == 0) continue;
                byte[]? root = Read(rootId);
                if (root == null) continue;

                int offset = 0;
                while (offset + 8 <= root.Length)
                {
                    uint id = BitConverter.ToUInt32(root, offset);
                    uint size = BitConverter.ToUInt32(root, offset + 4);
                    int ds = offset + 8;
                    if ((id == 0x4D434E4B || id == 0x4B4E434D) && ds + 128 <= root.Length)
                    {
                        uint ix = BitConverter.ToUInt32(root, ds + 4);
                        uint iy = BitConverter.ToUInt32(root, ds + 8);
                        ulong holes64 = BitConverter.ToUInt64(root, ds + 0x14);
                        ushort holes16 = BitConverter.ToUInt16(root, ds + 0x3C);
                        if (holes64 != 0 || holes16 != 0)
                        {
                            byte[] hb = new byte[8];
                            Array.Copy(root, ds + 0x14, hb, 0, 8);
                            string hex = string.Join(" ", hb.Select(b => $"{b:X2}"));
                            _output.WriteLine($"Tile [{tx}, {ty}] Chunk [{ix}, {iy}]: h64=0x{holes64:X16}, bytes={hex}");
                        }
                    }
                    offset = ds + (int)size;
                }
            }
        }
    }

    [Fact]
    public void InspectSplitAdtChunks()
    {
        const string installDir = @"I:\wow12\World of Warcraft";
        if (!Directory.Exists(installDir)) return;

        var storage = CascStorage.OpenLocal(installDir, "wow_classic_beta", @"output\cache\casc", allowCdnFill: false);
        byte[]? Read(uint id) => id != 0 && storage.TryReadFile(id, out byte[]? b) == CascReadStatus.Ok ? b : null;

        byte[]? wdt = Read(775971);
        Assert.NotNull(wdt);

        int maidOffset = -1;
        for (int i = 0; i + 8 <= wdt.Length; i++)
        {
            if ((wdt[i] == 'D' && wdt[i + 1] == 'I' && wdt[i + 2] == 'A' && wdt[i + 3] == 'M') ||
                (wdt[i] == 'M' && wdt[i + 1] == 'A' && wdt[i + 2] == 'I' && wdt[i + 3] == 'D'))
            {
                maidOffset = i;
                break;
            }
        }
        Assert.True(maidOffset >= 0);

        int maidPayload = maidOffset + 8;
        int slot = 29 * 64 + 28;
        // MAID entry has 8 uint32s: rootId, obj0Id, obj1Id, tex0Id, lodId, mapBufId, lodObj0Id, lodObj1Id
        for (int e = 0; e < 8; e++)
        {
            uint fid = BitConverter.ToUInt32(wdt, maidPayload + slot * 32 + e * 4);
            if (fid == 0) continue;
            byte[]? b = Read(fid);
            _output.WriteLine($"MAID slot {slot} entry {e}: FileDataID={fid}, length={b?.Length ?? 0}");
            if (b != null && b.Length >= 8)
            {
                int o = 0;
                var chunkNames = new System.Collections.Generic.HashSet<string>();
                while (o + 8 <= b.Length)
                {
                    string tag = System.Text.Encoding.ASCII.GetString(b, o, 4);
                    string revTag = new string(tag.Reverse().ToArray());
                    chunkNames.Add($"{tag}/{revTag}");
                    uint sz = BitConverter.ToUInt32(b, o + 4);
                    if (sz > b.Length - o - 8) break;
                    o += 8 + (int)sz;
                }
                _output.WriteLine($"  Chunks in file {fid}: {string.Join(", ", chunkNames)}");
            }
        }
    }

    [Fact]
    public void DumpTile28_28Placements()
    {
        const string installDir = @"I:\wow12\World of Warcraft";
        if (!Directory.Exists(installDir)) return;

        var storage = CascStorage.OpenLocal(installDir, "wow_classic_beta", @"output\cache\casc", allowCdnFill: false);
        byte[]? Read(uint id) => id != 0 && storage.TryReadFile(id, out byte[]? b) == CascReadStatus.Ok ? b : null;

        byte[]? wdt = Read(775971);
        Assert.NotNull(wdt);

        int maidOffset = -1;
        for (int i = 0; i + 8 <= wdt.Length; i++)
        {
            if ((wdt[i] == 'D' && wdt[i + 1] == 'I' && wdt[i + 2] == 'A' && wdt[i + 3] == 'M') ||
                (wdt[i] == 'M' && wdt[i + 1] == 'A' && wdt[i + 2] == 'I' && wdt[i + 3] == 'D'))
            {
                maidOffset = i;
                break;
            }
        }
        Assert.True(maidOffset >= 0);

        int maidPayload = maidOffset + 8;
        int slot = 28 * 64 + 28;
        uint rootId = BitConverter.ToUInt32(wdt, maidPayload + slot * 32);
        uint obj0Id = BitConverter.ToUInt32(wdt, maidPayload + slot * 32 + 4);
        uint obj1Id = BitConverter.ToUInt32(wdt, maidPayload + slot * 32 + 8);
        byte[]? rootBytes = Read(rootId);
        byte[]? obj0Bytes = Read(obj0Id);
        byte[]? obj1Bytes = Read(obj1Id);

        _output.WriteLine($"Tile [28, 28]: rootId={rootId}, obj0Id={obj0Id}, obj1Id={obj1Id}");
        if (obj0Bytes != null) DumpModf(obj0Bytes);
        if (obj1Bytes != null) DumpModf(obj1Bytes);
    }

    private void SolveHoleMapping(byte[] wdt, int maidPayload, Func<uint, byte[]?> Read)
    {
        // Test on Tile [52, 38] and Tile [31, 32] which have extensive holes spanning multiple chunks
        int[] testSlots = new int[] { 52 * 64 + 38, 31 * 64 + 32, 29 * 64 + 28 };

        foreach (int slot in testSlots)
        {
            uint rid = BitConverter.ToUInt32(wdt, maidPayload + slot * 32);
            if (rid == 0) continue;
            byte[]? rb = Read(rid);
            if (rb == null) continue;

            byte[,][] chunkHoles = new byte[16, 16][];
            int offset = 0;
            while (offset + 8 <= rb.Length)
            {
                uint id = BitConverter.ToUInt32(rb, offset);
                uint size = BitConverter.ToUInt32(rb, offset + 4);
                int dataStart = offset + 8;
                if ((id == 0x4D434E4B || id == 0x4B4E434D) && dataStart + 128 <= rb.Length)
                {
                    uint ix = BitConverter.ToUInt32(rb, dataStart + 4);
                    uint iy = BitConverter.ToUInt32(rb, dataStart + 8);
                    byte[] h = new byte[8];
                    Array.Copy(rb, dataStart + 0x14, h, 0, 8);
                    chunkHoles[ix, iy] = h;
                }
                offset = dataStart + (int)size;
            }

            _output.WriteLine($"--- Solving for Tile slot {slot} (row={slot/64}, col={slot%64}) ---");

            // Test 16 combinations:
            // swapChunkAxes (bool): true = chunkRow=ix, chunkCol=iy; false = chunkRow=iy, chunkCol=ix
            // swapCellAxes (bool): true = cellRow=c, cellCol=r; false = cellRow=r, cellCol=c
            // invertRow (bool)
            // invertCol (bool)
            for (int swapChunk = 0; swapChunk <= 1; swapChunk++)
            {
                for (int swapCell = 0; swapCell <= 1; swapCell++)
                {
                    for (int invR = 0; invR <= 1; invR++)
                    {
                        for (int invC = 0; invC <= 1; invC++)
                        {
                            bool[,] grid = new bool[128, 128];
                            for (int ix = 0; ix < 16; ix++)
                            {
                                for (int iy = 0; iy < 16; iy++)
                                {
                                    byte[] h = chunkHoles[ix, iy];
                                    if (h == null) continue;

                                    int cRow = swapChunk == 1 ? ix : iy;
                                    int cCol = swapChunk == 1 ? iy : ix;

                                    for (int r = 0; r < 8; r++)
                                    {
                                        for (int c = 0; c < 8; c++)
                                        {
                                            bool isHole = ((h[r] >> c) & 1) != 0;
                                            if (!isHole) continue;

                                            int effR = invR == 1 ? (7 - r) : r;
                                            int effC = invC == 1 ? (7 - c) : c;

                                            int cellRow = swapCell == 1 ? effC : effR;
                                            int cellCol = swapCell == 1 ? effR : effC;

                                            int gR = cRow * 8 + cellRow;
                                            int gC = cCol * 8 + cellCol;
                                            grid[gR, gC] = true;
                                        }
                                    }
                                }
                            }

                            // Evaluate seam continuity:
                            int seamMatches = 0;
                            int totalBorderHoles = 0;

                            // Horizontal seams (between rows 8*k-1 and 8*k)
                            for (int k = 1; k < 16; k++)
                            {
                                int rTop = k * 8 - 1;
                                int rBot = k * 8;
                                for (int col = 0; col < 128; col++)
                                {
                                    if (grid[rTop, col] || grid[rBot, col])
                                    {
                                        totalBorderHoles++;
                                        if (grid[rTop, col] && grid[rBot, col])
                                            seamMatches++;
                                    }
                                }
                            }

                            // Vertical seams (between cols 8*k-1 and 8*k)
                            for (int k = 1; k < 16; k++)
                            {
                                int cLeft = k * 8 - 1;
                                int cRight = k * 8;
                                for (int row = 0; row < 128; row++)
                                {
                                    if (grid[row, cLeft] || grid[row, cRight])
                                    {
                                        totalBorderHoles++;
                                        if (grid[row, cLeft] && grid[row, cRight])
                                            seamMatches++;
                                    }
                                }
                            }

                            if (seamMatches > 0)
                            {
                                _output.WriteLine($"  swapChunk={swapChunk}, swapCell={swapCell}, invR={invR}, invC={invC} => matches={seamMatches} / {totalBorderHoles} ({(float)seamMatches*100f/totalBorderHoles:F1}%)");
                            }
                        }
                    }
                }
            }
        }
    }

    private void AnalyzeTileHoles(byte[] rootBytes, byte[]? objBytes, int tx, int ty)
    {
        // Find MCNK chunks in root
        int offset = 0;
        int chunkIdx = 0;
        while (offset + 8 <= rootBytes.Length)
        {
            uint id = BitConverter.ToUInt32(rootBytes, offset);
            uint size = BitConverter.ToUInt32(rootBytes, offset + 4);
            int dataStart = offset + 8;
            if (id == 0x4D434E4B || id == 0x4B4E434D) // MCNK
            {
                if (dataStart + 128 <= rootBytes.Length)
                {
                    uint flags = BitConverter.ToUInt32(rootBytes, dataStart);
                    uint ix = BitConverter.ToUInt32(rootBytes, dataStart + 4);
                    uint iy = BitConverter.ToUInt32(rootBytes, dataStart + 8);
                    ulong holes64 = BitConverter.ToUInt64(rootBytes, dataStart + 0x14);
                    ushort holes16 = BitConverter.ToUInt16(rootBytes, dataStart + 0x3C);
                    float baseZ = BitConverter.ToSingle(rootBytes, dataStart + 0x70);
                    float posX = BitConverter.ToSingle(rootBytes, dataStart + 0x74);
                    float posY = BitConverter.ToSingle(rootBytes, dataStart + 0x78);

                    if (holes64 != 0 || holes16 != 0)
                    {
                        byte[] holeBytes = new byte[8];
                        Array.Copy(rootBytes, dataStart + 0x14, holeBytes, 0, 8);
                        string byteHex = string.Join(" ", holeBytes.Select(b => $"{b:X2}"));
                        _output.WriteLine($"    Chunk [{ix}, {iy}] (idx={chunkIdx}): flags=0x{flags:X8}, baseZ={baseZ}, pos=({posX}, {posY}), holes16=0x{holes16:X4}, holes64=0x{holes64:X16}");
                        _output.WriteLine($"      Hole Bytes: {byteHex}");

                        // Print grid of bits as byte[row] >> col
                        for (int r = 0; r < 8; r++)
                        {
                            string rowBits = "";
                            for (int c = 0; c < 8; c++)
                            {
                                int bit = (holeBytes[r] >> c) & 1;
                                rowBits += bit == 1 ? "X" : ".";
                            }
                            _output.WriteLine($"      r{r}: {rowBits} (byte=0x{holeBytes[r]:X2})");
                        }
                    }
                }
                chunkIdx++;
            }
            offset = dataStart + (int)size;
        }
    }

    private void DumpModf(byte[] objBytes)
    {
        int offset = 0;
        while (offset + 8 <= objBytes.Length)
        {
            uint id = BitConverter.ToUInt32(objBytes, offset);
            uint size = BitConverter.ToUInt32(objBytes, offset + 4);
            int dataStart = offset + 8;
            string fourCc = System.Text.Encoding.ASCII.GetString(objBytes, offset, 4);
            _output.WriteLine($"  obj chunk: {fourCc} (size={size})");
            if (fourCc == "DMLM")
            {
                int count = (int)size / 40;
                for (int i = 0; i < count; i++)
                {
                    int e = dataStart + i * 40;
                    uint nameId = BitConverter.ToUInt32(objBytes, e);
                    uint uniqueId = BitConverter.ToUInt32(objBytes, e + 4);
                    float rawX = BitConverter.ToSingle(objBytes, e + 8);
                    float rawZ = BitConverter.ToSingle(objBytes, e + 12);
                    float rawY = BitConverter.ToSingle(objBytes, e + 16);
                    float rendX = MapOrigin - rawY;
                    float rendY = MapOrigin - rawX;
                    _output.WriteLine($"    MLMD WMO #{i}: nameId={nameId}, uniqueId={uniqueId}, raw=({rawX:F2}, {rawY:F2}, {rawZ:F2}), rend=({rendX:F2}, {rendY:F2}, {rawZ:F2})");
                }
            }
            if ((id == 0x46444F4D || id == 0x4D4F4446) && dataStart + 64 <= objBytes.Length)
            {
                int count = (int)size / 64;
                for (int i = 0; i < count; i++)
                {
                    int e = dataStart + i * 64;
                    uint nameId = BitConverter.ToUInt32(objBytes, e);
                    uint uniqueId = BitConverter.ToUInt32(objBytes, e + 4);
                    float rawX = BitConverter.ToSingle(objBytes, e + 8);
                    float rawZ = BitConverter.ToSingle(objBytes, e + 12);
                    float rawY = BitConverter.ToSingle(objBytes, e + 16);
                    float rendX = MapOrigin - rawY;
                    float rendY = MapOrigin - rawX;
                    float rotX = BitConverter.ToSingle(objBytes, e + 20);
                    float rotZ = BitConverter.ToSingle(objBytes, e + 24);
                    float rotY = BitConverter.ToSingle(objBytes, e + 28);
                    float bbMinX = BitConverter.ToSingle(objBytes, e + 32);
                    float bbMinZ = BitConverter.ToSingle(objBytes, e + 36);
                    float bbMinY = BitConverter.ToSingle(objBytes, e + 40);
                    float bbMaxX = BitConverter.ToSingle(objBytes, e + 44);
                    float bbMaxZ = BitConverter.ToSingle(objBytes, e + 48);
                    float bbMaxY = BitConverter.ToSingle(objBytes, e + 52);
                    _output.WriteLine($"  MODF #{i}: nameId={nameId}, uniqueId={uniqueId}, rend=({rendX:F2}, {rendY:F2}, {rawZ:F2}), rot=({rotX:F1},{rotY:F1},{rotZ:F1})");
                    _output.WriteLine($"    bbRawMin=({bbMinX:F2},{bbMinY:F2},{bbMinZ:F2}), bbRawMax=({bbMaxX:F2},{bbMaxY:F2},{bbMaxZ:F2})");
                    _output.WriteLine($"    bbRendX=[{MapOrigin - bbMaxY:F2}, {MapOrigin - bbMinY:F2}], bbRendY=[{MapOrigin - bbMaxX:F2}, {MapOrigin - bbMinX:F2}]");
                }
            }
            offset = dataStart + (int)size;
        }
    }

    private void FindPlacementsNearChunks(byte[] rootBytes, byte[]? obj0Bytes, byte[]? obj1Bytes, int tileX, int tileY)
    {
        float chunkSmall = ChunkSize / 16f;
        float subCellSize = chunkSmall / 8f;

        // Print chunks [5, 8] and [5, 9] bounds
        for (int cy = 8; cy <= 9; cy++)
        {
            int cx = 5;
            float wx = MapOrigin - tileX * ChunkSize - cy * chunkSmall;
            float wy = MapOrigin - tileY * ChunkSize - cx * chunkSmall;
            _output.WriteLine($"Chunk [{cx}, {cy}]: X=[{wx - chunkSmall:F2}, {wx:F2}], Y=[{wy - chunkSmall:F2}, {wy:F2}]");
        }

        if (obj1Bytes != null)
        {
            _output.WriteLine("WMO Placements near chunks:");
            DumpNearbyWmo(obj1Bytes, tileX, tileY);
        }
        if (obj0Bytes != null)
        {
            _output.WriteLine("M2 Placements near chunks:");
            DumpNearbyM2(obj0Bytes, tileX, tileY);
        }
    }

    private void DumpNearbyWmo(byte[] objBytes, int tileX, int tileY)
    {
        int offset = 0;
        while (offset + 8 <= objBytes.Length)
        {
            uint id = BitConverter.ToUInt32(objBytes, offset);
            uint size = BitConverter.ToUInt32(objBytes, offset + 4);
            int dataStart = offset + 8;
            string fourCc = System.Text.Encoding.ASCII.GetString(objBytes, offset, 4);
            if (fourCc == "DMLM")
            {
                int count = (int)size / 40;
                for (int i = 0; i < count; i++)
                {
                    int e = dataStart + i * 40;
                    uint nameId = BitConverter.ToUInt32(objBytes, e);
                    uint uniqueId = BitConverter.ToUInt32(objBytes, e + 4);
                    float rawX = BitConverter.ToSingle(objBytes, e + 8);
                    float rawZ = BitConverter.ToSingle(objBytes, e + 12);
                    float rawY = BitConverter.ToSingle(objBytes, e + 16);
                    float rendX = MapOrigin - rawY;
                    float rendY = MapOrigin - rawX;
                    _output.WriteLine($"    MLMD WMO #{i}: nameId={nameId}, uniqueId={uniqueId}, rend=({rendX:F2}, {rendY:F2}, {rawZ:F2})");
                }
            }
            offset = dataStart + (int)size;
        }
    }

    private void DumpNearbyM2(byte[] objBytes, int tileX, int tileY)
    {
        int offset = 0;
        while (offset + 8 <= objBytes.Length)
        {
            uint id = BitConverter.ToUInt32(objBytes, offset);
            uint size = BitConverter.ToUInt32(objBytes, offset + 4);
            int dataStart = offset + 8;
            if ((id == 0x4644444D || id == 0x4D444446) && dataStart + 36 <= objBytes.Length)
            {
                int count = (int)size / 36;
                for (int i = 0; i < count; i++)
                {
                    int e = dataStart + i * 36;
                    uint nameId = BitConverter.ToUInt32(objBytes, e);
                    uint uniqueId = BitConverter.ToUInt32(objBytes, e + 4);
                    float rawX = BitConverter.ToSingle(objBytes, e + 8);
                    float rawZ = BitConverter.ToSingle(objBytes, e + 12);
                    float rawY = BitConverter.ToSingle(objBytes, e + 16);
                    float rendX = MapOrigin - rawY;
                    float rendY = MapOrigin - rawX;
                    // Check if within chunks [5, 8] or [5, 9] (wx around 1266..1333, wy around 1933..1966)
                    if (rendX >= 1250 && rendX <= 1350 && rendY >= 1900 && rendY <= 2000)
                    {
                        _output.WriteLine($"    MDDF M2 #{i}: nameId={nameId}, uniqueId={uniqueId}, rend=({rendX:F2}, {rendY:F2}, {rawZ:F2})");
                    }
                }
            }
            offset = dataStart + (int)size;
        }
    }

    [Fact]
    public void FindAndInspectModernM2Cameras()
    {
        const string installDir = @"I:\wow12\World of Warcraft";
        if (!Directory.Exists(installDir)) return;

        var storage = CascStorage.OpenLocal(installDir, "wow_classic_beta", @"output\cache\casc", allowCdnFill: false);
        var fdids = storage.GetAvailableFileDataIds();
        _output.WriteLine($"Total available FileDataIDs: {fdids.Count}");

        int m2Count = 0;
        int m2WithCameras = 0;

        foreach (uint id in fdids)
        {
            if (storage.TryReadFile(id, out byte[]? data) != CascReadStatus.Ok || data == null || data.Length < 0x120)
                continue;

            uint magic = BitConverter.ToUInt32(data, 0);
            byte[]? md20 = null;
            if (magic == 0x3032444D) // MD20
            {
                md20 = data;
            }
            else if (magic == 0x3132444D && data.Length >= 16) // MD21
            {
                uint chunkSize = BitConverter.ToUInt32(data, 4);
                uint innerMagic = BitConverter.ToUInt32(data, 8);
                if (innerMagic == 0x3032444D)
                {
                    md20 = new byte[Math.Min(chunkSize, (uint)(data.Length - 8))];
                    Array.Copy(data, 8, md20, 0, md20.Length);
                }
            }

            if (md20 == null || md20.Length < 0x120)
                continue;

            m2Count++;
            uint version = BitConverter.ToUInt32(md20, 4);
            uint nCameras = BitConverter.ToUInt32(md20, 0x110);
            uint ofsCameras = BitConverter.ToUInt32(md20, 0x114);

            if (nCameras > 0 && nCameras < 100 && ofsCameras < md20.Length)
            {
                m2WithCameras++;
                _output.WriteLine($"Found M2 with cameras! FDID={id}, isChunked={magic == 0x3132444D}, version={version}, nCameras={nCameras}, ofsCameras=0x{ofsCameras:X}");

                // Try reading with M2ModelReader
                try
                {
                    using MemoryStream ms = new(md20);
                    var doc = WowViewer.Core.IO.M2.M2ModelReader.Read(ms, $"fdid_{id}.m2");
                    _output.WriteLine($"  M2ModelReader success! Doc cameras: {doc.Cameras.Count}");
                    foreach (var cam in doc.Cameras)
                    {
                        _output.WriteLine($"    Cam[{cam.Index}]: type={cam.Type}, near={cam.NearClip}, far={cam.FarClip}, hasAnimFov={cam.HasAnimatedFieldOfView}, posTimestamps={cam.PositionTrack.TimestampArray.Count}, targetTimestamps={cam.TargetPositionTrack.TimestampArray.Count}");
                    }

                    // Test M2CameraPathImporter
                    var path = WowViewer.Core.Runtime.M2.M2CameraPathImporter.Import(doc);
                    _output.WriteLine($"  M2CameraPathImporter success! Keyframes={path.Keyframes.Count}, duration={path.DurationMs}ms");
                }
                catch (Exception ex)
                {
                    _output.WriteLine($"  M2ModelReader failed: {ex.Message}");
                }

                if (m2WithCameras >= 10) break;
            }
        }

        _output.WriteLine($"Scanned {m2Count} M2s, found {m2WithCameras} with cameras.");
    }

    [Fact]
    public void InspectCinematicCameraDb2()
    {
        const string installDir = @"I:\wow12\World of Warcraft";
        if (!Directory.Exists(installDir)) return;

        var storage = CascStorage.OpenLocal(installDir, "wow_classic_beta", @"output\cache\casc", allowCdnFill: false);
        
        // Find DB2 for CinematicCamera
        // Let's test reading DBC/DB2 files from CASC
        var fdids = storage.GetAvailableFileDataIds();
        _output.WriteLine($"Checking for DBC/DB2 or specific camera FDIDs in {fdids.Count} files...");

        // Known FDID 116902: What is it?
        // Let's check FDID 116902 to 116915 (which we found earlier to be cameras!)
        for (uint id = 116902; id <= 116912; id++)
        {
            if (storage.TryReadFile(id, out byte[]? data) == CascReadStatus.Ok && data != null)
            {
                byte[] md20 = WowViewer.Core.IO.M2.M2ChunkedFileIds.GetMd20Payload(data);
                using MemoryStream ms = new(md20);
                var doc = WowViewer.Core.IO.M2.M2ModelReader.Read(ms, $"fdid_{id}.m2");
                _output.WriteLine($"Camera FDID {id}: Name='{doc.ModelName}', Cameras={doc.Cameras.Count}, Seq={doc.Sequences.Count}");
                if (doc.Sequences.Count > 0)
                {
                    _output.WriteLine($"  Seq[0]: id={doc.Sequences[0].AnimationId}, duration={doc.Sequences[0].Duration}ms");
                }
            }
        }

#if false
        // Now test DBCD loading of CinematicCamera on wow_classic_beta
        try
        {
            var dbcd = new DBCD.DBCD(new CascDbcProvider(storage), new FilesystemDBDProvider(@"output\dbd"));
            var db = dbcd.Load("CinematicCamera", "1.60.1.70205");
            _output.WriteLine($"Loaded CinematicCamera.db2! Rows: {db.Values.Count}");
            foreach (var row in db.Values)
            {
                dynamic r = row;
                _output.WriteLine($"  Row ID={r.ID}: FileDataID={TryGet(r, "FileDataID")}, Model={TryGet(r, "Model")}, SoundID={TryGet(r, "SoundID")}");
            }
        }
        catch (Exception ex)
        {
            _output.WriteLine($"Failed to load CinematicCamera with DBCD: {ex.Message}");
        }

        static object? TryGet(dynamic r, string col)
        {
            try { return r[col]; } catch { return null; }
        }
#endif
    }
}
