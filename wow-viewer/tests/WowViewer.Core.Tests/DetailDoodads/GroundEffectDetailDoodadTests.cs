using System.Text;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests.DetailDoodads;

public sealed class GroundEffectDetailDoodadTests
{
    [Fact]
    public void GroundEffectLookup_ParsesFlagsDensityAndModelsCorrectly()
    {
        // Setup:
        // Doodad 1: 3.x style with AlignToNormal (0x1) and animScale 1.5
        // Doodad 2: 3.x style with IgnoreMCCV (0x2) and pushScale 0.8
        // Doodad 3: Modern FileDataId style (123456)
        byte[] doodadDbc = BuildDbcWithStrings(
            fieldCount: 5,
            rows:
            [
                (stringOffsets) => [101u, stringOffsets["World\\Plants\\Bush01.m2"], 0x1u, BitConverter.SingleToUInt32Bits(1.5f), BitConverter.SingleToUInt32Bits(1.0f)],
                (stringOffsets) => [102u, stringOffsets["World\\Plants\\Fern02.m2"], 0x2u, BitConverter.SingleToUInt32Bits(1.0f), BitConverter.SingleToUInt32Bits(0.8f)],
                (stringOffsets) => [103u, 123456u, 0x3u, BitConverter.SingleToUInt32Bits(1.2f), BitConverter.SingleToUInt32Bits(1.0f)],
            ],
            stringBlockEntries: ["World\\Plants\\Bush01.m2", "World\\Plants\\Fern02.m2"]);

        FakeArchiveReader archiveReader = new(
            new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase)
            {
                ["DBFilesClient\\GroundEffectDoodad.dbc"] = doodadDbc,
                ["DBFilesClient\\GroundEffectTexture.dbc"] = BuildDbc(
                    fieldCount: 7,
                    rows:
                    [
                        [201u, 101u, 102u, 103u, 0u, 16u, 45u],
                    ],
                    stringBlockEntries: []),
            });


        GroundEffectLookup lookup = new();
        lookup.Load(Array.Empty<string>(), archiveReader);

        Assert.True(lookup.IsLoaded);

        // Check Doodad 101 (AlignToNormal)
        GroundEffectDoodadRecord? d101 = lookup.GetDoodadRecord(101);
        Assert.NotNull(d101);
        Assert.Equal("World\\Plants\\Bush01.m2", d101.ModelPath);
        Assert.True(d101.AlignToNormal);
        Assert.False(d101.IgnoreMCCV);
        Assert.Equal(1.5f, d101.AnimScale);

        // Check Doodad 102 (IgnoreMCCV)
        GroundEffectDoodadRecord? d102 = lookup.GetDoodadRecord(102);
        Assert.NotNull(d102);
        Assert.Equal("World\\Plants\\Fern02.m2", d102.ModelPath);
        Assert.False(d102.AlignToNormal);
        Assert.True(d102.IgnoreMCCV);
        Assert.Equal(0.8f, d102.PushScale);

        // Check Doodad 103 (FileDataID)
        GroundEffectDoodadRecord? d103 = lookup.GetDoodadRecord(103);
        Assert.NotNull(d103);
        Assert.Equal(123456u, d103.FileDataId);
        Assert.True(d103.AlignToNormal);
        Assert.True(d103.IgnoreMCCV);

        // Check Texture Record 201
        GroundEffectTextureRecord? tex = lookup.GetTextureRecord(201);
        Assert.NotNull(tex);
        Assert.Equal(16u, tex.Density);
        Assert.Equal(45u, tex.SoundId);
        Assert.Equal(3, tex.DoodadIds.Count);

        // Check GetDoodadRecordsForEffect
        IReadOnlyList<GroundEffectDoodadRecord> records = lookup.GetDoodadRecordsForEffect(201);
        Assert.Equal(3, records.Count);
        Assert.Equal(101u, records[0].Id);
        Assert.Equal(102u, records[1].Id);
        Assert.Equal(103u, records[2].Id);
    }

    [Fact]
    public void WmoMddlReader_ParsesLayersAndGroupDataRleCorrectly()
    {
        // Synthesize MDDL payload:
        // float minTriangleArea = 0.5f (4 bytes)
        // ushort layerCount = 1 (2 bytes)
        // Layer 0: density = 24 (1 byte), doodadCount = 2 (1 byte)
        //   Doodad 0: doodadId = 5001 (4 bytes), weight = 10 (1 byte)
        //   Doodad 1: doodadId = 5002 (4 bytes), weight = 20 (1 byte)
        // Group 0 data: groupIndex = 3 (2 bytes), dataSize = 14 (4 bytes)
        //   RLE payload:
        //     layer_index = 0 (2 bytes)
        //     batch_index = 0x8005 (2 bytes) -> rollAll = true for batch 5
        //     batch_index = 2 (2 bytes)
        //     locrange_index_part = 0x81 (1 byte) -> locrange_index = 1, single_loc = true
        //     loc_part = 4 (1 byte) -> loc = 4
        //     loc_part = 0xFF (1 byte) -> end locs for this part
        //     locrange_index_part = 0xFF (1 byte) -> end loc ranges for this batch
        //     batch_index = 0xFFFF (2 bytes) -> end batches for layer 0
        //     layer_index = 0xFFFF (2 bytes) -> end of group data

        using MemoryStream ms = new();
        using BinaryWriter w = new(ms);

        w.Write(0.5f); // minTriangleArea
        w.Write((ushort)1); // layerCount

        // Layer 0:
        w.Write((byte)24); // density
        w.Write((byte)2);  // doodadCount
        w.Write(5001u);    // doodad 1 id
        w.Write((byte)10); // weight
        w.Write(5002u);    // doodad 2 id
        w.Write((byte)20); // weight

        // Group data for group 3:
        using MemoryStream rleStream = new();
        using BinaryWriter rw = new(rleStream);
        rw.Write((ushort)0); // layer 0
        rw.Write((ushort)0x8005); // roll all for batch 5
        rw.Write((ushort)2); // batch 2
        rw.Write((byte)0x81); // locrange part: locrange = 1, singleLoc = true
        rw.Write((byte)4); // loc = 4
        rw.Write((byte)0xFF); // end loc parts
        rw.Write((byte)0xFF); // end loc range parts
        rw.Write((ushort)0xFFFF); // end batches
        rw.Write((ushort)0xFFFF); // end layers

        byte[] rleBytes = rleStream.ToArray();
        w.Write((ushort)3); // groupIndex 3
        w.Write((uint)rleBytes.Length); // dataSize
        w.Write(rleBytes);

        byte[] mddlPayload = ms.ToArray();

        WmoDetailDoodadDocument doc = WmoMddlReader.Read(mddlPayload);

        Assert.Equal(0.5f, doc.MinTriangleArea);
        Assert.Single(doc.Layers);
        Assert.Equal(24, doc.Layers[0].Density);
        Assert.Equal(2, doc.Layers[0].Doodads.Count);
        Assert.Equal(5001u, doc.Layers[0].Doodads[0].DoodadId);
        Assert.Equal(10, doc.Layers[0].Doodads[0].Weight);
        Assert.Equal(5002u, doc.Layers[0].Doodads[1].DoodadId);
        Assert.Equal(20, doc.Layers[0].Doodads[1].Weight);

        Assert.Single(doc.Groups);
        Assert.Equal(3, doc.Groups[0].GroupIndex);
        Assert.Equal(2, doc.Groups[0].Commands.Count);

        // Command 0: batch 5 rollAll
        Assert.Equal(0, doc.Groups[0].Commands[0].LayerIndex);
        Assert.Equal(5, doc.Groups[0].Commands[0].BatchIndex);
        Assert.True(doc.Groups[0].Commands[0].RollAllLocations);

        // Command 1: batch 2 loc range
        Assert.Equal(0, doc.Groups[0].Commands[1].LayerIndex);
        Assert.Equal(2, doc.Groups[0].Commands[1].BatchIndex);
        Assert.False(doc.Groups[0].Commands[1].RollAllLocations);
        Assert.Equal(1, doc.Groups[0].Commands[1].LocRangeIndex);
        Assert.True(doc.Groups[0].Commands[1].SingleLocation);
        Assert.Equal([4], doc.Groups[0].Commands[1].Locations);
    }

    private static byte[] BuildDbc(uint fieldCount, IReadOnlyList<uint[]> rows, IReadOnlyList<string> stringBlockEntries)
    {
        using MemoryStream stringStream = new();
        stringStream.WriteByte(0);

        foreach (string entry in stringBlockEntries)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(entry);
            stringStream.Write(bytes, 0, bytes.Length);
            stringStream.WriteByte(0);
        }

        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(0x43424457u);
        writer.Write(checked((uint)rows.Count));
        writer.Write(fieldCount);
        writer.Write(fieldCount * 4u);
        writer.Write(checked((uint)stringStream.Length));

        foreach (uint[] row in rows)
        {
            Assert.Equal((int)fieldCount, row.Length);
            foreach (uint value in row)
                writer.Write(value);
        }

        stringStream.Position = 0;
        stringStream.CopyTo(stream);
        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] BuildDbcWithStrings(
        uint fieldCount,
        IReadOnlyList<Func<IReadOnlyDictionary<string, uint>, uint[]>> rows,
        IReadOnlyList<string> stringBlockEntries)
    {
        using MemoryStream stringStream = new();
        stringStream.WriteByte(0);
        Dictionary<string, uint> offsets = [];

        foreach (string entry in stringBlockEntries)
        {
            offsets[entry] = (uint)stringStream.Position;
            byte[] bytes = Encoding.UTF8.GetBytes(entry);
            stringStream.Write(bytes, 0, bytes.Length);
            stringStream.WriteByte(0);
        }

        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(0x43424457u);
        writer.Write(checked((uint)rows.Count));
        writer.Write(fieldCount);
        writer.Write(fieldCount * 4u);
        writer.Write(checked((uint)stringStream.Length));

        foreach (var rowFunc in rows)
        {
            uint[] row = rowFunc(offsets);
            Assert.Equal((int)fieldCount, row.Length);
            foreach (uint value in row)
                writer.Write(value);
        }

        stringStream.Position = 0;
        stringStream.CopyTo(stream);
        writer.Flush();
        return stream.ToArray();
    }


    private sealed class FakeArchiveReader : IArchiveReader
    {
        private readonly IReadOnlyDictionary<string, byte[]> _files;

        public FakeArchiveReader(IReadOnlyDictionary<string, byte[]> files)
        {
            _files = files;
        }

        public bool FileExists(string virtualPath) => _files.ContainsKey(virtualPath);

        public byte[]? ReadFile(string virtualPath)
        {
            return _files.TryGetValue(virtualPath, out byte[]? data) ? data : null;
        }
    }
}

