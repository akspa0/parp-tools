using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Phys;
using WowViewer.Core.Phys;

namespace WowViewer.Core.Tests;

/// <summary>
/// Spec 214 — <c>.phys</c> sidecar path resolution and container reading.
/// </summary>
/// <remarks>
/// These are synthetic bytes. They prove the reader is <em>safe</em> — that it fails closed, skips
/// what the client skips, and preserves what the client does not interpret. They do not prove
/// <em>fidelity</em> to real client assets; that requires the real-client manifest (task T002).
/// </remarks>
public sealed class PhysSidecarReaderTests
{
    [Theory]
    [InlineData("World\\Model.m2", "World\\Model.phys")]
    [InlineData("World/Model.mdx", "World/Model.phys")]
    [InlineData("Model.M2", "Model.phys")]
    [InlineData("Model", "Model.phys")]
    public void SidecarPath_ReplacesExtension(string modelPath, string expected)
    {
        Assert.True(PhysSidecarPath.TryResolve(modelPath, out string sidecarPath));
        Assert.Equal(expected, sidecarPath);
    }

    [Fact]
    public void SidecarPath_DoesNotTreatDirectoryDotAsExtension()
    {
        Assert.True(PhysSidecarPath.TryResolve("World\\v1.0\\Model", out string sidecarPath));
        Assert.Equal("World\\v1.0\\Model.phys", sidecarPath);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void SidecarPath_RejectsBlankInput(string? modelPath)
    {
        Assert.False(PhysSidecarPath.TryResolve(modelPath, out string sidecarPath));
        Assert.Equal(string.Empty, sidecarPath);
    }

    [Fact]
    public void SidecarPath_RejectsPathBeyondTheClientBuffer()
    {
        string longPath = new string('a', PhysSidecarPath.ClientPathBufferLength) + ".m2";

        Assert.False(PhysSidecarPath.TryResolve(longPath, out _));
    }

    [Fact]
    public void Reader_RejectsWrongMagic()
    {
        byte[] data = new PhysBuilder().WithMagic("XXXX").Build();

        Assert.False(PhysReader.TryRead(data, out PhysDocument document));
        Assert.Contains(document.Diagnostics, d => d.Code == "phys.bad-magic");
    }

    [Fact]
    public void Reader_RejectsUnsupportedVersion()
    {
        byte[] data = new PhysBuilder().WithVersion(1).Build();

        Assert.False(PhysReader.TryRead(data, out PhysDocument document));
        Assert.Contains(document.Diagnostics, d => d.Code == "phys.unsupported-version");
    }

    [Fact]
    public void Reader_RejectsTruncatedFile()
    {
        Assert.False(PhysReader.TryRead([0x53, 0x59], out PhysDocument document));
        Assert.Contains(document.Diagnostics, d => d.Code == "phys.truncated-header");
    }

    [Fact]
    public void Reader_SkipsUnknownChunkBySizeAndKeepsWalking()
    {
        // The client skips unrecognised tags via the size field. A reader that rejected them would be
        // stricter than 5.0.1 and would break on later-era files.
        byte[] data = new PhysBuilder()
            .WithChunk("ZZZZ", new byte[13])
            .WithChunk("SPHS", SphereRecord(new Vector3(1f, 2f, 3f), 4f))
            .Build();

        Assert.True(PhysReader.TryRead(data, out PhysDocument document));
        Assert.Contains(document.Diagnostics, d => d.Code == "phys.unknown-chunk");

        PhysSphereShape sphere = Assert.Single(document.SphereShapes);
        Assert.Equal(new Vector3(1f, 2f, 3f), sphere.Center);
        Assert.Equal(4f, sphere.Radius);
    }

    [Fact]
    public void Reader_ReportsTruncatedChunkAndStopsWithoutThrowing()
    {
        byte[] data = new PhysBuilder().WithChunk("SPHS", new byte[16]).Build();
        // Declare more payload than the file actually carries.
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(data.Length - 16 - 4), 999u);

        Assert.True(PhysReader.TryRead(data, out PhysDocument document));
        Assert.Contains(document.Diagnostics, d => d.Code == "phys.truncated-chunk");
        Assert.Empty(document.SphereShapes);
    }

    [Fact]
    public void Reader_ReportsStrideMismatchAndKeepsWholeRecords()
    {
        byte[] payload = new byte[16 + 5];
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(12), 7f);
        byte[] data = new PhysBuilder().WithChunk("SPHS", payload).Build();

        Assert.True(PhysReader.TryRead(data, out PhysDocument document));
        Assert.Contains(document.Diagnostics, d => d.Code == "phys.stride-mismatch");
        Assert.Equal(7f, Assert.Single(document.SphereShapes).Radius);
    }

    [Fact]
    public void Reader_AbsentChunksProduceEmptyCollectionsWithoutDiagnostics()
    {
        byte[] data = new PhysBuilder().Build();

        Assert.True(PhysReader.TryRead(data, out PhysDocument document));
        Assert.Empty(document.Bodies);
        Assert.Empty(document.Shapes);
        Assert.Empty(document.Joints);
        Assert.Empty(document.Diagnostics);
        Assert.False(document.HasSimulatableContent);
    }

    [Fact]
    public void Reader_PreservesNoBoneSentinel()
    {
        byte[] record = new byte[28];
        BinaryPrimitives.WriteUInt16LittleEndian(record.AsSpan(16), PhysBody.NoBone);
        byte[] data = new PhysBuilder().WithChunk("BODY", record).Build();

        Assert.True(PhysReader.TryRead(data, out PhysDocument document));

        PhysBody body = Assert.Single(document.Bodies);
        Assert.Equal(PhysBody.NoBone, body.BoneIndex);
        Assert.False(body.HasBone);
    }

    [Fact]
    public void Reader_ReadsBodyFieldsAtMeasuredOffsets()
    {
        byte[] record = new byte[28];
        BinaryPrimitives.WriteUInt16LittleEndian(record.AsSpan(0), 1);
        BinaryPrimitives.WriteSingleLittleEndian(record.AsSpan(4), 1.5f);
        BinaryPrimitives.WriteSingleLittleEndian(record.AsSpan(8), 2.5f);
        BinaryPrimitives.WriteSingleLittleEndian(record.AsSpan(12), 3.5f);
        BinaryPrimitives.WriteUInt16LittleEndian(record.AsSpan(16), 9);
        BinaryPrimitives.WriteUInt32LittleEndian(record.AsSpan(20), 4u);
        BinaryPrimitives.WriteUInt32LittleEndian(record.AsSpan(24), 2u);
        byte[] data = new PhysBuilder().WithChunk("BODY", record).Build();

        Assert.True(PhysReader.TryRead(data, out PhysDocument document));

        PhysBody body = Assert.Single(document.Bodies);
        Assert.Equal(1, body.RawType);
        Assert.Equal(new Vector3(1.5f, 2.5f, 3.5f), body.Position);
        Assert.Equal(9, body.BoneIndex);
        Assert.True(body.HasBone);
        Assert.Equal(4u, body.FirstShapeIndex);
        Assert.Equal(2u, body.ShapeCount);
        Assert.True(document.HasSimulatableContent);
    }

    [Fact]
    public void Reader_PreservesUnverifiedBoxPrefixInsteadOfInterpretingIt()
    {
        byte[] record = new byte[60];
        for (int index = 0; index < 48; index++)
            record[index] = (byte)(index + 1);
        BinaryPrimitives.WriteSingleLittleEndian(record.AsSpan(48), 1f);
        BinaryPrimitives.WriteSingleLittleEndian(record.AsSpan(52), 2f);
        BinaryPrimitives.WriteSingleLittleEndian(record.AsSpan(56), 3f);
        byte[] data = new PhysBuilder().WithChunk("BOXS", record).Build();

        Assert.True(PhysReader.TryRead(data, out PhysDocument document));

        PhysBoxShape box = Assert.Single(document.BoxShapes);
        Assert.Equal(48, box.UnverifiedPrefix.Length);
        Assert.Equal(record.AsSpan(0, 48).ToArray(), box.UnverifiedPrefix.ToArray());
        Assert.Equal(new Vector3(1f, 2f, 3f), box.HalfExtents);
    }

    [Fact]
    public void Reader_ReadsShapeIndirectionAndFlagsUnknownKind()
    {
        byte[] known = new byte[20];
        BinaryPrimitives.WriteUInt16LittleEndian(known.AsSpan(0), (ushort)PhysShapeKind.Capsule);
        BinaryPrimitives.WriteUInt16LittleEndian(known.AsSpan(2), 3);

        byte[] unknown = new byte[20];
        BinaryPrimitives.WriteUInt16LittleEndian(unknown.AsSpan(0), 77);

        byte[] data = new PhysBuilder().WithChunk("SHAP", [.. known, .. unknown]).Build();

        Assert.True(PhysReader.TryRead(data, out PhysDocument document));
        Assert.Equal(2, document.Shapes.Count);
        Assert.Equal(PhysShapeKind.Capsule, document.Shapes[0].Kind);
        Assert.Equal(3, document.Shapes[0].Index);
        Assert.Contains(document.Diagnostics, d => d.Code == "phys.unknown-shape-kind");
    }

    [Fact]
    public void Reader_ReadsJointAndFlagsUnknownKind()
    {
        byte[] known = new byte[16];
        BinaryPrimitives.WriteUInt32LittleEndian(known.AsSpan(0), 1u);
        BinaryPrimitives.WriteUInt32LittleEndian(known.AsSpan(4), 2u);
        BinaryPrimitives.WriteUInt16LittleEndian(known.AsSpan(12), (ushort)PhysJointKind.Shoulder);
        BinaryPrimitives.WriteUInt16LittleEndian(known.AsSpan(14), 5);

        byte[] unknown = new byte[16];
        BinaryPrimitives.WriteUInt16LittleEndian(unknown.AsSpan(12), 42);

        byte[] data = new PhysBuilder().WithChunk("JOIN", [.. known, .. unknown]).Build();

        Assert.True(PhysReader.TryRead(data, out PhysDocument document));
        Assert.Equal(2, document.Joints.Count);
        Assert.Equal(1u, document.Joints[0].BodyAIndex);
        Assert.Equal(2u, document.Joints[0].BodyBIndex);
        Assert.Equal(PhysJointKind.Shoulder, document.Joints[0].Kind);
        Assert.Equal(5, document.Joints[0].Index);
        Assert.Contains(document.Diagnostics, d => d.Code == "phys.unknown-joint-kind");
    }

    [Theory]
    [InlineData("SPHJ", 28)]
    [InlineData("SHOJ", 108)]
    [InlineData("WELJ", 104)]
    public void Reader_KeepsTypedJointPayloadsOpaqueAtMeasuredStrides(string tag, int stride)
    {
        byte[] payload = new byte[stride * 2];
        payload[0] = 0xAB;
        payload[stride] = 0xCD;
        byte[] data = new PhysBuilder().WithChunk(tag, payload).Build();

        Assert.True(PhysReader.TryRead(data, out PhysDocument document));

        IReadOnlyList<PhysOpaqueRecord> records = tag switch
        {
            "SPHJ" => document.SphericalJoints,
            "SHOJ" => document.ShoulderJoints,
            _ => document.WeldJoints,
        };

        Assert.Equal(2, records.Count);
        Assert.Equal(stride, records[0].Bytes.Length);
        Assert.Equal(0xAB, records[0].Bytes.Span[0]);
        Assert.Equal(0xCD, records[1].Bytes.Span[0]);
    }

    private static byte[] SphereRecord(Vector3 center, float radius)
    {
        byte[] record = new byte[16];
        BinaryPrimitives.WriteSingleLittleEndian(record.AsSpan(0), center.X);
        BinaryPrimitives.WriteSingleLittleEndian(record.AsSpan(4), center.Y);
        BinaryPrimitives.WriteSingleLittleEndian(record.AsSpan(8), center.Z);
        BinaryPrimitives.WriteSingleLittleEndian(record.AsSpan(12), radius);
        return record;
    }

    /// <summary>Builds a synthetic container using the measured reversed-tag convention.</summary>
    private sealed class PhysBuilder
    {
        private readonly List<byte> _chunks = [];
        private string _magic = "PHYS";
        private ushort _version;

        public PhysBuilder WithMagic(string magic)
        {
            _magic = magic;
            return this;
        }

        public PhysBuilder WithVersion(ushort version)
        {
            _version = version;
            return this;
        }

        public PhysBuilder WithChunk(string tag, byte[] payload)
        {
            _chunks.AddRange(TagBytes(tag));
            byte[] size = new byte[4];
            BinaryPrimitives.WriteUInt32LittleEndian(size, (uint)payload.Length);
            _chunks.AddRange(size);
            _chunks.AddRange(payload);
            return this;
        }

        public byte[] Build()
        {
            List<byte> data = [];
            data.AddRange(TagBytes(_magic));

            byte[] headerPayload = new byte[4];
            BinaryPrimitives.WriteUInt16LittleEndian(headerPayload, _version);

            byte[] headerSize = new byte[4];
            BinaryPrimitives.WriteUInt32LittleEndian(headerSize, (uint)headerPayload.Length);

            data.AddRange(headerSize);
            data.AddRange(headerPayload);
            data.AddRange(_chunks);
            return [.. data];
        }

        /// <summary>Tags are stored reversed on disk, the same convention as MVER/REVM.</summary>
        private static byte[] TagBytes(string tag)
        {
            byte[] readable = Encoding.ASCII.GetBytes(tag);
            Array.Reverse(readable);
            return readable;
        }
    }
}
