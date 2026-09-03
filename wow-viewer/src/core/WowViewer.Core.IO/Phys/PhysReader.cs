using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Chunks;
using WowViewer.Core.Phys;

namespace WowViewer.Core.IO.Phys;

/// <summary>
/// Reads a <c>.phys</c> physics sidecar.
/// </summary>
/// <remarks>
/// The container and every record stride below are MEASURED from the 5.0.1 client
/// (<c>FUN_005a5080</c> and the <c>PhysData.h</c> consumers); see
/// <c>specs/214-mop-physics-domino/evidence/physics-contract.md</c>.
/// <para>
/// Two behaviours are deliberate and load-bearing. First, this reader <b>fails closed</b>: malformed
/// input yields <see langword="false"/> and a diagnostic, never an exception, so a bad sidecar leaves
/// the model rendering normally exactly as the client does. Second, <b>unknown chunk tags are skipped
/// by their size field and never rejected</b> — the client is permissive here, so a stricter reader
/// would break on files from later eras.
/// </para>
/// <para>
/// Where the client never reads a byte range, this reader preserves it rather than interpreting it.
/// Nothing here has been validated against real client bytes; that is spec 214 task T002.
/// </para>
/// </remarks>
public static class PhysReader
{
    private const int ChunkHeaderSize = 8;

    // MEASURED: the parser derives every count by dividing chunk size by these strides.
    private const int BoxShapeStride = 60;
    private const int CapsuleShapeStride = 28;
    private const int SphereShapeStride = 16;
    private const int ShapeStride = 20;
    private const int BodyStride = 28;
    private const int JointStride = 16;
    private const int SphericalJointStride = 28;
    private const int ShoulderJointStride = 108;
    private const int WeldJointStride = 104;

    private const int BoxShapeUnverifiedPrefixLength = 48;

    private static readonly FourCC PhysTag = FourCC.FromString("PHYS");
    private static readonly FourCC BoxShapeTag = FourCC.FromString("BOXS");
    private static readonly FourCC CapsuleShapeTag = FourCC.FromString("CAPS");
    private static readonly FourCC SphereShapeTag = FourCC.FromString("SPHS");
    private static readonly FourCC ShapeTag = FourCC.FromString("SHAP");
    private static readonly FourCC BodyTag = FourCC.FromString("BODY");
    private static readonly FourCC SphericalJointTag = FourCC.FromString("SPHJ");
    private static readonly FourCC ShoulderJointTag = FourCC.FromString("SHOJ");
    private static readonly FourCC WeldJointTag = FourCC.FromString("WELJ");
    private static readonly FourCC JointTag = FourCC.FromString("JOIN");

    /// <summary>
    /// Parses <paramref name="data"/> as a <c>.phys</c> sidecar.
    /// </summary>
    /// <returns>
    /// <see langword="true"/> when the header is valid and the chunk walk completed. A
    /// <see langword="false"/> result still returns a document carrying the diagnostics that explain
    /// the refusal.
    /// </returns>
    public static bool TryRead(ReadOnlySpan<byte> data, out PhysDocument document)
    {
        List<PhysDiagnostic> diagnostics = [];

        if (data.Length < ChunkHeaderSize)
        {
            diagnostics.Add(Error("phys.truncated-header", "The file is shorter than one chunk header."));
            document = Empty(diagnostics);
            return false;
        }

        FourCC magic = FourCC.FromFileBytes(data);
        if (magic != PhysTag)
        {
            diagnostics.Add(Error(
                "phys.bad-magic",
                $"Expected the container magic 'PHYS' but found '{magic}'."));
            document = Empty(diagnostics);
            return false;
        }

        uint headerSize = BinaryPrimitives.ReadUInt32LittleEndian(data[4..]);
        long payloadStart = (long)ChunkHeaderSize + headerSize;
        if (headerSize < sizeof(ushort) || payloadStart > data.Length)
        {
            diagnostics.Add(Error(
                "phys.bad-header-size",
                $"The PHYS header declares {headerSize} payload bytes, which does not fit the {data.Length}-byte file."));
            document = Empty(diagnostics);
            return false;
        }

        ushort version = BinaryPrimitives.ReadUInt16LittleEndian(data[ChunkHeaderSize..]);
        if (version != PhysDocument.SupportedVersion)
        {
            diagnostics.Add(Error(
                "phys.unsupported-version",
                $"PHYS version {version} is not supported; 5.0.1 accepts only version {PhysDocument.SupportedVersion}."));
            document = Empty(diagnostics);
            return false;
        }

        List<PhysBoxShape> boxShapes = [];
        List<PhysCapsuleShape> capsuleShapes = [];
        List<PhysSphereShape> sphereShapes = [];
        List<PhysShape> shapes = [];
        List<PhysBody> bodies = [];
        List<PhysJoint> joints = [];
        List<PhysOpaqueRecord> sphericalJoints = [];
        List<PhysOpaqueRecord> shoulderJoints = [];
        List<PhysOpaqueRecord> weldJoints = [];

        int offset = (int)payloadStart;
        while (offset + ChunkHeaderSize <= data.Length)
        {
            FourCC tag = FourCC.FromFileBytes(data[offset..]);
            uint size = BinaryPrimitives.ReadUInt32LittleEndian(data[(offset + 4)..]);
            int payloadOffset = offset + ChunkHeaderSize;
            long payloadEnd = (long)payloadOffset + size;

            if (payloadEnd > data.Length)
            {
                diagnostics.Add(Error(
                    "phys.truncated-chunk",
                    $"Chunk '{tag}' at offset {offset} declares {size} bytes but the file ends at {data.Length}."));
                break;
            }

            ReadOnlySpan<byte> payload = data.Slice(payloadOffset, (int)size);

            if (tag == BoxShapeTag)
                ReadBoxShapes(tag, payload, boxShapes, diagnostics);
            else if (tag == CapsuleShapeTag)
                ReadCapsuleShapes(tag, payload, capsuleShapes, diagnostics);
            else if (tag == SphereShapeTag)
                ReadSphereShapes(tag, payload, sphereShapes, diagnostics);
            else if (tag == ShapeTag)
                ReadShapes(tag, payload, shapes, diagnostics);
            else if (tag == BodyTag)
                ReadBodies(tag, payload, bodies, diagnostics);
            else if (tag == JointTag)
                ReadJoints(tag, payload, joints, diagnostics);
            else if (tag == SphericalJointTag)
                ReadOpaque(tag, payload, SphericalJointStride, sphericalJoints, diagnostics);
            else if (tag == ShoulderJointTag)
                ReadOpaque(tag, payload, ShoulderJointStride, shoulderJoints, diagnostics);
            else if (tag == WeldJointTag)
                ReadOpaque(tag, payload, WeldJointStride, weldJoints, diagnostics);
            else
                // MEASURED: the client skips unrecognised tags by size and keeps walking. Matching
                // that keeps the reader forward-compatible; rejecting would be stricter than 5.0.1.
                diagnostics.Add(new PhysDiagnostic(
                    PhysDiagnosticSeverity.Info,
                    "phys.unknown-chunk",
                    $"Skipped unrecognised chunk '{tag}' ({size} bytes) at offset {offset}."));

            offset = (int)payloadEnd;
        }

        document = new PhysDocument(
            version,
            boxShapes,
            capsuleShapes,
            sphereShapes,
            shapes,
            bodies,
            joints,
            sphericalJoints,
            shoulderJoints,
            weldJoints,
            diagnostics);
        return true;
    }

    private static void ReadBoxShapes(
        FourCC tag,
        ReadOnlySpan<byte> payload,
        List<PhysBoxShape> target,
        List<PhysDiagnostic> diagnostics)
    {
        int count = CountRecords(tag, payload.Length, BoxShapeStride, diagnostics);
        for (int index = 0; index < count; index++)
        {
            ReadOnlySpan<byte> record = payload.Slice(index * BoxShapeStride, BoxShapeStride);
            target.Add(new PhysBoxShape(
                record[..BoxShapeUnverifiedPrefixLength].ToArray(),
                ReadVector3(record[BoxShapeUnverifiedPrefixLength..])));
        }
    }

    private static void ReadCapsuleShapes(
        FourCC tag,
        ReadOnlySpan<byte> payload,
        List<PhysCapsuleShape> target,
        List<PhysDiagnostic> diagnostics)
    {
        int count = CountRecords(tag, payload.Length, CapsuleShapeStride, diagnostics);
        for (int index = 0; index < count; index++)
        {
            ReadOnlySpan<byte> record = payload.Slice(index * CapsuleShapeStride, CapsuleShapeStride);
            target.Add(new PhysCapsuleShape(
                ReadVector3(record),
                ReadVector3(record[12..]),
                BinaryPrimitives.ReadSingleLittleEndian(record[24..])));
        }
    }

    private static void ReadSphereShapes(
        FourCC tag,
        ReadOnlySpan<byte> payload,
        List<PhysSphereShape> target,
        List<PhysDiagnostic> diagnostics)
    {
        int count = CountRecords(tag, payload.Length, SphereShapeStride, diagnostics);
        for (int index = 0; index < count; index++)
        {
            ReadOnlySpan<byte> record = payload.Slice(index * SphereShapeStride, SphereShapeStride);
            target.Add(new PhysSphereShape(
                ReadVector3(record),
                BinaryPrimitives.ReadSingleLittleEndian(record[12..])));
        }
    }

    private static void ReadShapes(
        FourCC tag,
        ReadOnlySpan<byte> payload,
        List<PhysShape> target,
        List<PhysDiagnostic> diagnostics)
    {
        int count = CountRecords(tag, payload.Length, ShapeStride, diagnostics);
        for (int index = 0; index < count; index++)
        {
            ReadOnlySpan<byte> record = payload.Slice(index * ShapeStride, ShapeStride);
            ushort rawKind = BinaryPrimitives.ReadUInt16LittleEndian(record);
            if (rawKind > (ushort)PhysShapeKind.Sphere)
            {
                diagnostics.Add(new PhysDiagnostic(
                    PhysDiagnosticSeverity.Warning,
                    "phys.unknown-shape-kind",
                    $"SHAP record {index} declares unknown shape kind {rawKind}; the record is retained but cannot be built."));
            }

            target.Add(new PhysShape(
                (PhysShapeKind)rawKind,
                BinaryPrimitives.ReadUInt16LittleEndian(record[2..]),
                BinaryPrimitives.ReadUInt32LittleEndian(record[4..]),
                BinaryPrimitives.ReadUInt32LittleEndian(record[8..]),
                BinaryPrimitives.ReadUInt32LittleEndian(record[12..]),
                BinaryPrimitives.ReadUInt32LittleEndian(record[16..])));
        }
    }

    private static void ReadBodies(
        FourCC tag,
        ReadOnlySpan<byte> payload,
        List<PhysBody> target,
        List<PhysDiagnostic> diagnostics)
    {
        int count = CountRecords(tag, payload.Length, BodyStride, diagnostics);
        for (int index = 0; index < count; index++)
        {
            ReadOnlySpan<byte> record = payload.Slice(index * BodyStride, BodyStride);
            target.Add(new PhysBody(
                BinaryPrimitives.ReadUInt16LittleEndian(record),
                ReadVector3(record[4..]),
                BinaryPrimitives.ReadUInt16LittleEndian(record[16..]),
                BinaryPrimitives.ReadUInt32LittleEndian(record[20..]),
                BinaryPrimitives.ReadUInt32LittleEndian(record[24..])));
        }
    }

    private static void ReadJoints(
        FourCC tag,
        ReadOnlySpan<byte> payload,
        List<PhysJoint> target,
        List<PhysDiagnostic> diagnostics)
    {
        int count = CountRecords(tag, payload.Length, JointStride, diagnostics);
        for (int index = 0; index < count; index++)
        {
            ReadOnlySpan<byte> record = payload.Slice(index * JointStride, JointStride);
            ushort rawKind = BinaryPrimitives.ReadUInt16LittleEndian(record[12..]);
            if (rawKind > (ushort)PhysJointKind.Weld)
            {
                diagnostics.Add(new PhysDiagnostic(
                    PhysDiagnosticSeverity.Warning,
                    "phys.unknown-joint-kind",
                    $"JOIN record {index} declares unknown joint kind {rawKind}; the record is retained but cannot be built."));
            }

            target.Add(new PhysJoint(
                BinaryPrimitives.ReadUInt32LittleEndian(record),
                BinaryPrimitives.ReadUInt32LittleEndian(record[4..]),
                BinaryPrimitives.ReadUInt32LittleEndian(record[8..]),
                (PhysJointKind)rawKind,
                BinaryPrimitives.ReadUInt16LittleEndian(record[14..])));
        }
    }

    private static void ReadOpaque(
        FourCC tag,
        ReadOnlySpan<byte> payload,
        int stride,
        List<PhysOpaqueRecord> target,
        List<PhysDiagnostic> diagnostics)
    {
        int count = CountRecords(tag, payload.Length, stride, diagnostics);
        for (int index = 0; index < count; index++)
            target.Add(new PhysOpaqueRecord(payload.Slice(index * stride, stride).ToArray()));
    }

    private static int CountRecords(
        FourCC tag,
        int payloadLength,
        int stride,
        List<PhysDiagnostic> diagnostics)
    {
        int count = payloadLength / stride;
        int remainder = payloadLength % stride;
        if (remainder != 0)
        {
            diagnostics.Add(new PhysDiagnostic(
                PhysDiagnosticSeverity.Warning,
                "phys.stride-mismatch",
                $"Chunk '{tag}' is {payloadLength} bytes, which is not a multiple of its {stride}-byte record stride; {remainder} trailing byte(s) ignored."));
        }

        return count;
    }

    private static Vector3 ReadVector3(ReadOnlySpan<byte> source)
    {
        return new Vector3(
            BinaryPrimitives.ReadSingleLittleEndian(source),
            BinaryPrimitives.ReadSingleLittleEndian(source[4..]),
            BinaryPrimitives.ReadSingleLittleEndian(source[8..]));
    }

    private static PhysDiagnostic Error(string code, string message)
    {
        return new PhysDiagnostic(PhysDiagnosticSeverity.Error, code, message);
    }

    private static PhysDocument Empty(IReadOnlyList<PhysDiagnostic> diagnostics)
    {
        return new PhysDocument(
            version: 0,
            boxShapes: [],
            capsuleShapes: [],
            sphereShapes: [],
            shapes: [],
            bodies: [],
            joints: [],
            sphericalJoints: [],
            shoulderJoints: [],
            weldJoints: [],
            diagnostics: diagnostics);
    }
}
