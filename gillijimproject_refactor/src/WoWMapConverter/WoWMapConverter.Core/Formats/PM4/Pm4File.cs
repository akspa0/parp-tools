using System.Numerics;
using System.Text;

namespace WoWMapConverter.Core.Formats.PM4;

/// <summary>
/// PM4 pathfinding file parser.
/// Extracts navigation mesh data, surfaces, and placement references.
/// </summary>
public class Pm4File
{
    public uint Version { get; private set; }
    public Pm4Header Header { get; private set; } = new();
    public List<MslkEntry> LinkEntries { get; } = new();
    public List<Vector3> PathVertices { get; } = new();      // MSPV
    public List<uint> PathIndices { get; } = new();          // MSPI
    public List<Vector3> MeshVertices { get; } = new();      // MSVT
    public List<uint> MeshIndices { get; } = new();          // MSVI
    public List<MsurEntry> Surfaces { get; } = new();        // MSUR
    public List<Vector3> ExteriorVertices { get; } = new();  // MSCN
    public List<MprlEntry> PositionRefs { get; } = new();    // MPRL
    public List<MprrEntry> MprrEntries { get; } = new();     // MPRR

    public Dictionary<string, uint> ChunkSizes { get; } = new();
    public List<string> UnparsedChunks { get; } = new();

    public static Pm4File FromFile(string path)
    {
        return new Pm4File(File.ReadAllBytes(path));
    }

    public Pm4File(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);
        Parse(br);
    }

    private void Parse(BinaryReader br)
    {
        while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
        {
            var sigBytes = br.ReadBytes(4);
            Array.Reverse(sigBytes);
            string sig = Encoding.ASCII.GetString(sigBytes);
            uint size = br.ReadUInt32();
            long dataStart = br.BaseStream.Position;

            ChunkSizes[sig] = size;

            switch (sig)
            {
                case "MVER":
                    Version = br.ReadUInt32();
                    break;
                case "MSHD":
                    Header = ReadHeader(br);
                    break;
                case "MSLK":
                    ReadMslk(br, size);
                    break;
                case "MSPV":
                    ReadVectors(br, size, PathVertices);
                    break;
                case "MSPI":
                    ReadUints(br, size, PathIndices);
                    break;
                case "MSVT":
                    ReadVectors(br, size, MeshVertices, isMsvt: true);
                    break;
                case "MSVI":
                    ReadUints(br, size, MeshIndices);
                    break;
                case "MSUR":
                    ReadMsur(br, size);
                    break;
                case "MSCN":
                    ReadVectors(br, size, ExteriorVertices);
                    break;
                case "MPRL":
                    ReadMprl(br, size);
                    break;
                case "MPRR":
                    ReadMprr(br, size);
                    break;
                default:
                    UnparsedChunks.Add($"{sig}:{size}");
                    break;
            }

            br.BaseStream.Position = dataStart + size;
        }
    }

    private Pm4Header ReadHeader(BinaryReader br)
    {
        return new Pm4Header
        {
            Unk00 = br.ReadUInt32(),
            Unk04 = br.ReadUInt32(),
            Unk08 = br.ReadUInt32(),
            Unk0C = br.ReadUInt32(),
            Unk10 = br.ReadUInt32(),
            Unk14 = br.ReadUInt32(),
            Unk18 = br.ReadUInt32(),
            Unk1C = br.ReadUInt32()
        };
    }

    private void ReadMslk(BinaryReader br, uint size)
    {
        int count = (int)(size / 20);
        for (int i = 0; i < count; i++)
        {
            var entry = new MslkEntry
            {
                TypeFlags = br.ReadByte(),
                Subtype = br.ReadByte(),
                Padding = br.ReadUInt16(),
                GroupObjectId = br.ReadUInt32()
            };

            // Read Int24 for MspiFirstIndex
            byte[] b = br.ReadBytes(3);
            int mspiFirst = b[0] | (b[1] << 8) | (b[2] << 16);
            if ((mspiFirst & 0x800000) != 0) mspiFirst |= unchecked((int)0xFF000000);
            entry.MspiFirstIndex = mspiFirst;

            entry.MspiIndexCount = br.ReadByte();

            // Read Int24 for MsviFirstIndex
            b = br.ReadBytes(3);
            int msviFirst = b[0] | (b[1] << 8) | (b[2] << 16);
            if ((msviFirst & 0x800000) != 0) msviFirst |= unchecked((int)0xFF000000);
            entry.MsviFirstIndex = msviFirst;

            entry.MsviIndexCount = br.ReadByte();
            entry.MsurIndex = br.ReadUInt32();

            LinkEntries.Add(entry);
        }
    }

    private void ReadVectors(BinaryReader br, uint size, List<Vector3> target, bool isMsvt = false)
    {
        int count = (int)(size / 12);
        for (int i = 0; i < count; i++)
        {
            float x = br.ReadSingle();
            float y = br.ReadSingle();
            float z = br.ReadSingle();
            target.Add(new Vector3(x, y, z));
        }
    }

    private void ReadUints(BinaryReader br, uint size, List<uint> target)
    {
        int count = (int)(size / 4);
        for (int i = 0; i < count; i++)
            target.Add(br.ReadUInt32());
    }

    private void ReadMsur(BinaryReader br, uint size)
    {
        // MSUR entries are 32 bytes each
        int count = (int)(size / 32);
        for (int i = 0; i < count; i++)
        {
            Surfaces.Add(new MsurEntry
            {
                GroupKey = br.ReadByte(),
                IndexCount = br.ReadByte(),
                AttributeMask = br.ReadByte(),
                Padding = br.ReadByte(),
                NormalX = br.ReadSingle(),
                NormalY = br.ReadSingle(),
                NormalZ = br.ReadSingle(),
                Height = br.ReadSingle(),
                MsviFirstIndex = br.ReadUInt32(),
                MdosIndex = br.ReadUInt32(),
                PackedParams = br.ReadUInt32()
            });
        }
    }

    private void ReadMprl(BinaryReader br, uint size)
    {
        int count = (int)(size / 16);
        for (int i = 0; i < count; i++)
        {
            PositionRefs.Add(new MprlEntry
            {
                PositionX = br.ReadSingle(),
                PositionY = br.ReadSingle(),
                PositionZ = br.ReadSingle(),
                RotationOrFlags = br.ReadUInt32()
            });
        }
    }

    private void ReadMprr(BinaryReader br, uint size)
    {
        int count = (int)(size / 4);
        for (int i = 0; i < count; i++)
        {
            ushort val1 = br.ReadUInt16();
            ushort val2 = br.ReadUInt16();
            MprrEntries.Add(new MprrEntry(val1, val2));
        }
    }

    /// <summary>
    /// Get triangles for a specific surface.
    /// </summary>
    public IEnumerable<(Vector3 v0, Vector3 v1, Vector3 v2)> GetSurfaceTriangles(int surfaceIndex)
    {
        if (surfaceIndex < 0 || surfaceIndex >= Surfaces.Count)
            yield break;

        var surface = Surfaces[surfaceIndex];
        int triCount = (int)(surface.IndexCount / 3);

        for (int i = 0; i < triCount; i++)
        {
            int baseIdx = (int)surface.MsviFirstIndex + i * 3;
            if (baseIdx + 2 >= MeshIndices.Count) break;

            int i0 = (int)MeshIndices[baseIdx];
            int i1 = (int)MeshIndices[baseIdx + 1];
            int i2 = (int)MeshIndices[baseIdx + 2];

            if (i0 < MeshVertices.Count && i1 < MeshVertices.Count && i2 < MeshVertices.Count)
                yield return (MeshVertices[i0], MeshVertices[i1], MeshVertices[i2]);
        }
    }

    /// <summary>
    /// Export all mesh geometry to OBJ format.
    /// </summary>
    public string ExportToObj()
    {
        using var sw = new StringWriter();
        sw.WriteLine("# PM4 Mesh Export");
        sw.WriteLine($"# Vertices: {MeshVertices.Count}");
        sw.WriteLine($"# Surfaces: {Surfaces.Count}");
        sw.WriteLine($"# Indices: {MeshIndices.Count}");
        sw.WriteLine();

        // Write vertices
        foreach (var v in MeshVertices)
            sw.WriteLine($"v {v.X:F6} {v.Y:F6} {v.Z:F6}");

        sw.WriteLine();

        // Write faces per surface
        int faceOffset = 1; // OBJ is 1-indexed
        int validFaces = 0;
        int skippedFaces = 0;
        int indexCount = MeshIndices.Count;
        int vertexCount = MeshVertices.Count;

        foreach (var surface in Surfaces)
        {
            sw.WriteLine($"g surface_{surface.GroupKey}");
            
            // Validate surface bounds
            if (surface.MsviFirstIndex >= indexCount)
            {
                skippedFaces += surface.IndexCount / 3;
                continue;
            }

            // IndexCount is now a byte (0-255), represents triangle count * 3
            int triCount = surface.IndexCount / 3;
            int startIdx = (int)surface.MsviFirstIndex;

            for (int i = 0; i < triCount; i++)
            {
                int baseIdx = startIdx + i * 3;
                if (baseIdx + 2 >= indexCount || baseIdx < 0)
                {
                    skippedFaces++;
                    continue;
                }

                int i0 = (int)MeshIndices[baseIdx];
                int i1 = (int)MeshIndices[baseIdx + 1];
                int i2 = (int)MeshIndices[baseIdx + 2];

                // Validate vertex indices
                if (i0 < 0 || i1 < 0 || i2 < 0 ||
                    i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount)
                {
                    skippedFaces++;
                    continue;
                }

                sw.WriteLine($"f {i0 + faceOffset} {i1 + faceOffset} {i2 + faceOffset}");
                validFaces++;
            }
        }

        sw.WriteLine();
        sw.WriteLine($"# Valid faces: {validFaces}, Skipped: {skippedFaces}");
        return sw.ToString();
    }
}

public class Pm4Header
{
    public uint Unk00 { get; set; }
    public uint Unk04 { get; set; }
    public uint Unk08 { get; set; }
    public uint Unk0C { get; set; }
    public uint Unk10 { get; set; }
    public uint Unk14 { get; set; }
    public uint Unk18 { get; set; }
    public uint Unk1C { get; set; }
}

public class MslkEntry
{
    public byte TypeFlags { get; set; }
    public byte Subtype { get; set; }
    public ushort Padding { get; set; }
    public uint GroupObjectId { get; set; }
    public int MspiFirstIndex { get; set; }
    public byte MspiIndexCount { get; set; }
    public int MsviFirstIndex { get; set; }
    public byte MsviIndexCount { get; set; }
    public uint MsurIndex { get; set; }
}

public class MsurEntry
{
    public byte GroupKey { get; set; }
    public byte IndexCount { get; set; }
    public byte AttributeMask { get; set; }
    public byte Padding { get; set; }
    public float NormalX { get; set; }
    public float NormalY { get; set; }
    public float NormalZ { get; set; }
    public float Height { get; set; }
    public uint MsviFirstIndex { get; set; }
    public uint MdosIndex { get; set; }
    public uint PackedParams { get; set; }
}

public class MprlEntry
{
    public float PositionX { get; set; }
    public float PositionY { get; set; }
    public float PositionZ { get; set; }
    public uint RotationOrFlags { get; set; }

    public Vector3 Position => new(PositionX, PositionY, PositionZ);
}

public record MprrEntry(ushort Value1, ushort Value2)
{
    public bool IsSentinel => Value1 == 0xFFFF;
}
