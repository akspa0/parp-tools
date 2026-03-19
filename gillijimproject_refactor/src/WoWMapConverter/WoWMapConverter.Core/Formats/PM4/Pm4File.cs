using System.Numerics;
namespace WoWMapConverter.Core.Formats.PM4;

/// <summary>
/// PM4 pathfinding file parser.
/// Extracts navigation mesh data, surfaces, and placement references.
/// </summary>
public class Pm4File
{
    public Pm4FileStructure Structure { get; }

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
        Structure = Pm4Decoder.Decode(data);
        PopulateLegacyView(Structure);
    }

    private void PopulateLegacyView(Pm4FileStructure structure)
    {
        Version = structure.Version;

        if (structure.Header != null)
        {
            Header = new Pm4Header
            {
                Unk00 = structure.Header.Field00,
                Unk04 = structure.Header.Field04,
                Unk08 = structure.Header.Field08,
                Unk0C = structure.Header.Field0C,
                Unk10 = structure.Header.Field10,
                Unk14 = structure.Header.Field14,
                Unk18 = structure.Header.Field18,
                Unk1C = structure.Header.Field1C
            };
        }

        PathVertices.AddRange(structure.PathVertices);
        PathIndices.AddRange(structure.PathIndices);
        MeshVertices.AddRange(structure.MeshVertices);
        MeshIndices.AddRange(structure.MeshIndices);
        ExteriorVertices.AddRange(structure.SceneNodes);

        foreach (MslkChunk link in structure.LinkEntries)
        {
            LinkEntries.Add(new MslkEntry
            {
                TypeFlags = link.TypeFlags,
                Subtype = link.Subtype,
                Padding = link.Padding,
                GroupObjectId = link.GroupObjectId,
                MspiFirstIndex = link.MspiFirstIndex,
                MspiIndexCount = link.MspiIndexCount,
                LinkId = link.LinkId,
                RefIndex = link.RefIndex,
                SystemFlag = link.SystemFlag
            });
        }

        foreach (MsurChunk surface in structure.Surfaces)
        {
            Surfaces.Add(new MsurEntry
            {
                GroupKey = surface.GroupKey,
                IndexCount = surface.IndexCount,
                AttributeMask = surface.AttributeMask,
                Padding = surface.Padding,
                NormalX = surface.Normal.X,
                NormalY = surface.Normal.Y,
                NormalZ = surface.Normal.Z,
                Height = surface.Height,
                MsviFirstIndex = surface.MsviFirstIndex,
                MdosIndex = surface.MdosIndex,
                PackedParams = surface.PackedParams
            });
        }

        foreach (MprlChunk positionRef in structure.PositionRefs)
        {
            PositionRefs.Add(new MprlEntry
            {
                Unk00 = positionRef.Unk00,
                Unk02 = positionRef.Unk02,
                Unk04 = positionRef.Unk04,
                Unk06 = positionRef.Unk06,
                PositionX = positionRef.Position.X,
                PositionY = positionRef.Position.Y,
                PositionZ = positionRef.Position.Z,
                Unk14 = positionRef.Unk14,
                Unk16 = positionRef.Unk16
            });
        }

        foreach (MprrChunk graphEntry in structure.GraphEntries)
            MprrEntries.Add(new MprrEntry(graphEntry.Value1, graphEntry.Value2));

        foreach ((string key, uint value) in structure.ChunkSizes)
            ChunkSizes[key] = value;

        UnparsedChunks.AddRange(structure.UnparsedChunks);
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
    public uint LinkId { get; set; }
    public ushort RefIndex { get; set; }
    public ushort SystemFlag { get; set; }
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

    public uint Ck24 => (PackedParams >> 8) & 0xFFFFFF;

    public byte Ck24Type => (byte)((PackedParams >> 24) & 0xFF);

    public ushort Ck24ObjectId => (ushort)(Ck24 & 0xFFFF);
}

public class MprlEntry
{
    public ushort Unk00 { get; set; }
    public short Unk02 { get; set; }
    public ushort Unk04 { get; set; }
    public ushort Unk06 { get; set; }
    public float PositionX { get; set; }
    public float PositionY { get; set; }
    public float PositionZ { get; set; }
    public short Unk14 { get; set; }
    public ushort Unk16 { get; set; }

    public uint RotationOrFlags
    {
        get => Unk04;
        set => Unk04 = (ushort)value;
    }

    public Vector3 Position => new(PositionX, PositionY, PositionZ);
}

public record MprrEntry(ushort Value1, ushort Value2)
{
    public bool IsSentinel => Value1 == 0xFFFF;
}
