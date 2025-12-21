using System.Text;
using System.Numerics;
using WoWRollback.PM4Module.Pipeline;

namespace WoWRollback.PM4Module.Generation;

/// <summary>
/// Writes valid WMO v17 files for debugging PM4 geometry.
/// Creates a Root .wmo and Group .wmo pair.
/// Adds diamond markers for MSCN points.
/// </summary>
public class Pm4DebugWmoWriter
{
    private const float MarkerSize = 0.5f;

    public void WriteWmo(Pm4WmoCandidate candidate, string outputDir, string baseName)
    {
        Directory.CreateDirectory(outputDir);
        
        string groupFileName = $"{baseName}_000.wmo";
        string rootFileName = $"{baseName}.wmo";

        // 1. Prepare Geometry
        // Combine Mesh Geometry + Generated MSCN Markers
        var vertices = new List<Vector3>();
        var indices = new List<int>();

        // Add Mesh
        if (candidate.DebugGeometry != null)
        {
            vertices.AddRange(candidate.DebugGeometry);
        }
        
        if (candidate.DebugFaces != null)
        {
            foreach (var face in candidate.DebugFaces)
            {
                // WMO winding is usually standard ccw? Or cw?
                // Pm4WmoWriter reversed winding. Let's assume re-winding is safer for visibility.
                if (face.Length >= 3)
                {
                    // Standard: 0, 1, 2
                    // If we want two-sided or safe, we can add both? No, too heavy.
                    // Let's try standard first.
                    indices.Add(face[0]);
                    indices.Add(face[1]);
                    indices.Add(face[2]);
                }
            }
        }

        // Add MSCN Markers (Diamonds)
        if (candidate.DebugMscnVertices != null)
        {
            foreach (var p in candidate.DebugMscnVertices)
            {
                AddDiamondMarker(p, vertices, indices);
            }
        }

        // 2. Calculate Local Space (Center Geometry)
        // WMOs are local space object. Placement in ADT defines position.
        // We center the geometry around (0,0,0) so the WMO origin is the center.
        // The CANDIDATE bounding box is in World Space.
        
        // Calculate centroid
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        foreach (var v in vertices)
        {
            min = Vector3.Min(min, v);
            max = Vector3.Max(max, v);
        }
        Vector3 centroid = (min + max) / 2f;

        // Make vertices local
        var localVertices = vertices.Select(v => v - centroid).ToList();
        var localBox = new BoundingBox(min - centroid, max - centroid);

        // 3. Write Files
        WriteGroupFile(Path.Combine(outputDir, groupFileName), localVertices, indices, localBox);
        WriteRootFile(Path.Combine(outputDir, rootFileName), localBox);
    }

    private void AddDiamondMarker(Vector3 p, List<Vector3> verts, List<int> indices)
    {
        int startIdx = verts.Count;
        
        // 6 Vertices
        verts.Add(p + new Vector3(0, MarkerSize, 0));  // Top (Y-up?) Or Z-up? Assuming Z-up based on PM4Decoder raw.
                                                       // Wait, if Z is up, Top is +Z.
        verts.Add(p + new Vector3(0, -MarkerSize, 0)); // Bottom
        verts.Add(p + new Vector3(-MarkerSize, 0, 0)); // Left
        verts.Add(p + new Vector3(MarkerSize, 0, 0));  // Right
        verts.Add(p + new Vector3(0, 0, MarkerSize));  // Front
        verts.Add(p + new Vector3(0, 0, -MarkerSize)); // Back

        // Actually, let's use +Z -Z for Top/Bottom if we assume Z-Up.
        // But PM4Decoder read raw X, Y, Z.
        // If we want a uniform "Blob", X/Y/Z offsets work regardless of axis.
        // But let's be explicit:
        // 0: Top (+Z)
        // 1: Bottom (-Z)
        // 2: Left (-X)
        // 3: Right (+X)
        // 4: Front (-Y)
        // 5: Back (+Y)
        
        // Let's redefine for Z-Up
        int top = startIdx + 0; verts[top] = p + new Vector3(0, 0, MarkerSize);
        int bot = startIdx + 1; verts[bot] = p + new Vector3(0, 0, -MarkerSize);
        int lft = startIdx + 2; verts[lft] = p + new Vector3(-MarkerSize, 0, 0);
        int rgt = startIdx + 3; verts[rgt] = p + new Vector3(MarkerSize, 0, 0);
        int fnt = startIdx + 4; verts[fnt] = p + new Vector3(0, -MarkerSize, 0);
        int bck = startIdx + 5; verts[bck] = p + new Vector3(0, MarkerSize, 0);

        // Indices (Triangles)
        // Top to sides
        AddTri(indices, top, lft, fnt);
        AddTri(indices, top, fnt, rgt);
        AddTri(indices, top, rgt, bck);
        AddTri(indices, top, bck, lft);
        
        // Bottom to sides (winding reversed relative to top)
        AddTri(indices, bot, fnt, lft);
        AddTri(indices, bot, rgt, fnt);
        AddTri(indices, bot, bck, rgt);
        AddTri(indices, bot, lft, bck);
    }

    private void AddTri(List<int> indices, int a, int b, int c)
    {
        indices.Add(a); indices.Add(b); indices.Add(c);
    }

    private void WriteRootFile(string path, BoundingBox bounds)
    {
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);

        WriteChunk(bw, "REVM", 17);
        
        // MOHD
        bw.Write(ToFourCC("DHOM"));
        bw.Write(64);
        bw.Write(1); // nTextures
        bw.Write(1); // nGroups
        bw.Write(0); // nPortals
        bw.Write(0); // nLights
        bw.Write(0); // nDoodadNames
        bw.Write(0); // nDoodadDefs
        bw.Write(0); // nDoodadSets
        bw.Write(0x007F7F7F); // ambColor
        bw.Write(0); // wmoID
        WriteBounds(bw, bounds);
        bw.Write((ushort)0);
        bw.Write((ushort)0);

        // MOTX
        string defaultTex = "tileset\\generic\\black.blp\0";
        bw.Write(ToFourCC("XTOM"));
        bw.Write(defaultTex.Length);
        bw.Write(Encoding.ASCII.GetBytes(defaultTex));

        // MOMT
        bw.Write(ToFourCC("TMOM"));
        bw.Write(64);
        bw.Write((uint)0); // flags
        bw.Write((uint)0); // shader
        bw.Write((uint)0); // blendMode
        bw.Write((uint)0); // texture_1
        bw.Write((uint)0xFF00FF00); // Emissive Green for visibility!
        bw.Write((uint)0);
        bw.Write((uint)0);
        bw.Write((uint)0xFFFFFFFF);
        bw.Write((uint)0);
        bw.Write((uint)0);
        bw.Write((uint)0);
        bw.Write((uint)0);
        bw.Write((uint)0); // runTimeData
        bw.Write((uint)0);
        bw.Write((uint)0);
        bw.Write((uint)0);

        // Empty Chunks
        WriteChunkHeader(bw, "NGOM", 0);
        
        // MOGI
        bw.Write(ToFourCC("IGOM"));
        bw.Write(32);
        bw.Write(0); // flags
        WriteBounds(bw, bounds);
        bw.Write(-1); // nameIndex

        // Empty Chunks set
        foreach(var cc in new[]{"BSOM", "VPOM", "TPOM", "RPOM", "VVOM", "BVOM", "TLOM", "SDOM", "NDOM", "DDOM"})
            WriteChunkHeader(bw, cc, 0);

        // MFOG
        bw.Write(ToFourCC("GOFM"));
        bw.Write(48);
        bw.Write(0);
        bw.Write(0f); bw.Write(0f); bw.Write(0f);
        bw.Write(100f); bw.Write(1000f); // Wide fog range
        bw.Write(1000f); bw.Write(0.1f); bw.Write(0xFFFFFFFF);
        bw.Write(1000f); bw.Write(0.1f); bw.Write(0xFFFFFFFF);
    }

    private void WriteGroupFile(string path, List<Vector3> vertices, List<int> indices, BoundingBox bounds)
    {
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);

        WriteChunk(bw, "REVM", 17);

        // MOGP Header
        bw.Write(ToFourCC("PGOM"));
        long sizePos = fs.Position;
        bw.Write(0);
        long startPos = fs.Position;

        bw.Write(0); // group_name
        bw.Write(0); // desc_name
        bw.Write(0); // flags
        WriteBounds(bw, bounds);
        bw.Write((ushort)0); // portal_start
        bw.Write((ushort)0); // portal_count
        bw.Write((ushort)0); // trans_batches
        bw.Write((ushort)0); // interior
        bw.Write((ushort)1); // exterior
        bw.Write((ushort)0); // padding
        bw.Write(0); // fogs
        bw.Write(0); // liquid
        bw.Write(0); // id
        bw.Write(0); // unk
        bw.Write(0); // unk

        // MOPY (Materials)
        bw.Write(ToFourCC("YPOM"));
        bw.Write((indices.Count / 3) * 2);
        for (int i = 0; i < indices.Count / 3; i++)
        {
            bw.Write((byte)0x00); // flags
            bw.Write((byte)0x00); // material 0
        }

        // MOVI (Indices)
        bw.Write(ToFourCC("IVOM"));
        bw.Write(indices.Count * 2);
        foreach (var idx in indices) bw.Write((ushort)idx);

        // MOVT (Vertices)
        bw.Write(ToFourCC("TVOM"));
        bw.Write(vertices.Count * 12);
        foreach (var v in vertices)
        {
            bw.Write(v.X);
            bw.Write(v.Y);
            bw.Write(v.Z);
        }

        // MONR (Normals)
        bw.Write(ToFourCC("RNOM"));
        bw.Write(vertices.Count * 12);
        foreach (var v in vertices)
        {
            bw.Write(0f); bw.Write(0f); bw.Write(1f);
        }

        // MOTV (TexCoords)
        bw.Write(ToFourCC("VTOM"));
        bw.Write(vertices.Count * 8);
        foreach (var v in vertices)
        {
            bw.Write(0f); bw.Write(0f);
        }

        // MOBA (Batches)
        bw.Write(ToFourCC("ABOM"));
        bw.Write(24);
        // Box shorts
        bw.Write((short)bounds.Min.X); bw.Write((short)bounds.Min.Y); bw.Write((short)bounds.Min.Z);
        bw.Write((short)bounds.Max.X); bw.Write((short)bounds.Max.Y); bw.Write((short)bounds.Max.Z);
        bw.Write((uint)0); // start
        bw.Write((ushort)indices.Count);
        bw.Write((ushort)0); // min
        bw.Write((ushort)(vertices.Count > 0 ? vertices.Count - 1 : 0));
        bw.Write((byte)0);
        bw.Write((byte)0);

        // MOGP Size - fixed header size for WMO v17 groups (68 bytes)
        // Sub-chunks (MOPY, MOVI, etc.) follow MOGP, they are NOT inside it in standard parsers.
        long endPos = fs.Position;
        fs.Position = sizePos;
        bw.Write((uint)68); 
        fs.Position = endPos;
    }

    private int ToFourCC(string s) => BitConverter.ToInt32(Encoding.ASCII.GetBytes(s), 0);
    
    private void WriteChunk(BinaryWriter bw, string fourCC, int value)
    {
        bw.Write(ToFourCC(fourCC));
        bw.Write(4);
        bw.Write(value);
    }
    
    private void WriteChunkHeader(BinaryWriter bw, string fourCC, int size)
    {
        bw.Write(ToFourCC(fourCC));
        bw.Write(size);
    }
    
    private void WriteBounds(BinaryWriter bw, BoundingBox b)
    {
        // WoW Standard Bounds: (minX, minZ, -minY, maxX, maxZ, -maxY) ???
        // Or (X, Z, Y)?
        // C3Vector: X, Y, Z.
        // CAaBox: min(X, Y, Z) max(X, Y, Z).
        // Standard chunk writing typically writes X, Y, Z order if that matches vector struct.
        // Let's write X, Y, Z.
        // WMO files usually store coordinates in Y-Up (Graphics Standard): X, Height(Z), Depth(Y)
        // Our Vector3 is Z-Up (WoW Standard): X, Y(Depth), Z(Height)
        // Map: v.X -> X, v.Z -> Y_Up, v.Y -> Z_Depth
        bw.Write(b.Min.X); bw.Write(b.Min.Z); bw.Write(b.Min.Y);
        bw.Write(b.Max.X); bw.Write(b.Max.Z); bw.Write(b.Max.Y);
    }

    public struct BoundingBox
    {
        public Vector3 Min;
        public Vector3 Max;
        public BoundingBox(Vector3 min, Vector3 max) { Min = min; Max = max; }
    }
}
