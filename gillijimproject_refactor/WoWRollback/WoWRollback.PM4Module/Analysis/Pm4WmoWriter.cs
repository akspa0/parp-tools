using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using System.Linq;

namespace WoWRollback.PM4Module.Analysis;

/// <summary>
/// Writes minimal valid WMO v17 files from raw geometry.
/// Creates a linked Root .wmo and Group .wmo file pair.
/// Centers geometry around (0,0,0) and returns the world placement position.
/// </summary>
public class Pm4WmoWriter
{
    /// <summary>
    /// Writes WMO files and returns the centroid (World Position) of the object.
    /// Input vertices are assumed to be in PM4 space (Y-Up).
    /// Output WMO is Z-Up centered.
    /// Returned centroid is Z-Up world space.
    /// </summary>
    public Vector3 WriteWmo(string outputDir, string baseName, List<Vector3> pm4Vertices, List<int> indices)
    {
        if (pm4Vertices.Count == 0 || indices.Count == 0) return Vector3.Zero;

        // Ensure output directory
        Directory.CreateDirectory(outputDir);
        
        // 1. Convert to Z-Up (X, Y, Z) -> (X, Z, Y)
        // PM4 (Y-up): (x, height, z)
        // WoW (Z-up): (x, z, height)
        var wowVertices = pm4Vertices.Select(v => new Vector3(v.X, v.Z, v.Y)).ToList();

        // 2. Calculate Centroid (World Position)
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        foreach (var v in wowVertices)
        {
            min = Vector3.Min(min, v);
            max = Vector3.Max(max, v);
        }
        Vector3 centroid = (min + max) / 2f;

        // 3. Center Vertices (Local Space)
        var localVertices = wowVertices.Select(v => v - centroid).ToList();
        
        // 4. Calculate Local Bounds
        var bounds = CalculateBounds(localVertices);
        
        // Write Group File first (_000.wmo)
        string groupFileName = $"{baseName}_000.wmo";
        string rootFileName = $"{baseName}.wmo";
        
        WriteGroupFile(Path.Combine(outputDir, groupFileName), localVertices, indices, bounds);
        WriteRootFile(Path.Combine(outputDir, rootFileName), bounds);

        return centroid;
    }

    private void WriteRootFile(string path, BoundingBox bounds)
    {
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);

        // MVER
        WriteChunk(bw, "REVM", 17); // MVER = 17

        // MOHD
        bw.Write(ToFourCC("DHOM")); // MOHD
        bw.Write(64); // Size
        bw.Write(0); // nTextures
        bw.Write(1); // nGroups
        bw.Write(0); // nPortals
        bw.Write(0); // nLights
        bw.Write(0); // nModels
        bw.Write(0); // nDoodads
        bw.Write(0); // nDoodadSets
        bw.Write(0x00FF00FF); // ambColor?
        bw.Write(0); // wmoID
        WriteBounds(bw, bounds);
        bw.Write((short)0); // flags?
        bw.Write((short)0); // padding?

        // MOGN (Group Names) - Empty
        WriteChunkHeader(bw, "NGOM", 0);

        // MOGI (Group Info) - 32 bytes per group
        bw.Write(ToFourCC("IGOM"));
        bw.Write(32); // 1 group * 32 bytes
        bw.Write(0); // flags
        WriteBounds(bw, bounds);
        bw.Write(-1); // nameIndex

        // MOSI (Skybox) - empty
        WriteChunkHeader(bw, "ISOM", 0);
        
        // MOPV (Portal Verts) - empty
        WriteChunkHeader(bw, "VPOM", 0);

        // MOPT (Portal Info) - empty
        WriteChunkHeader(bw, "TPOM", 0);

        // MOPR (Portal Refs) - empty
        WriteChunkHeader(bw, "RPOM", 0);

        // MOLT (Lights) - empty
        WriteChunkHeader(bw, "TLOM", 0);

        // MODS (Doodad Sets) - empty
        WriteChunkHeader(bw, "SDOM", 0);

        // MODN (Doodad Names) - empty
        WriteChunkHeader(bw, "NDOM", 0);

        // MODD (Doodad Defs) - empty
        WriteChunkHeader(bw, "DDOM", 0);

        // MFOG (Fog) - empty
        WriteChunkHeader(bw, "GOFM", 0);
    }

    private void WriteGroupFile(string path, List<Vector3> vertices, List<int> indices, BoundingBox bounds)
    {
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);

        // MVER
        WriteChunk(bw, "REVM", 17);

        // MOGP Header
        bw.Write(ToFourCC("PGOM"));
        
        long sizePos = fs.Position;
        bw.Write(0); // Placeholder size

        long startPos = fs.Position;
        
        // MOGP Body (68 bytes)
        bw.Write(0); // nameOffset
        bw.Write(0); // descriptiveNameOffset
        bw.Write(0); // flags
        WriteBounds(bw, bounds);
        bw.Write((short)0); // portalStart
        bw.Write((short)0); // portalCount
        bw.Write((short)0); // transBatchCount
        bw.Write((short)0); // intBatchCount
        bw.Write((short)0); // extBatchCount
        bw.Write((short)0); // padding
        bw.Write((byte)0); bw.Write((byte)0); bw.Write((byte)0); bw.Write((byte)0); // fogIndices
        bw.Write(0); // liquidType
        bw.Write(0); // wmoID
        bw.Write(0); // flags2
        bw.Write(0); // unk

        // MOPY (Material Info) - 2 bytes per triangle
        bw.Write(ToFourCC("YPOM"));
        bw.Write((indices.Count / 3) * 2);
        for (int i = 0; i < indices.Count / 3; i++)
        {
            bw.Write((byte)0x00); // flags?
            bw.Write((byte)0x00); // material ID?
        }

        // MOVI (Indices) - 2 bytes per index
        bw.Write(ToFourCC("IVOM"));
        bw.Write(indices.Count * 2);
        foreach (var idx in indices)
        {
            bw.Write((ushort)idx);
        }

        // MOVT (Vertices) - 12 bytes per vertex
        bw.Write(ToFourCC("TVOM"));
        bw.Write(vertices.Count * 12);
        foreach (var v in vertices)
        {
            // Vertices are already local Z-Up. Write X, Y, Z.
            // WoW WMO coords: X, Y, Z (where Z is height).
            // Wait, Standard WMO (Z-up) usually writes vertices as X, Z, -Y relative to world?
            // "The orientation of the model is specified by the placement in the ADT"
            // If the ADT placement has Rotation=(0,0,0), the WMO axes align with World axes.
            // World X = North, World Y = West, World Z = Up.
            // So we just write X, Y, Z (where Y is West, Z is Up).
            // Our "wowVertices" are (pm4.X, pm4.Z, pm4.Y).
            // This is (WorldX, WorldY, WorldZ).
            // So we write X, Y, Z. CORRECT.
            bw.Write(v.X);
            bw.Write(v.Y); 
            bw.Write(v.Z);
        }

        // MONR (Normals) - Dummy Up
        bw.Write(ToFourCC("RNOM"));
        bw.Write(vertices.Count * 12);
        foreach (var v in vertices)
        {
            bw.Write(0f);
            bw.Write(0f);
            bw.Write(1f);
        }

        // MOTV (TexCoords)
        bw.Write(ToFourCC("VTOM"));
        bw.Write(vertices.Count * 8);
        foreach (var v in vertices)
        {
            bw.Write(0f);
            bw.Write(0f);
        }

        // MOBA (Render Batches)
        bw.Write(ToFourCC("ABOM"));
        bw.Write(24);
        bw.Write((short)0); bw.Write((short)0); bw.Write((short)0); 
        bw.Write((short)0); // rX
        bw.Write(0); // startIndex
        bw.Write((ushort)(indices.Count)); // count
        bw.Write((ushort)0); // minIndex
        bw.Write((ushort)vertices.Count); // maxIndex
        bw.Write((byte)0); bw.Write((byte)0); bw.Write((short)0);

        // Fix MOGP size
        long endPos = fs.Position;
        fs.Position = sizePos;
        bw.Write((uint)(endPos - startPos));
        fs.Position = endPos;
    }

    private int ToFourCC(string fourCC)
    {
        byte[] bytes = Encoding.ASCII.GetBytes(fourCC);
        return BitConverter.ToInt32(bytes, 0);
    }

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
        bw.Write(b.Min.X);
        bw.Write(b.Min.Z); // CAaBox is usually X, Z, Y (min), X, Z, Y (max) in 3.3.5?
                           // WMO header bounds: (min.x, min.z, -min.y) ???
                           // Let's stick to X, Y, Z (Height) for now, or X, Z(Height), Y.
                           // Standard ADT chunks use X, Z, Y ordering for min/max.
                           // Let's swap Y/Z for bounds to match CAaBox convention of (X, Z(height), Y).
                           // Our Vector3 is (X, Y=West, Z=Height).
                           // So we write X, Z, Y.
        bw.Write(b.Min.Y);
        bw.Write(b.Max.X);
        bw.Write(b.Max.Z);
        bw.Write(b.Max.Y);
    }

    private BoundingBox CalculateBounds(List<Vector3> verts)
    {
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        foreach (var v in verts)
        {
            min = Vector3.Min(min, v);
            max = Vector3.Max(max, v);
        }
        return new BoundingBox(min, max);
    }

    public struct BoundingBox
    {
        public Vector3 Min;
        public Vector3 Max;
        public BoundingBox(Vector3 min, Vector3 max) { Min = min; Max = max; }
    }
}
