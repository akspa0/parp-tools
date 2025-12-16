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
    /// Input vertices are assumed to be in PM4 space (right-handed, Y-Up).
    /// Output WMO is left-handed Z-Up centered.
    /// Returned centroid is Z-Up world space.
    /// </summary>
    public Vector3 WriteWmo(string outputDir, string baseName, List<Vector3> pm4Vertices, List<int> indices)
    {
        if (pm4Vertices.Count == 0 || indices.Count == 0) return Vector3.Zero;

        // Ensure output directory
        Directory.CreateDirectory(outputDir);
        
        // 1. Convert PM4 (right-handed Y-up) to WoW (left-handed Z-up)
        // PM4: (X, Y_height, Z) -> WoW: (-X, Z, Y_height)
        // The -X flip converts from right-handed to left-handed coordinate system
        var wowVertices = pm4Vertices.Select(v => new Vector3(-v.X, v.Z, v.Y)).ToList();
        
        // 2. Reverse triangle winding to preserve face normals after X-flip
        // (indices come in sets of 3 for triangles)
        var fixedIndices = new List<int>(indices.Count);
        for (int i = 0; i + 2 < indices.Count; i += 3)
        {
            fixedIndices.Add(indices[i]);
            fixedIndices.Add(indices[i + 2]); // Swap winding: 0,1,2 -> 0,2,1
            fixedIndices.Add(indices[i + 1]);
        }

        // 3. Calculate Centroid (World Position)
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        foreach (var v in wowVertices)
        {
            min = Vector3.Min(min, v);
            max = Vector3.Max(max, v);
        }
        Vector3 centroid = (min + max) / 2f;

        // 4. Center Vertices (Local Space)
        var localVertices = wowVertices.Select(v => v - centroid).ToList();
        
        // 5. Calculate Local Bounds
        var bounds = CalculateBounds(localVertices);
        
        // Write Group File first (_000.wmo)
        string groupFileName = $"{baseName}_000.wmo";
        string rootFileName = $"{baseName}.wmo";
        
        WriteGroupFile(Path.Combine(outputDir, groupFileName), localVertices, fixedIndices, bounds);
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
        bw.Write(1); // nTextures = 1 (we have one material)
        bw.Write(1); // nGroups
        bw.Write(0); // nPortals
        bw.Write(0); // nLights
        bw.Write(0); // nDoodadNames
        bw.Write(0); // nDoodadDefs
        bw.Write(0); // nDoodadSets
        bw.Write(0x007F7F7F); // ambColor (neutral gray)
        bw.Write(0); // wmoID
        WriteBounds(bw, bounds);
        bw.Write((ushort)0); // flags
        bw.Write((ushort)0); // numLod

        // MOTX (Texture Names) - Real texture that exists in game
        string defaultTex = "tileset\\generic\\black.blp\0";
        bw.Write(ToFourCC("XTOM")); // MOTX
        bw.Write(defaultTex.Length);
        bw.Write(System.Text.Encoding.ASCII.GetBytes(defaultTex));

        // MOMT (Materials) - 64 bytes per material, 1 material for v17
        // See WMO.md SMOMaterial structure
        bw.Write(ToFourCC("TMOM")); // MOMT
        bw.Write(64); // 1 material * 64 bytes
        bw.Write((uint)0); // flags (F_UNLIT, etc.)
        bw.Write((uint)0); // shader
        bw.Write((uint)0); // blendMode
        bw.Write((uint)0); // texture_1 (offset into MOTX)
        bw.Write((uint)0xFF7F7F7F); // sidnColor (CImVector)
        bw.Write((uint)0xFF7F7F7F); // frameSidnColor
        bw.Write((uint)0); // texture_2
        bw.Write((uint)0xFFFFFFFF); // diffColor (white)
        bw.Write((uint)0); // ground_type
        bw.Write((uint)0); // texture_3
        bw.Write((uint)0); // color_2
        bw.Write((uint)0); // flags_2
        bw.Write((uint)0); // runTimeData[0]
        bw.Write((uint)0); // runTimeData[1]
        bw.Write((uint)0); // runTimeData[2]
        bw.Write((uint)0); // runTimeData[3]

        // MOGN (Group Names) - Empty
        WriteChunkHeader(bw, "NGOM", 0);

        // MOGI (Group Info) - 32 bytes per group
        bw.Write(ToFourCC("IGOM"));
        bw.Write(32); // 1 group * 32 bytes
        bw.Write(0); // flags
        WriteBounds(bw, bounds);
        bw.Write(-1); // nameIndex

        // MOSB (Skybox) - empty (Noggit expects MOSB, not MOSI!)
        WriteChunkHeader(bw, "BSOM", 0);
        
        // MOPV (Portal Verts) - empty
        WriteChunkHeader(bw, "VPOM", 0);

        // MOPT (Portal Info) - empty
        WriteChunkHeader(bw, "TPOM", 0);

        // MOPR (Portal Refs) - empty
        WriteChunkHeader(bw, "RPOM", 0);

        // MOVV (Visible Block Vertices) - empty (Noggit expects this!)
        WriteChunkHeader(bw, "VVOM", 0);

        // MOVB (Visible Blocks) - empty (Noggit expects this!)
        WriteChunkHeader(bw, "BVOM", 0);

        // MOLT (Lights) - empty
        WriteChunkHeader(bw, "TLOM", 0);

        // MODS (Doodad Sets) - empty
        WriteChunkHeader(bw, "SDOM", 0);

        // MODN (Doodad Names) - empty
        WriteChunkHeader(bw, "NDOM", 0);

        // MODD (Doodad Defs) - empty
        WriteChunkHeader(bw, "DDOM", 0);

        // MFOG (Fog) - MUST have at least 1 entry or Noggit crashes!
        // Structure: 48 bytes (0x30) per fog entry
        bw.Write(ToFourCC("GOFM")); // MFOG
        bw.Write(48); // 1 fog entry * 48 bytes
        bw.Write(0); // flags
        bw.Write(0f); bw.Write(0f); bw.Write(0f); // pos (3 floats)
        bw.Write(0f); // smaller_radius (start)
        bw.Write(0f); // larger_radius (end)
        // Fog structure (end, start_scalar, color)
        bw.Write(444.4445f); // fog end
        bw.Write(0.25f); // fog start_scalar
        bw.Write(0x00000000); // fog color (black)
        // Underwater fog
        bw.Write(222.2222f); // underwater fog end
        bw.Write(-0.5f); // underwater fog start_scalar
        bw.Write(0x00000000); // underwater fog color
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
        
        // MOGP Body - must match wmo_group_header exactly (68 bytes)
        bw.Write(0);        // group_name (uint32) - offset into MOGN
        bw.Write(0);        // descriptive_group_name (uint32) - offset into MOGN  
        bw.Write(0);        // flags (uint32) - wmo_group_flags
        WriteBounds(bw, bounds); // box1[3] + box2[3] = 24 bytes
        bw.Write((ushort)0); // portal_start
        bw.Write((ushort)0); // portal_count
        bw.Write((ushort)0); // transparency_batches_count (trans batches)
        bw.Write((ushort)0); // interior_batch_count
        bw.Write((ushort)1); // exterior_batch_count - we have 1 exterior batch
        bw.Write((ushort)0); // padding_or_batch_type_d
        bw.Write((byte)0); bw.Write((byte)0); bw.Write((byte)0); bw.Write((byte)0); // fogs[4] - 4 fog indices
        bw.Write(0);         // group_liquid (uint32)
        bw.Write(0);         // id (uint32)
        bw.Write(0);         // unk2 (int32)  
        bw.Write(0);         // unk3 (int32) - total 68 bytes

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

        // MOBA (Render Batches) - 24 bytes per batch
        bw.Write(ToFourCC("ABOM"));
        bw.Write(24); // 1 batch * 24 bytes
        // Bounding box for culling (6 shorts = 12 bytes)
        bw.Write((short)bounds.Min.X); bw.Write((short)bounds.Min.Y); bw.Write((short)bounds.Min.Z);
        bw.Write((short)bounds.Max.X); bw.Write((short)bounds.Max.Y); bw.Write((short)bounds.Max.Z);
        bw.Write((uint)0); // startIndex (uint32 for 3.3.5)
        bw.Write((ushort)indices.Count); // count
        bw.Write((ushort)0); // minIndex
        bw.Write((ushort)(vertices.Count - 1)); // maxIndex (inclusive)
        bw.Write((byte)0); // flags
        bw.Write((byte)0); // material_id = 0 (references MOMT[0])


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
