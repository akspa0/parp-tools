using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using System.Linq;

namespace WoWRollback.PM4Module.Analysis;

/// <summary>
/// Writes minimal valid M2 files (static, no animation) from raw geometry.
/// Creates a single M2 file with accompanying .skin file for 3.3.5 (version 264).
/// Centers geometry around (0,0,0) and returns the world placement position.
/// </summary>
public class Pm4M2Writer
{
    private const uint M2_MAGIC = 0x3032444D; // "MD20"
    private const uint M2_VERSION = 264; // WotLK version
    private const uint SKIN_MAGIC = 0x4E494B53; // "SKIN"

    /// <summary>
    /// Writes M2 files and returns the centroid (World Position) of the object.
    /// Input vertices are in PM4 space (right-handed, Y-Up).
    /// Output M2 is left-handed Z-Up centered.
    /// </summary>
    public Vector3 WriteM2(string outputDir, string baseName, List<Vector3> pm4Vertices, List<int> indices)
    {
        if (pm4Vertices.Count == 0 || indices.Count < 3) return Vector3.Zero;

        Directory.CreateDirectory(outputDir);

        // 1. Convert PM4 (right-handed Y-up) to WoW (left-handed Z-up)
        // PM4: (X, Y_height, Z) -> WoW: (-X, Z, Y_height)
        var wowVertices = pm4Vertices.Select(v => new Vector3(-v.X, v.Z, v.Y)).ToList();

        // 2. Reverse triangle winding to preserve face normals after X-flip
        var fixedIndices = new List<int>(indices.Count);
        for (int i = 0; i + 2 < indices.Count; i += 3)
        {
            fixedIndices.Add(indices[i]);
            fixedIndices.Add(indices[i + 2]); // Swap winding
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
        var localMin = min - centroid;
        var localMax = max - centroid;
        float radius = Math.Max(localMax.Length(), (-localMin).Length());

        // Write M2 file
        string m2Path = Path.Combine(outputDir, $"{baseName}.m2");
        string skinPath = Path.Combine(outputDir, $"{baseName}00.skin");

        WriteM2File(m2Path, baseName, localVertices, localMin, localMax, radius);
        WriteSkinFile(skinPath, localVertices.Count, fixedIndices);

        return centroid;
    }

    private void WriteM2File(string path, string name, List<Vector3> vertices, Vector3 min, Vector3 max, float radius)
    {
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);

        // We'll write header first with placeholder offsets, then write data and fix offsets
        long headerStart = fs.Position;

        // MD20 header for version 264 (WotLK)
        bw.Write(M2_MAGIC);          // magic "MD20"
        bw.Write(M2_VERSION);        // version 264
        
        // Name (offset will be updated)
        long nameArrayPos = fs.Position;
        bw.Write(name.Length + 1);   // nName (includes null terminator)
        bw.Write(0u);                // ofsName (placeholder)
        
        // Global flags
        bw.Write(0u);                // flags (no special flags)
        
        // Global sequences
        bw.Write(0u); bw.Write(0u);  // nGlobalSequences, ofsGlobalSequences
        
        // Animations
        bw.Write(0u); bw.Write(0u);  // nSequences, ofsSequences
        
        // Animation lookups
        bw.Write(0u); bw.Write(0u);  // nSequenceLookup, ofsSequenceLookup
        
        // Bones
        bw.Write(0u); bw.Write(0u);  // nBones, ofsBones
        
        // Key bone lookup
        bw.Write(0u); bw.Write(0u);  // nKeyBoneLookup, ofsKeyBoneLookup
        
        // Vertices
        long vertexArrayPos = fs.Position;
        bw.Write((uint)vertices.Count); // nVertices
        bw.Write(0u);                   // ofsVertices (placeholder)
        
        // Views (skins) - we have 1 view
        bw.Write(1u);                // nViews = 1
        
        // Colors
        bw.Write(0u); bw.Write(0u);  // nColors, ofsColors
        
        // Textures
        long textureArrayPos = fs.Position;
        bw.Write(1u);                // nTextures = 1 (default texture)
        bw.Write(0u);                // ofsTextures (placeholder)
        
        // Transparency
        bw.Write(0u); bw.Write(0u);  // nTransparency, ofsTransparency
        
        // Texture animations
        bw.Write(0u); bw.Write(0u);  // nTexAnims, ofsTexAnims
        
        // Replaceable textures
        bw.Write(0u); bw.Write(0u);  // nReplaceableTextures, ofsReplaceableTextures
        
        // Materials (render flags)
        long materialArrayPos = fs.Position;
        bw.Write(1u);                // nMaterials = 1
        bw.Write(0u);                // ofsMaterials (placeholder)
        
        // Bone combos
        bw.Write(0u); bw.Write(0u);  // nBoneCombos, ofsBoneCombos
        
        // Texture combos
        long texCombosPos = fs.Position;
        bw.Write(1u);                // nTextureCombos = 1
        bw.Write(0u);                // ofsTextureCombos (placeholder)
        
        // Texture coord combos
        long texCoordCombosPos = fs.Position;
        bw.Write(1u);                // nTextureCoordCombos = 1
        bw.Write(0u);                // ofsTextureCoordCombos (placeholder)
        
        // Texture weights
        bw.Write(0u); bw.Write(0u);  // nTextureWeights, ofsTextureWeights
        
        // Texture transforms
        bw.Write(0u); bw.Write(0u);  // nTextureTransforms, ofsTextureTransforms
        
        // Bounding box
        WriteVector3(bw, min);
        WriteVector3(bw, max);
        bw.Write(radius);            // bounding sphere radius
        
        // Collision box (same as bounding)
        WriteVector3(bw, min);
        WriteVector3(bw, max);
        bw.Write(radius);            // collision sphere radius
        
        // Collision triangles
        bw.Write(0u); bw.Write(0u);  // nBoundingTriangles, ofsBoundingTriangles
        
        // Collision vertices
        bw.Write(0u); bw.Write(0u);  // nBoundingVertices, ofsBoundingVertices
        
        // Collision normals
        bw.Write(0u); bw.Write(0u);  // nBoundingNormals, ofsBoundingNormals
        
        // Attachments
        bw.Write(0u); bw.Write(0u);  // nAttachments, ofsAttachments
        
        // Attachment lookups
        bw.Write(0u); bw.Write(0u);  // nAttachmentLookups, ofsAttachmentLookups
        
        // Events
        bw.Write(0u); bw.Write(0u);  // nEvents, ofsEvents
        
        // Lights
        bw.Write(0u); bw.Write(0u);  // nLights, ofsLights
        
        // Cameras
        bw.Write(0u); bw.Write(0u);  // nCameras, ofsCameras
        
        // Camera lookups
        bw.Write(0u); bw.Write(0u);  // nCameraLookups, ofsCameraLookups
        
        // Ribbon emitters
        bw.Write(0u); bw.Write(0u);  // nRibbonEmitters, ofsRibbonEmitters
        
        // Particle emitters
        bw.Write(0u); bw.Write(0u);  // nParticleEmitters, ofsParticleEmitters
        
        // Texture combiner combos (only if flag set)
        // Not used for static models
        
        // --- DATA SECTION ---
        
        // Write name string
        uint ofsName = (uint)fs.Position;
        bw.Write(Encoding.ASCII.GetBytes(name));
        bw.Write((byte)0); // null terminator
        
        // Write vertices (M2Vertex structure: 48 bytes each)
        // Align to 16 bytes
        while (fs.Position % 16 != 0) bw.Write((byte)0);
        uint ofsVertices = (uint)fs.Position;
        foreach (var v in vertices)
        {
            WriteVector3(bw, v);           // position
            bw.Write((byte)0); bw.Write((byte)0); bw.Write((byte)0); bw.Write((byte)0); // bone weights (none)
            bw.Write((byte)0); bw.Write((byte)0); bw.Write((byte)0); bw.Write((byte)0); // bone indices
            WriteVector3(bw, new Vector3(0, 0, 1)); // normal (up)
            bw.Write(0f); bw.Write(0f);    // texcoord 1
            bw.Write(0f); bw.Write(0f);    // texcoord 2
        }
        
        // Write texture definition (16 bytes)
        uint ofsTextures = (uint)fs.Position;
        bw.Write(0u);                      // type = 0 (hardcoded filename)
        bw.Write(0u);                      // flags
        bw.Write(0u);                      // filename length (empty = default)
        bw.Write(0u);                      // filename offset
        
        // Write material (render flags) - 4 bytes
        uint ofsMaterials = (uint)fs.Position;
        bw.Write((ushort)0);               // flags
        bw.Write((ushort)0);               // blendMode = opaque
        
        // Write texture combos - 2 bytes
        uint ofsTextureCombos = (uint)fs.Position;
        bw.Write((ushort)0);               // texture index 0
        
        // Write texture coord combos - 2 bytes
        uint ofsTextureCoordCombos = (uint)fs.Position;
        bw.Write((ushort)0);               // texunit 0
        
        // --- FIX OFFSETS ---
        fs.Position = nameArrayPos + 4;
        bw.Write(ofsName);
        
        fs.Position = vertexArrayPos + 4;
        bw.Write(ofsVertices);
        
        fs.Position = textureArrayPos + 4;
        bw.Write(ofsTextures);
        
        fs.Position = materialArrayPos + 4;
        bw.Write(ofsMaterials);
        
        fs.Position = texCombosPos + 4;
        bw.Write(ofsTextureCombos);
        
        fs.Position = texCoordCombosPos + 4;
        bw.Write(ofsTextureCoordCombos);
    }

    private void WriteSkinFile(string path, int vertexCount, List<int> indices)
    {
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);

        // SKIN header for WotLK
        bw.Write(SKIN_MAGIC);        // "SKIN"
        
        // Indices (vertex lookup) - all vertices
        bw.Write((uint)vertexCount); // nIndices
        long ofsIndicesPos = fs.Position;
        bw.Write(0u);                // ofsIndices (placeholder)
        
        // Triangles
        bw.Write((uint)indices.Count); // nTriangles
        long ofsTrianglesPos = fs.Position;
        bw.Write(0u);                // ofsTriangles (placeholder)
        
        // Bone lookup (vertex properties)
        bw.Write((uint)vertexCount); // nBones
        long ofsBonesPos = fs.Position;
        bw.Write(0u);                // ofsBones (placeholder)
        
        // Submeshes
        bw.Write(1u);                // nSubmeshes = 1
        long ofsSubmeshesPos = fs.Position;
        bw.Write(0u);                // ofsSubmeshes (placeholder)
        
        // Texture units (batches)
        bw.Write(1u);                // nBatches = 1
        long ofsBatchesPos = fs.Position;
        bw.Write(0u);                // ofsBatches (placeholder)
        
        // LOD
        bw.Write(0u);                // boneCountMax
        
        // --- DATA ---
        
        // Vertex indices
        uint ofsIndices = (uint)fs.Position;
        for (int i = 0; i < vertexCount; i++)
            bw.Write((ushort)i);
        
        // Triangles
        uint ofsTriangles = (uint)fs.Position;
        foreach (var idx in indices)
            bw.Write((ushort)idx);
        
        // Bone lookup (4 bytes per vertex)
        uint ofsBones = (uint)fs.Position;
        for (int i = 0; i < vertexCount; i++)
        {
            bw.Write((byte)0); bw.Write((byte)0); bw.Write((byte)0); bw.Write((byte)0);
        }
        
        // Submesh (48 bytes for WotLK)
        uint ofsSubmeshes = (uint)fs.Position;
        bw.Write(0u);                // ID
        bw.Write((ushort)0);         // startVertex
        bw.Write((ushort)vertexCount); // nVertices
        bw.Write((ushort)0);         // startTriangle
        bw.Write((ushort)indices.Count); // nTriangles
        bw.Write((ushort)0);         // nBones
        bw.Write((ushort)0);         // startBones
        bw.Write((ushort)0);         // unknown
        bw.Write((ushort)0);         // rootBone
        bw.Write(0f); bw.Write(0f); bw.Write(0f); // centerMass
        bw.Write(0f); bw.Write(0f); bw.Write(0f); bw.Write(0f); // unknown floats
        
        // Texture unit (batch) - 24 bytes for WotLK
        uint ofsBatches = (uint)fs.Position;
        bw.Write((byte)0);           // flags
        bw.Write((byte)0);           // priority
        bw.Write((ushort)0);         // shader
        bw.Write((ushort)0);         // submesh index
        bw.Write((ushort)0);         // submesh index 2
        bw.Write((short)-1);         // color index (-1 = none)
        bw.Write((ushort)0);         // material index
        bw.Write((ushort)0);         // material layer
        bw.Write((ushort)1);         // texture count
        bw.Write((ushort)0);         // texture combo index
        bw.Write((ushort)0);         // texture coord combo index
        bw.Write((ushort)0);         // texture weight combo index
        bw.Write((ushort)0);         // texture transform combo index
        
        // --- FIX OFFSETS ---
        fs.Position = ofsIndicesPos;
        bw.Write(ofsIndices);
        
        fs.Position = ofsTrianglesPos;
        bw.Write(ofsTriangles);
        
        fs.Position = ofsBonesPos;
        bw.Write(ofsBones);
        
        fs.Position = ofsSubmeshesPos;
        bw.Write(ofsSubmeshes);
        
        fs.Position = ofsBatchesPos;
        bw.Write(ofsBatches);
    }

    private static void WriteVector3(BinaryWriter bw, Vector3 v)
    {
        bw.Write(v.X);
        bw.Write(v.Y);
        bw.Write(v.Z);
    }
}
