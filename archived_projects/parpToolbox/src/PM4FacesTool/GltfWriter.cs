using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Text.Json;

namespace PM4FacesTool;

internal static class GltfWriter
{
    public static void WriteGltf(
        string gltfPath,
        IReadOnlyList<Vector3> vertices,
        IReadOnlyList<(int A, int B, int C)> triangles,
        bool legacyParity,
        bool projectLocal,
        bool forceFlipX)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(gltfPath)!);
        string binPath = Path.ChangeExtension(gltfPath, ".bin");
        WriteGltfAndBin(gltfPath, binPath, vertices, triangles, legacyParity, projectLocal, forceFlipX);
    }

    public static void WriteGlb(
        string glbPath,
        IReadOnlyList<Vector3> vertices,
        IReadOnlyList<(int A, int B, int C)> triangles,
        bool legacyParity,
        bool projectLocal,
        bool forceFlipX)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(glbPath)!);

        // Generate JSON and BIN in-memory
        using var jsonMs = new MemoryStream();
        using var binMs = new MemoryStream();
        BuildGltfStreams(jsonMs, binMs, Path.GetFileNameWithoutExtension(glbPath), vertices, triangles, legacyParity, projectLocal, forceFlipX);

        // Pack into GLB (glTF 2.0)
        var jsonBytes = jsonMs.ToArray();
        var binBytes = binMs.ToArray();

        // 4-byte padding
        int Pad4(int x) => (x + 3) & ~3;
        int jsonPaddedLen = Pad4(jsonBytes.Length);
        int binPaddedLen = Pad4(binBytes.Length);

        using var fs = new FileStream(glbPath, FileMode.Create, FileAccess.Write);
        using var bw = new BinaryWriter(fs);

        // Header
        bw.Write(0x46546C67); // magic 'glTF'
        bw.Write(2);          // version
        bw.Write(12 + 8 + jsonPaddedLen + 8 + binPaddedLen); // total length

        // JSON chunk
        bw.Write(jsonPaddedLen);
        bw.Write(0x4E4F534A); // 'JSON'
        bw.Write(jsonBytes);
        for (int i = jsonBytes.Length; i < jsonPaddedLen; i++) bw.Write((byte)0x20); // spaces

        // BIN chunk
        bw.Write(binPaddedLen);
        bw.Write(0x004E4942); // 'BIN\0'
        bw.Write(binBytes);
        for (int i = binBytes.Length; i < binPaddedLen; i++) bw.Write((byte)0x00);
    }

    private static void WriteGltfAndBin(
        string gltfPath,
        string binPath,
        IReadOnlyList<Vector3> vertices,
        IReadOnlyList<(int A, int B, int C)> triangles,
        bool legacyParity,
        bool projectLocal,
        bool forceFlipX)
    {
        using var jsonMs = new MemoryStream();
        using var binMs = new MemoryStream();
        BuildGltfStreams(jsonMs, binMs, Path.GetFileNameWithoutExtension(gltfPath), vertices, triangles, legacyParity, projectLocal, forceFlipX);

        File.WriteAllBytes(binPath, binMs.ToArray());
        File.WriteAllText(gltfPath, Encoding.UTF8.GetString(jsonMs.ToArray()));
    }

    private static void BuildGltfStreams(
        Stream jsonOut,
        Stream binOut,
        string name,
        IReadOnlyList<Vector3> vertices,
        IReadOnlyList<(int A, int B, int C)> triangles,
        bool legacyParity,
        bool projectLocal,
        bool forceFlipX)
    {
        // Prepare transformed vertices
        var verts = new Vector3[vertices.Count];
        if (projectLocal && vertices.Count > 0)
        {
            var mean = ComputeMean(vertices);
            for (int i = 0; i < vertices.Count; i++)
                verts[i] = vertices[i] - mean;
        }
        else
        {
            for (int i = 0; i < vertices.Count; i++) verts[i] = vertices[i];
        }

        bool flip = legacyParity || forceFlipX;
        if (flip)
        {
            for (int i = 0; i < verts.Length; i++)
            {
                var v = verts[i];
                verts[i] = new Vector3(-v.X, v.Y, v.Z);
            }
        }

        // Flatten triangles, swapping b/c if flipped to preserve winding
        var indexList = new List<uint>(Math.Max(0, triangles.Count * 3));
        for (int i = 0; i < triangles.Count; i++)
        {
            int a = triangles[i].A;
            int b = triangles[i].B;
            int c = triangles[i].C;
            if (flip) (b, c) = (c, b);
            if (!IsValidIndex(a, verts.Length) || !IsValidIndex(b, verts.Length) || !IsValidIndex(c, verts.Length))
                continue;
            indexList.Add((uint)a);
            indexList.Add((uint)b);
            indexList.Add((uint)c);
        }

        // Build BIN buffer: positions then indices (both 4-byte aligned)
        using var bw = new BinaryWriter(binOut, Encoding.UTF8, leaveOpen: true);
        // Positions
        int posOffset = 0;
        foreach (var v in verts)
        {
            bw.Write(v.X);
            bw.Write(v.Y);
            bw.Write(v.Z);
        }
        int posByteLength = verts.Length * sizeof(float) * 3;
        PadTo4(bw, posByteLength);
        int posPaddedLength = Pad4(posByteLength);

        // Indices (uint)
        int idxOffset = posOffset + posPaddedLength;
        foreach (var idx in indexList) bw.Write(idx);
        int idxByteLength = indexList.Count * sizeof(uint);
        PadTo4(bw, idxByteLength);

        // Prepare JSON
        var (min, max) = ComputeMinMax(verts);
        int vertexCount = verts.Length;
        int indexCount = indexList.Count;
        int bufferByteLength = idxOffset + Pad4(idxByteLength);

        var gltf = new
        {
            asset = new { version = "2.0", generator = "PM4FacesTool" },
            buffers = new object[]
            {
                new { byteLength = bufferByteLength }
            },
            bufferViews = new object[]
            {
                new { buffer = 0, byteOffset = posOffset, byteLength = posByteLength, target = 34962 }, // ARRAY_BUFFER
                new { buffer = 0, byteOffset = idxOffset, byteLength = idxByteLength, target = 34963 }  // ELEMENT_ARRAY_BUFFER
            },
            accessors = new object[]
            {
                new
                {
                    bufferView = 0,
                    byteOffset = 0,
                    componentType = 5126, // FLOAT
                    count = vertexCount,
                    type = "VEC3",
                    min = new [] { min.X, min.Y, min.Z },
                    max = new [] { max.X, max.Y, max.Z }
                },
                new
                {
                    bufferView = 1,
                    byteOffset = 0,
                    componentType = 5125, // UNSIGNED_INT
                    count = indexCount,
                    type = "SCALAR"
                }
            },
            meshes = new object[]
            {
                new
                {
                    primitives = new object[]
                    {
                        new
                        {
                            attributes = new { POSITION = 0 },
                            indices = 1,
                            mode = 4 // TRIANGLES
                        }
                    }
                }
            },
            nodes = new object[]
            {
                new { mesh = 0, name = SafeName(name) }
            },
            scenes = new object[]
            {
                new { nodes = new [] { 0 } }
            },
            scene = 0
        };

        var options = new JsonSerializerOptions { WriteIndented = true };
        var json = JsonSerializer.Serialize(gltf, options);
        using var sw = new StreamWriter(jsonOut, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false), leaveOpen: true);
        sw.Write(json);
        sw.Flush();
    }

    private static (Vector3 min, Vector3 max) ComputeMinMax(IReadOnlyList<Vector3> verts)
    {
        if (verts.Count == 0) return (new Vector3(0,0,0), new Vector3(0,0,0));
        float minX = verts[0].X, minY = verts[0].Y, minZ = verts[0].Z;
        float maxX = minX, maxY = minY, maxZ = minZ;
        for (int i = 1; i < verts.Count; i++)
        {
            var v = verts[i];
            if (v.X < minX) minX = v.X; if (v.X > maxX) maxX = v.X;
            if (v.Y < minY) minY = v.Y; if (v.Y > maxY) maxY = v.Y;
            if (v.Z < minZ) minZ = v.Z; if (v.Z > maxZ) maxZ = v.Z;
        }
        return (new Vector3(minX, minY, minZ), new Vector3(maxX, maxY, maxZ));
    }

    private static Vector3 ComputeMean(IReadOnlyList<Vector3> vertices)
    {
        double sx = 0, sy = 0, sz = 0;
        for (int i = 0; i < vertices.Count; i++)
        {
            sx += vertices[i].X;
            sy += vertices[i].Y;
            sz += vertices[i].Z;
        }
        double inv = 1.0 / Math.Max(1, vertices.Count);
        return new Vector3((float)(sx * inv), (float)(sy * inv), (float)(sz * inv));
    }

    private static void PadTo4(BinaryWriter bw, int currentLength)
    {
        int padded = Pad4(currentLength);
        for (int i = currentLength; i < padded; i++) bw.Write((byte)0x00);
    }

    private static int Pad4(int x) => (x + 3) & ~3;

    private static bool IsValidIndex(int i, int count) => i >= 0 && i < count;

    private static string SafeName(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var s = string.Join("_", name.Split(invalid, StringSplitOptions.RemoveEmptyEntries)).TrimEnd('.');
        return string.IsNullOrWhiteSpace(s) ? "object" : s;
    }
}
