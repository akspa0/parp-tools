using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Text.Json;

namespace ParpDataHarvester.Export
{
    internal static class GltfRawWriter
    {
        internal sealed class PrimitiveSpec
        {
            public string? Name { get; init; }
            public List<uint> Indices { get; } = new();
            public Dictionary<string, object>? Extras { get; init; }
        }

        public static void WriteGlb(
            string glbPath,
            IReadOnlyList<Vector3> vertices,
            IReadOnlyList<PrimitiveSpec> primitives,
            string? sceneName = null)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(glbPath)!);

            using var jsonMs = new MemoryStream();
            using var binMs = new MemoryStream();

            BuildGltfStreams(jsonMs, binMs, Path.GetFileNameWithoutExtension(glbPath), vertices, primitives, sceneName);

            var jsonBytes = jsonMs.ToArray();
            var binBytes = binMs.ToArray();

            int Pad4(int x) => (x + 3) & ~3;
            int jsonPaddedLen = Pad4(jsonBytes.Length);
            int binPaddedLen = Pad4(binBytes.Length);

            using var fs = new FileStream(glbPath, FileMode.Create, FileAccess.Write);
            using var bw = new BinaryWriter(fs);

            // Header
            bw.Write(0x46546C67); // 'glTF'
            bw.Write(2);          // version
            bw.Write(12 + 8 + jsonPaddedLen + 8 + binPaddedLen);

            // JSON chunk
            bw.Write(jsonPaddedLen);
            bw.Write(0x4E4F534A); // 'JSON'
            bw.Write(jsonBytes);
            for (int i = jsonBytes.Length; i < jsonPaddedLen; i++) bw.Write((byte)0x20);

            // BIN chunk
            bw.Write(binPaddedLen);
            bw.Write(0x004E4942); // 'BIN\0'
            bw.Write(binBytes);
            for (int i = binBytes.Length; i < binPaddedLen; i++) bw.Write((byte)0x00);
        }

        private static void BuildGltfStreams(
            Stream jsonOut,
            Stream binOut,
            string defaultName,
            IReadOnlyList<Vector3> vertices,
            IReadOnlyList<PrimitiveSpec> primitives,
            string? sceneName)
        {
            using var bw = new BinaryWriter(binOut, Encoding.UTF8, leaveOpen: true);

            // Positions block (vertices are already oriented by upstream)
            int posOffset = 0;
            foreach (var v in vertices)
            {
                bw.Write(v.X);
                bw.Write(v.Y);
                bw.Write(v.Z);
            }
            int posByteLength = vertices.Count * sizeof(float) * 3;
            PadTo4(bw, posByteLength);
            int posPaddedLength = Pad4(posByteLength);

            // Indices block (concatenate all primitive indices)
            int idxOffset = posOffset + posPaddedLength;
            var primitiveOffsets = new List<(int byteOffset, int count)>();
            foreach (var prim in primitives)
            {
                int startByte = (int)binOut.Position;
                foreach (var idx in prim.Indices)
                    bw.Write(idx);
                int count = prim.Indices.Count;
                primitiveOffsets.Add((startByte, count));
            }
            int idxByteLength = (int)(binOut.Position - idxOffset);
            PadTo4(bw, idxByteLength);

            // Compute min/max of vertices
            var (min, max) = ComputeMinMax(vertices);

            // JSON construction
            int bufferByteLength = idxOffset + Pad4(idxByteLength);

            // Accessor 0: POSITION
            var accessors = new List<object>
            {
                new
                {
                    bufferView = 0,
                    byteOffset = 0,
                    componentType = 5126, // FLOAT
                    count = vertices.Count,
                    type = "VEC3",
                    min = new [] { min.X, min.Y, min.Z },
                    max = new [] { max.X, max.Y, max.Z }
                }
            };

            // BufferViews: 0 = positions, 1 = indices (all primitives)
            var bufferViews = new List<object>
            {
                new { buffer = 0, byteOffset = posOffset, byteLength = posByteLength, target = 34962 },
                new { buffer = 0, byteOffset = idxOffset, byteLength = idxByteLength, target = 34963 }
            };

            // Build index accessors per primitive
            int indicesBufferViewIndex = 1;
            foreach (var (byteOffset, count) in primitiveOffsets)
            {
                accessors.Add(new
                {
                    bufferView = indicesBufferViewIndex,
                    byteOffset = byteOffset - idxOffset,
                    componentType = 5125, // UNSIGNED_INT
                    count = count,
                    type = "SCALAR"
                });
            }

            // Mesh/primitives JSON
            var primitiveJson = new List<object>();
            for (int i = 0; i < primitives.Count; i++)
            {
                var prim = primitives[i];
                int indicesAccessorIndex = 1 + i; // 0 is POSITION
                var primObj = new Dictionary<string, object>
                {
                    ["attributes"] = new { POSITION = 0 },
                    ["indices"] = indicesAccessorIndex,
                    ["mode"] = 4, // TRIANGLES
                    ["material"] = 0
                };
                if (prim.Extras is not null && prim.Extras.Count > 0)
                {
                    primObj["extras"] = prim.Extras;
                }
                primitiveJson.Add(primObj);
            }

            var mesh = new
            {
                primitives = primitiveJson.ToArray()
            };

            var nodes = new object[]
            {
                new { mesh = 0, name = SafeName(sceneName ?? defaultName) }
            };

            var gltf = new
            {
                asset = new { version = "2.0", generator = "parpDataHarvester" },
                buffers = new object[] { new { byteLength = bufferByteLength } },
                bufferViews = bufferViews.ToArray(),
                accessors = accessors.ToArray(),
                meshes = new object[] { mesh },
                materials = new object[] { new { name = "Default", doubleSided = true } },
                nodes = nodes,
                scenes = new object[] { new { nodes = new[] { 0 } } },
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
            if (verts.Count == 0) return (new Vector3(0, 0, 0), new Vector3(0, 0, 0));
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

        private static void PadTo4(BinaryWriter bw, int currentLength)
        {
            int padded = Pad4(currentLength);
            for (int i = currentLength; i < padded; i++) bw.Write((byte)0x00);
        }

        private static int Pad4(int x) => (x + 3) & ~3;

        private static string SafeName(string name)
        {
            var invalid = Path.GetInvalidFileNameChars();
            var s = string.Join("_", name.Split(invalid, StringSplitOptions.RemoveEmptyEntries)).TrimEnd('.');
            return string.IsNullOrWhiteSpace(s) ? "object" : s;
        }
    }
}
