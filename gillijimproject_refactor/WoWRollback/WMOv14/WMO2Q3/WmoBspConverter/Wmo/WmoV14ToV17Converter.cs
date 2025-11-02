using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;

namespace WmoBspConverter.Wmo
{
    /// <summary>
    /// Converts WMO v14 format to v17 format for compatibility with wow.tools exporters.
    /// This allows us to leverage existing v17 export infrastructure while preserving
    /// the correct material assignments we fixed for v14 (MOBA-based).
    /// </summary>
    public class WmoV14ToV17Converter
    {
        /// <summary>
        /// Convert parsed v14 WMO data to v17 format and write to disk.
        /// </summary>
        public void ConvertAndWrite(WmoV14Parser.WmoV14Data v14Data, string outputPath)
        {
            Console.WriteLine($"[INFO] Converting WMO v14 → v17 format...");
            
            // Create output directory
            var outputDir = Path.GetDirectoryName(outputPath) ?? ".";
            Directory.CreateDirectory(outputDir);
            
            // Write root WMO file
            WriteRootFile(v14Data, outputPath);
            
            // Write group files
            WriteGroupFiles(v14Data, outputPath);
            
            Console.WriteLine($"[SUCCESS] Converted to v17 format: {outputPath}");
        }
        
        private void WriteRootFile(WmoV14Parser.WmoV14Data v14Data, string outputPath)
        {
            using var stream = File.Create(outputPath);
            using var writer = new BinaryWriter(stream);
            
            // Write MVER (version chunk) - v17
            WriteChunk(writer, "MVER", w =>
            {
                w.Write((uint)17); // Target version
            });
            
            // Write MOHD (header)
            WriteChunk(writer, "MOHD", w =>
            {
                w.Write((uint)v14Data.Materials.Count);     // nMaterials
                w.Write((uint)v14Data.Groups.Count);        // nGroups
                w.Write((uint)0);                           // nPortals (v14 doesn't have portals)
                w.Write((uint)0);                           // nLights
                w.Write((uint)0);                           // nModels
                w.Write((uint)0);                           // nDoodads
                w.Write((uint)1);                           // nSets
                w.Write((uint)0);                           // ambientColor
                w.Write((uint)0);                           // areaTableID
                
                // Bounding box - compute from all vertices
                var bounds = ComputeBounds(v14Data);
                WriteVector3(w, bounds.min);
                WriteVector3(w, bounds.max);
                
                w.Write((short)0);                          // flags
                w.Write((short)0);                          // nLod
            });
            
            // Write MOTX (textures)
            WriteChunk(writer, "MOTX", w =>
            {
                foreach (var texture in v14Data.Textures)
                {
                    var bytes = Encoding.UTF8.GetBytes(texture);
                    w.Write(bytes);
                    w.Write((byte)0); // null terminator
                }
            });
            
            // Write MOMT (materials) - expand v14 to v17 format
            WriteChunk(writer, "MOMT", w =>
            {
                foreach (var material in v14Data.Materials)
                {
                    WriteMaterialV17(w, material, v14Data);
                }
            });
            
            // Write MOGN (group names)
            WriteChunk(writer, "MOGN", w =>
            {
                foreach (var group in v14Data.Groups)
                {
                    var bytes = Encoding.UTF8.GetBytes(group.Name);
                    w.Write(bytes);
                    w.Write((byte)0);
                }
            });
            
            // Write MOGI (group info)
            WriteChunk(writer, "MOGI", w =>
            {
                foreach (var group in v14Data.Groups)
                {
                    w.Write((uint)group.Flags);             // flags
                    
                    // Bounding box for this group
                    var groupBounds = ComputeGroupBounds(group);
                    WriteVector3(w, groupBounds.min);
                    WriteVector3(w, groupBounds.max);
                    
                    w.Write((int)0);                        // nameOffset (filled during parsing)
                }
            });
            
            // Write MODS (doodad sets) - minimal
            WriteChunk(writer, "MODS", w =>
            {
                // Write one empty doodad set
                var bytes = Encoding.UTF8.GetBytes("Set_$DefaultGlobal");
                w.Write(bytes);
                w.Write((byte)0);
                w.Write((uint)0); // firstInstanceIndex
                w.Write((uint)0); // numDoodads
                w.Write((uint)0); // unused
            });
            
            Console.WriteLine($"[DEBUG] Wrote v17 root file with {v14Data.Groups.Count} groups, {v14Data.Materials.Count} materials");
        }
        
        private void WriteGroupFiles(WmoV14Parser.WmoV14Data v14Data, string rootPath)
        {
            var baseName = Path.GetFileNameWithoutExtension(rootPath);
            var outputDir = Path.GetDirectoryName(rootPath) ?? ".";
            
            for (int i = 0; i < v14Data.Groups.Count; i++)
            {
                var group = v14Data.Groups[i];
                var groupPath = Path.Combine(outputDir, $"{baseName}_{i:D3}.wmo");
                
                WriteGroupFile(group, groupPath, i);
            }
        }
        
        private void WriteGroupFile(WmoV14Parser.WmoGroupData group, string outputPath, int groupIndex)
        {
            using var stream = File.Create(outputPath);
            using var writer = new BinaryWriter(stream);
            
            // Write MVER
            WriteChunk(writer, "MVER", w => w.Write((uint)17));
            
            // Write MOGP (group header + subchunks)
            var mogpStart = writer.BaseStream.Position;
            writer.Write(Encoding.ASCII.GetBytes("MOGP"));
            var sizePos = writer.BaseStream.Position;
            writer.Write((uint)0); // Placeholder for size
            
            var mogpDataStart = writer.BaseStream.Position;
            
            // MOGP header (68 bytes)
            writer.Write((uint)0);              // nameOffset
            writer.Write((uint)0);              // descriptiveNameOffset
            writer.Write((uint)group.Flags);    // flags
            
            var bounds = ComputeGroupBounds(group);
            WriteVector3(writer, bounds.min);
            WriteVector3(writer, bounds.max);
            
            writer.Write((ushort)0);            // portalStart
            writer.Write((ushort)0);            // portalCount
            writer.Write((ushort)0);            // transBatchCount
            writer.Write((ushort)0);            // intBatchCount
            writer.Write((ushort)0);            // extBatchCount
            writer.Write((ushort)0);            // padding
            writer.Write(new byte[4]);          // fogIndices
            writer.Write((uint)0);              // liquidType
            writer.Write((uint)0);              // groupID
            writer.Write((uint)0);              // unused
            writer.Write((uint)0);              // unused
            
            // Write MOPY (face materials) - use our MOBA-based assignments
            WriteSubChunk(writer, "MOPY", w =>
            {
                foreach (var matId in group.FaceMaterials)
                {
                    w.Write((byte)0x00);        // flags (renderable)
                    w.Write(matId);             // material ID from MOBA
                }
            });
            
            // Write MOVI (indices)
            WriteSubChunk(writer, "MOVI", w =>
            {
                foreach (var idx in group.Indices)
                {
                    w.Write(idx);
                }
            });
            
            // Write MOVT (vertices)
            WriteSubChunk(writer, "MOVT", w =>
            {
                foreach (var vertex in group.Vertices)
                {
                    WriteVector3(w, vertex);
                }
            });
            
            // Write MONR (normals) - generate if not present
            WriteSubChunk(writer, "MONR", w =>
            {
                if (group.Vertices.Count > 0)
                {
                    // Generate normals from triangles
                    var normals = GenerateNormals(group);
                    foreach (var normal in normals)
                    {
                        WriteVector3(w, normal);
                    }
                }
            });
            
            // Write MOTV (texture coordinates)
            WriteSubChunk(writer, "MOTV", w =>
            {
                foreach (var uv in group.UVs)
                {
                    w.Write(uv.X);
                    w.Write(uv.Y);
                }
            });
            
            // Write MOBA (render batches) - convert from v14 MOBA
            WriteSubChunk(writer, "MOBA", w =>
            {
                foreach (var batch in group.Batches)
                {
                    // v17 MOBA is 24 bytes
                    w.Write((ushort)0);                     // possibleBox1_1
                    w.Write((ushort)0);                     // possibleBox1_2
                    w.Write((ushort)0);                     // possibleBox1_3
                    w.Write((ushort)0);                     // possibleBox2_1
                    w.Write((ushort)0);                     // possibleBox2_2
                    w.Write((ushort)0);                     // possibleBox2_3
                    w.Write((uint)batch.FirstFace);         // startIndex
                    w.Write((ushort)batch.NumFaces);        // count
                    w.Write((ushort)batch.FirstVertex);     // minIndex
                    w.Write((ushort)batch.LastVertex);      // maxIndex
                    w.Write((byte)batch.Flags);             // flags
                    w.Write((byte)batch.MaterialId);        // materialID
                }
            });
            
            // Update MOGP size
            var mogpEnd = writer.BaseStream.Position;
            var mogpSize = (uint)(mogpEnd - mogpDataStart);
            writer.BaseStream.Position = sizePos;
            writer.Write(mogpSize);
            writer.BaseStream.Position = mogpEnd;
            
            Console.WriteLine($"[DEBUG] Wrote group {groupIndex}: {group.Vertices.Count} verts, {group.Indices.Count} indices, {group.FaceMaterials.Count} faces");
        }
        
        private void WriteMaterialV17(BinaryWriter writer, WmoMaterial material, WmoV14Parser.WmoV14Data v14Data)
        {
            // v17 MOMT is 64 bytes
            writer.Write((uint)material.Flags);                 // flags
            writer.Write((uint)material.Shader);                // shader
            writer.Write((uint)material.BlendMode);             // blendMode
            writer.Write((uint)material.Texture1Offset);        // texture1
            writer.Write((uint)material.EmissiveColor);         // sidnColor (emissive)
            writer.Write((uint)0);                              // frameSidnColor (emissive)
            writer.Write((uint)material.Texture2Offset);        // texture2
            writer.Write((uint)material.DiffuseColor);          // diffColor
            writer.Write((uint)material.GroundType);            // groundType
            writer.Write((uint)material.Texture3Offset);        // texture3
            writer.Write((uint)0);                              // color2
            writer.Write((uint)0);                              // flags2
            writer.Write(new byte[16]);                         // runTimeData (4 uints)
        }
        
        private void WriteChunk(BinaryWriter writer, string chunkId, Action<BinaryWriter> writeData)
        {
            var chunkIdBytes = Encoding.ASCII.GetBytes(chunkId);
            if (chunkIdBytes.Length != 4)
                throw new ArgumentException($"Chunk ID must be 4 characters: {chunkId}");
            
            writer.Write(chunkIdBytes);
            
            // Write size placeholder
            var sizePos = writer.BaseStream.Position;
            writer.Write((uint)0);
            
            var dataStart = writer.BaseStream.Position;
            writeData(writer);
            var dataEnd = writer.BaseStream.Position;
            
            // Update size
            var size = (uint)(dataEnd - dataStart);
            writer.BaseStream.Position = sizePos;
            writer.Write(size);
            writer.BaseStream.Position = dataEnd;
        }
        
        private void WriteSubChunk(BinaryWriter writer, string chunkId, Action<BinaryWriter> writeData)
        {
            // Subchunks are the same as chunks
            WriteChunk(writer, chunkId, writeData);
        }
        
        private void WriteVector3(BinaryWriter writer, Vector3 v)
        {
            writer.Write(v.X);
            writer.Write(v.Y);
            writer.Write(v.Z);
        }
        
        private (Vector3 min, Vector3 max) ComputeBounds(WmoV14Parser.WmoV14Data v14Data)
        {
            var min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            var max = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            
            foreach (var group in v14Data.Groups)
            {
                foreach (var vertex in group.Vertices)
                {
                    min = Vector3.Min(min, vertex);
                    max = Vector3.Max(max, vertex);
                }
            }
            
            return (min, max);
        }
        
        private (Vector3 min, Vector3 max) ComputeGroupBounds(WmoV14Parser.WmoGroupData group)
        {
            if (group.Vertices.Count == 0)
                return (Vector3.Zero, Vector3.Zero);
            
            var min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            var max = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            
            foreach (var vertex in group.Vertices)
            {
                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
            }
            
            return (min, max);
        }
        
        private List<Vector3> GenerateNormals(WmoV14Parser.WmoGroupData group)
        {
            var normals = new List<Vector3>(new Vector3[group.Vertices.Count]);
            
            // Generate face normals and accumulate
            for (int i = 0; i + 2 < group.Indices.Count; i += 3)
            {
                var i0 = group.Indices[i];
                var i1 = group.Indices[i + 1];
                var i2 = group.Indices[i + 2];
                
                if (i0 >= group.Vertices.Count || i1 >= group.Vertices.Count || i2 >= group.Vertices.Count)
                    continue;
                
                var v0 = group.Vertices[i0];
                var v1 = group.Vertices[i1];
                var v2 = group.Vertices[i2];
                
                var e1 = v1 - v0;
                var e2 = v2 - v0;
                var normal = Vector3.Normalize(Vector3.Cross(e1, e2));
                
                // Accumulate normals at vertices
                normals[i0] += normal;
                normals[i1] += normal;
                normals[i2] += normal;
            }
            
            // Normalize accumulated normals
            for (int i = 0; i < normals.Count; i++)
            {
                if (normals[i].Length() > 0.001f)
                    normals[i] = Vector3.Normalize(normals[i]);
                else
                    normals[i] = Vector3.UnitY; // Default up
            }
            
            return normals;
        }
    }
}
