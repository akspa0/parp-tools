using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;

namespace WmoBspConverter.Quake3
{
    /// <summary>
    /// Simplified Quake 3 BSP writer based on LibBSP format knowledge.
    /// Targets Q3 BSP version 46 (IBSP format).
    /// </summary>
    public class Q3BspWriter
    {
        private const int IBSP_MAGIC = 0x50534249; // "IBSP"
        private const int Q3_VERSION = 46;
        private const int NUM_LUMPS = 17;
        
        private readonly Q3Bsp _bsp;
        
        public Q3BspWriter(Q3Bsp bsp)
        {
            _bsp = bsp;
        }
        
        public void Write(string path)
        {
            using var stream = File.Create(path);
            using var writer = new BinaryWriter(stream);
            
            // Write header placeholder
            var headerStart = stream.Position;
            writer.Write(IBSP_MAGIC);
            writer.Write(Q3_VERSION);
            
            // Reserve space for lump directory (17 lumps × 8 bytes each)
            var lumpDirStart = stream.Position;
            for (int i = 0; i < NUM_LUMPS * 2; i++)
            {
                writer.Write(0); // offset and length placeholders
            }
            
            // Write lumps and track their info
            var lumpInfos = new List<(int offset, int length)>();
            
            lumpInfos.Add(WriteLump(writer, WriteLump_Entities));      // 0: Entities
            lumpInfos.Add(WriteLump(writer, WriteLump_Textures));      // 1: Textures
            lumpInfos.Add(WriteLump(writer, WriteLump_Planes));        // 2: Planes
            lumpInfos.Add(WriteLump(writer, WriteLump_Nodes));         // 3: Nodes
            lumpInfos.Add(WriteLump(writer, WriteLump_Leaves));        // 4: Leaves
            lumpInfos.Add(WriteLump(writer, WriteLump_LeafFaces));     // 5: LeafFaces
            lumpInfos.Add(WriteLump(writer, WriteLump_LeafBrushes));   // 6: LeafBrushes
            lumpInfos.Add(WriteLump(writer, WriteLump_Models));        // 7: Models
            lumpInfos.Add(WriteLump(writer, WriteLump_Brushes));       // 8: Brushes
            lumpInfos.Add(WriteLump(writer, WriteLump_BrushSides));    // 9: BrushSides
            lumpInfos.Add(WriteLump(writer, WriteLump_Vertices));      // 10: Vertices
            lumpInfos.Add(WriteLump(writer, WriteLump_MeshVerts));     // 11: MeshVerts
            lumpInfos.Add(WriteLump(writer, WriteLump_Effects));       // 12: Effects
            lumpInfos.Add(WriteLump(writer, WriteLump_Faces));         // 13: Faces
            lumpInfos.Add(WriteLump(writer, WriteLump_Lightmaps));     // 14: Lightmaps
            lumpInfos.Add(WriteLump(writer, WriteLump_LightVols));     // 15: LightVols
            lumpInfos.Add(WriteLump(writer, WriteLump_VisData));       // 16: VisData
            
            // Write lump directory
            stream.Position = lumpDirStart;
            foreach (var (offset, length) in lumpInfos)
            {
                writer.Write(offset);
                writer.Write(length);
            }
            
            Console.WriteLine($"[Q3BSP] Wrote BSP: {lumpInfos.Count} lumps, {stream.Length} bytes");
        }
        
        private (int offset, int length) WriteLump(BinaryWriter writer, Action<BinaryWriter> writeLumpData)
        {
            var start = (int)writer.BaseStream.Position;
            writeLumpData(writer);
            var end = (int)writer.BaseStream.Position;
            return (start, end - start);
        }
        
        private void WriteLump_Entities(BinaryWriter writer)
        {
            var entities = _bsp.Entities ?? "{\n\"classname\" \"worldspawn\"\n}\n";
            if (!entities.EndsWith("\0"))
                entities += "\0";
            writer.Write(Encoding.ASCII.GetBytes(entities));
        }
        
        private void WriteLump_Textures(BinaryWriter writer)
        {
            // Q3 texture: 64 bytes name + 4 bytes flags + 4 bytes contents = 72 bytes
            foreach (var tex in _bsp.Textures)
            {
                var nameBytes = new byte[64];
                var texBytes = Encoding.ASCII.GetBytes(tex.Name);
                Array.Copy(texBytes, nameBytes, Math.Min(texBytes.Length, 63));
                writer.Write(nameBytes);
                writer.Write(tex.Flags);
                writer.Write(tex.Contents);
            }
        }
        
        private void WriteLump_Planes(BinaryWriter writer)
        {
            // Q3 plane: 16 bytes (Vector3 normal + float distance)
            foreach (var plane in _bsp.Planes)
            {
                writer.Write(plane.Normal.X);
                writer.Write(plane.Normal.Y);
                writer.Write(plane.Normal.Z);
                writer.Write(plane.Distance);
            }
        }
        
        private void WriteLump_Nodes(BinaryWriter writer)
        {
            // Q3 node: 36 bytes
            foreach (var node in _bsp.Nodes)
            {
                writer.Write(node.PlaneIndex);
                writer.Write(node.Children[0]);
                writer.Write(node.Children[1]);
                writer.Write(node.Mins[0]);
                writer.Write(node.Mins[1]);
                writer.Write(node.Mins[2]);
                writer.Write(node.Maxs[0]);
                writer.Write(node.Maxs[1]);
                writer.Write(node.Maxs[2]);
            }
        }
        
        private void WriteLump_Leaves(BinaryWriter writer)
        {
            // Q3 leaf: 48 bytes
            foreach (var leaf in _bsp.Leaves)
            {
                writer.Write(leaf.Cluster);
                writer.Write(leaf.Area);
                writer.Write(leaf.Mins[0]);
                writer.Write(leaf.Mins[1]);
                writer.Write(leaf.Mins[2]);
                writer.Write(leaf.Maxs[0]);
                writer.Write(leaf.Maxs[1]);
                writer.Write(leaf.Maxs[2]);
                writer.Write(leaf.FirstLeafFace);
                writer.Write(leaf.NumLeafFaces);
                writer.Write(leaf.FirstLeafBrush);
                writer.Write(leaf.NumLeafBrushes);
            }
        }
        
        private void WriteLump_LeafFaces(BinaryWriter writer)
        {
            foreach (var index in _bsp.LeafFaces)
            {
                writer.Write(index);
            }
        }
        
        private void WriteLump_LeafBrushes(BinaryWriter writer)
        {
            foreach (var index in _bsp.LeafBrushes)
            {
                writer.Write(index);
            }
        }
        
        private void WriteLump_Models(BinaryWriter writer)
        {
            // Q3 model: 40 bytes
            foreach (var model in _bsp.Models)
            {
                writer.Write(model.Mins.X);
                writer.Write(model.Mins.Y);
                writer.Write(model.Mins.Z);
                writer.Write(model.Maxs.X);
                writer.Write(model.Maxs.Y);
                writer.Write(model.Maxs.Z);
                writer.Write(model.FirstFace);
                writer.Write(model.NumFaces);
                writer.Write(model.FirstBrush);
                writer.Write(model.NumBrushes);
            }
        }
        
        private void WriteLump_Brushes(BinaryWriter writer)
        {
            // Q3 brush: 12 bytes
            foreach (var brush in _bsp.Brushes)
            {
                writer.Write(brush.FirstSide);
                writer.Write(brush.NumSides);
                writer.Write(brush.TextureIndex);
            }
        }
        
        private void WriteLump_BrushSides(BinaryWriter writer)
        {
            // Q3 brushside: 8 bytes
            foreach (var side in _bsp.BrushSides)
            {
                writer.Write(side.PlaneIndex);
                writer.Write(side.TextureIndex);
            }
        }
        
        private void WriteLump_Vertices(BinaryWriter writer)
        {
            // Q3 vertex: 44 bytes
            foreach (var vert in _bsp.Vertices)
            {
                writer.Write(vert.Position.X);
                writer.Write(vert.Position.Y);
                writer.Write(vert.Position.Z);
                writer.Write(vert.TexCoord.X);
                writer.Write(vert.TexCoord.Y);
                writer.Write(vert.LightmapCoord.X);
                writer.Write(vert.LightmapCoord.Y);
                writer.Write(vert.Normal.X);
                writer.Write(vert.Normal.Y);
                writer.Write(vert.Normal.Z);
                writer.Write(vert.Color);
            }
        }
        
        private void WriteLump_MeshVerts(BinaryWriter writer)
        {
            foreach (var index in _bsp.MeshVerts)
            {
                writer.Write(index);
            }
        }
        
        private void WriteLump_Effects(BinaryWriter writer)
        {
            // Q3 effect: 72 bytes
            foreach (var effect in _bsp.Effects)
            {
                var nameBytes = new byte[64];
                var effectBytes = Encoding.ASCII.GetBytes(effect.Name);
                Array.Copy(effectBytes, nameBytes, Math.Min(effectBytes.Length, 63));
                writer.Write(nameBytes);
                writer.Write(effect.Brush);
                writer.Write(effect.Unknown);
            }
        }
        
        private void WriteLump_Faces(BinaryWriter writer)
        {
            // Q3 face: 104 bytes
            foreach (var face in _bsp.Faces)
            {
                writer.Write(face.TextureIndex);
                writer.Write(face.Effect);
                writer.Write(face.Type);
                writer.Write(face.FirstVertex);
                writer.Write(face.NumVertices);
                writer.Write(face.FirstMeshVert);
                writer.Write(face.NumMeshVerts);
                writer.Write(face.LightmapIndex);
                writer.Write(face.LightmapStart[0]);
                writer.Write(face.LightmapStart[1]);
                writer.Write(face.LightmapSize[0]);
                writer.Write(face.LightmapSize[1]);
                writer.Write(face.LightmapOrigin.X);
                writer.Write(face.LightmapOrigin.Y);
                writer.Write(face.LightmapOrigin.Z);
                writer.Write(face.LightmapVecs[0].X);
                writer.Write(face.LightmapVecs[0].Y);
                writer.Write(face.LightmapVecs[0].Z);
                writer.Write(face.LightmapVecs[1].X);
                writer.Write(face.LightmapVecs[1].Y);
                writer.Write(face.LightmapVecs[1].Z);
                writer.Write(face.Normal.X);
                writer.Write(face.Normal.Y);
                writer.Write(face.Normal.Z);
                writer.Write(face.PatchSize[0]);
                writer.Write(face.PatchSize[1]);
            }
        }
        
        private void WriteLump_Lightmaps(BinaryWriter writer)
        {
            // Q3 lightmap: 128×128×3 bytes = 49152 bytes each
            foreach (var lightmap in _bsp.Lightmaps)
            {
                writer.Write(lightmap);
            }
        }
        
        private void WriteLump_LightVols(BinaryWriter writer)
        {
            // Q3 lightvol: 8 bytes
            foreach (var vol in _bsp.LightVols)
            {
                writer.Write(vol.Ambient);
                writer.Write(vol.Directional);
                writer.Write(vol.Dir);
            }
        }
        
        private void WriteLump_VisData(BinaryWriter writer)
        {
            if (_bsp.VisData != null && _bsp.VisData.Length > 0)
            {
                writer.Write(_bsp.VisData);
            }
        }
    }
}
