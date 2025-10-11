using System;
using System.IO;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using WoWToolbox.Core.Models;

namespace WoWToolbox.Core.WMO
{
    /// <summary>
    /// Writes WMO v17 (3.x/WotLK) root and group files, supporting all major features.
    /// </summary>
    public class WmoV17Writer
    {
        // --- Root file chunk writers ---
        public void WriteRoot(string filePath, WmoRootData rootData)
        {
            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            using var writer = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);
            WriteMVER(writer, 17); // v17 for 3.x/WotLK
            WriteMOHD(writer, rootData); // Placeholder for now
            WriteMOTX(writer, rootData.Textures); // Texture string table
            WriteMOMT(writer, rootData.Materials); // Material definitions
            WriteMOGN(writer, rootData.GroupNames); // Group name string table
            WriteMOGI(writer, rootData.GroupInfos); // Group info structs
            WriteMODD(writer, rootData.Doodads); // Doodad placements
            WriteMODN(writer, rootData.DoodadNames); // Doodad model names
            WriteMODS(writer, rootData.DoodadSets); // Doodad sets
            WriteMODR(writer, rootData.DoodadRefs); // Doodad references
            // TODO: Write any additional root chunks as needed
        }

        // --- Group file chunk writers ---
        public void WriteGroup(string filePath, WmoGroupData groupData)
        {
            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            using var writer = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);
            WriteMVER(writer, 17); // v17 for 3.x/WotLK
            WriteMOGP(writer, groupData); // Group header
            WriteMOPY(writer, groupData.Triangles); // Material/flag per triangle
            WriteMOVI(writer, groupData.Indices); // Indices
            WriteMOVT(writer, groupData.Vertices); // Vertex positions
            WriteMONR(writer, groupData.Normals); // Normals
            WriteMOTV(writer, groupData.UVs); // UVs
            WriteMOBA(writer, groupData.Batches); // Batches
            WriteMOLV(writer, groupData.LiquidVerts); // Liquid verts
            WriteMOIN(writer, groupData.LiquidIndices); // Liquid indices
            WriteMOBN(writer, groupData.Portals); // Portals
            WriteMOBR(writer, groupData.PortalRefs); // Portal refs
            WriteMOCV(writer, groupData.Colors); // Vertex colors
            WriteMOLM(writer, groupData.Lights); // Lights
            WriteMOLD(writer, groupData.DoodadRefs); // Doodad refs
            WriteMLIQ(writer, groupData.LiquidHeaders); // Liquid headers
            // TODO: Add any additional group chunks as needed
        }

        // --- Individual chunk writers (stubs) ---
        private void WriteMVER(BinaryWriter writer, int version)
        {
            // MVER chunk: 4 bytes header, 4 bytes size, 4 bytes version
            writer.Write(Encoding.ASCII.GetBytes("MVER"));
            writer.Write(4); // size
            writer.Write(version); // always 17 for v17
        }

        private void WriteMOHD(BinaryWriter writer, WmoRootData rootData)
        {
            // MOHD chunk: 4 bytes header, 4 bytes size, then struct (64 bytes in v17)
            writer.Write(Encoding.ASCII.GetBytes("MOHD"));
            writer.Write(64); // size (fixed for v17)
            var h = rootData.Header;
            writer.Write(h.MaterialCount);
            writer.Write(h.GroupCount);
            writer.Write(h.PortalCount);
            writer.Write(h.LightCount);
            writer.Write(h.ModelCount);
            writer.Write(h.DoodadCount);
            writer.Write(h.SetCount);
            writer.Write(h.AmbientColor);
            writer.Write(h.AreaTableID);
            writer.Write(h.BoundingBoxMin.X);
            writer.Write(h.BoundingBoxMin.Y);
            writer.Write(h.BoundingBoxMin.Z);
            writer.Write(h.BoundingBoxMax.X);
            writer.Write(h.BoundingBoxMax.Y);
            writer.Write(h.BoundingBoxMax.Z);
            writer.Write(h.Flags);
            writer.Write(h.LodCount);
        }

        private void WriteMOTX(BinaryWriter writer, List<string> textures)
        {
            // MOTX: null-terminated string table of texture paths
            using var ms = new MemoryStream();
            foreach (var tex in textures)
            {
                var bytes = Encoding.ASCII.GetBytes(tex);
                ms.Write(bytes, 0, bytes.Length);
                ms.WriteByte(0); // null terminator
            }
            var motxData = ms.ToArray();
            writer.Write(Encoding.ASCII.GetBytes("MOTX"));
            writer.Write(motxData.Length);
            writer.Write(motxData);
        }

        private void WriteMOMT(BinaryWriter writer, List<WmoMaterial> materials)
        {
            int structSize = 64;
            writer.Write(Encoding.ASCII.GetBytes("MOMT"));
            writer.Write(materials.Count * structSize);
            foreach (var mat in materials)
            {
                writer.Write(mat.Flags);
                writer.Write(mat.Shader);
                writer.Write(mat.BlendMode);
                writer.Write((uint)mat.Texture1Index);
                writer.Write(mat.Color1);
                writer.Write(mat.Color1b);
                writer.Write((uint)mat.Texture2Index);
                writer.Write(mat.Color2);
                writer.Write(mat.GroupType);
                writer.Write((uint)mat.Texture3Index);
                writer.Write(mat.Color3);
                writer.Write(mat.Flags3);
                for (int i = 0; i < 4; i++)
                    writer.Write(mat.RuntimeData != null && mat.RuntimeData.Length > i ? mat.RuntimeData[i] : 0u);
            }
        }

        private void WriteMOGN(BinaryWriter writer, List<string> groupNames)
        {
            // MOGN: null-terminated string table of group names
            using var ms = new MemoryStream();
            foreach (var name in groupNames)
            {
                var bytes = Encoding.ASCII.GetBytes(name);
                ms.Write(bytes, 0, bytes.Length);
                ms.WriteByte(0); // null terminator
            }
            var mognData = ms.ToArray();
            writer.Write(Encoding.ASCII.GetBytes("MOGN"));
            writer.Write(mognData.Length);
            writer.Write(mognData);
        }

        private void WriteMOGI(BinaryWriter writer, List<WmoGroupInfo> groupInfos)
        {
            int structSize = 32;
            writer.Write(Encoding.ASCII.GetBytes("MOGI"));
            writer.Write(groupInfos.Count * structSize);
            foreach (var info in groupInfos)
            {
                writer.Write(info.Flags);
                writer.Write(info.BoundingBoxMin.X);
                writer.Write(info.BoundingBoxMin.Y);
                writer.Write(info.BoundingBoxMin.Z);
                writer.Write(info.BoundingBoxMax.X);
                writer.Write(info.BoundingBoxMax.Y);
                writer.Write(info.BoundingBoxMax.Z);
                writer.Write(info.NameIndex);
            }
        }

        private void WriteMODD(BinaryWriter writer, List<WmoDoodad> doodads)
        {
            // MODD: array of doodad placement structs (40 bytes each in v17)
            int structSize = 40;
            writer.Write(Encoding.ASCII.GetBytes("MODD"));
            writer.Write(doodads.Count * structSize);
            foreach (var doodad in doodads)
            {
                writer.Write(new byte[structSize]); // TODO: Serialize actual doodad placement fields from doodad
            }
        }

        private void WriteMODN(BinaryWriter writer, List<string> doodadNames)
        {
            // MODN: null-terminated string table of doodad model names
            using var ms = new MemoryStream();
            foreach (var name in doodadNames)
            {
                var bytes = Encoding.ASCII.GetBytes(name);
                ms.Write(bytes, 0, bytes.Length);
                ms.WriteByte(0); // null terminator
            }
            var modnData = ms.ToArray();
            writer.Write(Encoding.ASCII.GetBytes("MODN"));
            writer.Write(modnData.Length);
            writer.Write(modnData);
        }

        private void WriteMODS(BinaryWriter writer, List<WmoDoodadSet> doodadSets)
        {
            // MODS: array of doodad set structs (32 bytes each in v17)
            int structSize = 32;
            writer.Write(Encoding.ASCII.GetBytes("MODS"));
            writer.Write(doodadSets.Count * structSize);
            foreach (var set in doodadSets)
            {
                writer.Write(new byte[structSize]); // TODO: Serialize actual doodad set fields from set
            }
        }

        private void WriteMODR(BinaryWriter writer, List<int> doodadRefs)
        {
            // MODR: array of int32 doodad references
            writer.Write(Encoding.ASCII.GetBytes("MODR"));
            writer.Write(doodadRefs.Count * 4);
            foreach (var i in doodadRefs)
            {
                writer.Write(i);
            }
        }

        private void WriteMOGP(BinaryWriter writer, WmoGroupData groupData)
        {
            var h = groupData.Header;
            writer.Write(Encoding.ASCII.GetBytes("MOGP"));
            writer.Write(68); // size
            writer.Write(h.GroupNameOffset);
            writer.Write(h.DescriptiveGroupNameOffset);
            writer.Write(h.Flags);
            writer.Write(h.BoundingBoxMin.X);
            writer.Write(h.BoundingBoxMin.Y);
            writer.Write(h.BoundingBoxMin.Z);
            writer.Write(h.BoundingBoxMax.X);
            writer.Write(h.BoundingBoxMax.Y);
            writer.Write(h.BoundingBoxMax.Z);
            writer.Write(h.FirstPortalReferenceIndex);
            writer.Write(h.PortalReferenceCount);
            writer.Write(h.RenderBatchCountA);
            writer.Write(h.RenderBatchCountInterior);
            writer.Write(h.RenderBatchCountExterior);
            writer.Write(h.Unknown);
            for (int i = 0; i < 4; i++) writer.Write(h.Unknown2[i]);
            writer.Write(h.LiquidType);
            writer.Write(h.GroupID);
            writer.Write(h.TerrainFlags);
            writer.Write(h.Unused);
        }

        private void WriteMOPY(BinaryWriter writer, List<WmoTriangle> triangles)
        {
            // MOPY: array of (byte flag, byte matId) per triangle
            writer.Write(Encoding.ASCII.GetBytes("MOPY"));
            writer.Write(triangles.Count * 2);
            foreach (var tri in triangles)
            {
                writer.Write(tri.Flags);
                writer.Write(tri.MaterialId);
            }
        }

        private void WriteMOVI(BinaryWriter writer, List<ushort> indices)
        {
            // MOVI: array of uint16 indices (3 per triangle)
            writer.Write(Encoding.ASCII.GetBytes("MOVI"));
            writer.Write(indices.Count * 2);
            foreach (var idx in indices)
            {
                writer.Write(idx);
            }
        }

        private void WriteMOVT(BinaryWriter writer, List<WmoVertex> vertices)
        {
            // MOVT: array of 3 floats per vertex
            writer.Write(Encoding.ASCII.GetBytes("MOVT"));
            writer.Write(vertices.Count * 12);
            foreach (var v in vertices)
            {
                writer.Write(v.Position.X);
                writer.Write(v.Position.Y);
                writer.Write(v.Position.Z);
            }
        }

        private void WriteMONR(BinaryWriter writer, List<System.Numerics.Vector3> normals)
        {
            // MONR: array of 3 floats per vertex (normals)
            writer.Write(Encoding.ASCII.GetBytes("MONR"));
            writer.Write(normals.Count * 12);
            foreach (var n in normals)
            {
                writer.Write(n.X);
                writer.Write(n.Y);
                writer.Write(n.Z);
            }
        }

        private void WriteMOTV(BinaryWriter writer, List<System.Numerics.Vector2> uvs)
        {
            // MOTV: array of 2 floats per vertex (UVs)
            writer.Write(Encoding.ASCII.GetBytes("MOTV"));
            writer.Write(uvs.Count * 8);
            foreach (var uv in uvs)
            {
                writer.Write(uv.X);
                writer.Write(uv.Y);
            }
        }

        private void WriteMOBA(BinaryWriter writer, List<WmoBatch> batches)
        {
            int structSize = 24;
            writer.Write(Encoding.ASCII.GetBytes("MOBA"));
            writer.Write(batches.Count * structSize);
            foreach (var batch in batches)
            {
                // Write 12 bytes of zero for UnknownBoxData (not present in WmoBatch struct)
                writer.Write(new byte[12]);
                writer.Write((uint)batch.StartIndex);
                writer.Write((ushort)batch.IndexCount);
                writer.Write((ushort)batch.MinVertex);
                writer.Write((ushort)batch.MaxVertex);
                writer.Write((byte)0); // Flags (not present in WmoBatch struct)
                writer.Write(batch.MaterialId);
            }
        }

        private void WriteMOLV(BinaryWriter writer, List<System.Numerics.Vector3> liquidVerts)
        {
            // MOLV: array of 3 floats per liquid vertex
            writer.Write(Encoding.ASCII.GetBytes("MOLV"));
            writer.Write(liquidVerts.Count * 12);
            foreach (var v in liquidVerts)
            {
                writer.Write(v.X);
                writer.Write(v.Y);
                writer.Write(v.Z);
            }
        }

        private void WriteMOIN(BinaryWriter writer, List<ushort> liquidIndices)
        {
            // MOIN: array of uint16 indices for liquid
            writer.Write(Encoding.ASCII.GetBytes("MOIN"));
            writer.Write(liquidIndices.Count * 2);
            foreach (var idx in liquidIndices)
            {
                writer.Write(idx);
            }
        }

        private void WriteMOBN(BinaryWriter writer, List<WmoPortal> portals)
        {
            // MOBN: array of portal structs (16 bytes each)
            int structSize = 16;
            writer.Write(Encoding.ASCII.GetBytes("MOBN"));
            writer.Write(portals.Count * structSize);
            foreach (var portal in portals)
            {
                writer.Write(new byte[structSize]); // TODO: Serialize actual portal fields from portal
            }
        }

        private void WriteMOBR(BinaryWriter writer, List<ushort> portalRefs)
        {
            // MOBR: array of uint16 portal refs
            writer.Write(Encoding.ASCII.GetBytes("MOBR"));
            writer.Write(portalRefs.Count * 2);
            foreach (var pr in portalRefs)
            {
                writer.Write(pr);
            }
        }

        private void WriteMOCV(BinaryWriter writer, List<System.Numerics.Vector3> colors)
        {
            // MOCV: array of 3 floats per vertex (color)
            writer.Write(Encoding.ASCII.GetBytes("MOCV"));
            writer.Write(colors.Count * 12);
            foreach (var c in colors)
            {
                writer.Write(c.X);
                writer.Write(c.Y);
                writer.Write(c.Z);
            }
        }

        private void WriteMOLM(BinaryWriter writer, List<WmoLight> lights)
        {
            // MOLM: array of light structs (8 bytes each)
            int structSize = 8;
            writer.Write(Encoding.ASCII.GetBytes("MOLM"));
            writer.Write(lights.Count * structSize);
            foreach (var light in lights)
            {
                writer.Write(new byte[structSize]); // TODO: Serialize actual light fields from light
            }
        }

        private void WriteMOLD(BinaryWriter writer, List<uint> doodadRefs)
        {
            // MOLD: array of uint32 doodad refs
            writer.Write(Encoding.ASCII.GetBytes("MOLD"));
            writer.Write(doodadRefs.Count * 4);
            foreach (var dr in doodadRefs)
            {
                writer.Write(dr);
            }
        }

        private void WriteMLIQ(BinaryWriter writer, List<byte[]> liquidHeaders)
        {
            // MLIQ: array of 48-byte liquid headers (raw for now)
            int structSize = 48;
            writer.Write(Encoding.ASCII.GetBytes("MLIQ"));
            writer.Write(liquidHeaders.Count * structSize);
            foreach (var lh in liquidHeaders)
            {
                writer.Write(lh, 0, structSize);
            }
        }
        // ... add other root chunk writers as needed ...
    }
    // Placeholder data structures for compilation (to be replaced with real ones)
    public class WmoRootData {
        public WmoRootHeader Header { get; set; } = new();
        public List<string> Textures { get; set; } = new();
        public List<WmoMaterial> Materials { get; set; } = new();
        public List<string> GroupNames { get; set; } = new();
        public List<WmoGroupInfo> GroupInfos { get; set; } = new();
        public List<WmoDoodad> Doodads { get; set; } = new();
        public List<string> DoodadNames { get; set; } = new();
        public List<WmoDoodadSet> DoodadSets { get; set; } = new();
        public List<int> DoodadRefs { get; set; } = new();
    }
    public class WmoRootHeader {
        public uint MaterialCount { get; set; }
        public uint GroupCount { get; set; }
        public uint PortalCount { get; set; }
        public uint LightCount { get; set; }
        public uint ModelCount { get; set; }
        public uint DoodadCount { get; set; }
        public uint SetCount { get; set; }
        public uint AmbientColor { get; set; }
        public uint AreaTableID { get; set; }
        public Vector3 BoundingBoxMin { get; set; }
        public Vector3 BoundingBoxMax { get; set; }
        public ushort Flags { get; set; }
        public ushort LodCount { get; set; }
    }
    public class WmoGroupData {
        public WmoGroupHeader Header { get; set; } = new();
        public List<WmoTriangle> Triangles { get; set; } = new();
        public List<ushort> Indices { get; set; } = new();
        public List<WmoVertex> Vertices { get; set; } = new();
        public List<System.Numerics.Vector3> Normals { get; set; } = new();
        public List<System.Numerics.Vector2> UVs { get; set; } = new();
        public List<WmoBatch> Batches { get; set; } = new();
        public List<System.Numerics.Vector3> LiquidVerts { get; set; } = new();
        public List<ushort> LiquidIndices { get; set; } = new();
        public List<WmoPortal> Portals { get; set; } = new();
        public List<ushort> PortalRefs { get; set; } = new();
        public List<System.Numerics.Vector3> Colors { get; set; } = new();
        public List<WmoLight> Lights { get; set; } = new();
        public List<uint> DoodadRefs { get; set; } = new();
        public List<byte[]> LiquidHeaders { get; set; } = new();
    }
    public class WmoGroupHeader {
        public uint GroupNameOffset { get; set; }
        public uint DescriptiveGroupNameOffset { get; set; }
        public uint Flags { get; set; }
        public Vector3 BoundingBoxMin { get; set; }
        public Vector3 BoundingBoxMax { get; set; }
        public ushort FirstPortalReferenceIndex { get; set; }
        public ushort PortalReferenceCount { get; set; }
        public ushort RenderBatchCountA { get; set; }
        public ushort RenderBatchCountInterior { get; set; }
        public ushort RenderBatchCountExterior { get; set; }
        public ushort Unknown { get; set; }
        public uint[] Unknown2 { get; set; } = new uint[4];
        public uint LiquidType { get; set; }
        public uint GroupID { get; set; }
        public uint TerrainFlags { get; set; }
        public uint Unused { get; set; }
    }
    public struct WmoDoodad { }
    public struct WmoDoodadSet { }
    public struct WmoPortal { }
    public struct WmoLight { }
} 