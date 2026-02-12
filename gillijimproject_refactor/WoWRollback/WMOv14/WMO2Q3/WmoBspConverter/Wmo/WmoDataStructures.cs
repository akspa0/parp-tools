using System;
using System.Collections.Generic;
using System.Numerics;

namespace WmoBspConverter.Wmo
{
    /// <summary>
    /// Container for parsed WMO v14 data structures.
    /// </summary>
    public class WmoRootData
    {
        public string? FileName { get; set; }
        public int GroupCount { get; set; }
        public List<string> GroupNames { get; set; } = new List<string>();
        public List<WmoMaterial> Materials { get; set; } = new List<WmoMaterial>();
        public List<WmoGroupData> Groups { get; set; } = new List<WmoGroupData>();
        public WmoHeader Header { get; set; } = new WmoHeader();
        public List<WmoPortal> Portals { get; set; } = new List<WmoPortal>();
        public List<WmoLight> Lights { get; set; } = new List<WmoLight>();
    }

    /// <summary>
    /// WMO root file header data.
    /// </summary>
    public class WmoHeader
    {
        public uint MaterialCount { get; set; }
        public uint GroupCount { get; set; }
        public uint PortalCount { get; set; }
        public uint LightCount { get; set; }
        public uint DoodadCount { get; set; }
        public uint SetCount { get; set; }
        public uint AmbientColor { get; set; }
        public uint AreaTableId { get; set; }
        public Vector3 BoundingBoxMin { get; set; }
        public Vector3 BoundingBoxMax { get; set; }
        public ushort Flags { get; set; }
        public ushort LodCount { get; set; }
    }

    /// <summary>
    /// WMO material definition.
    /// </summary>
    public class WmoMaterial
    {
        public uint Flags { get; set; }
        public uint Shader { get; set; }
        public uint BlendMode { get; set; }
        public uint Texture1Offset { get; set; }
        public uint Texture2Offset { get; set; }
        public uint Texture3Offset { get; set; }
        public uint DiffuseColor { get; set; }
        public uint AmbientColor { get; set; }
        public uint SpecularColor { get; set; }
        public uint EmissiveColor { get; set; }
        public uint GroundType { get; set; }

        // Additional runtime data for v17+
        public uint Color2 { get; set; }
        public uint Flags2 { get; set; }
        public uint[] RuntimeData { get; set; } = new uint[4];
    }

    /// <summary>
    /// WMO group geometry data.
    /// </summary>
    public class WmoGroupData
    {
        public string? Name { get; set; }
        public uint Flags { get; set; }
        public Vector3 BoundingBoxMin { get; set; }
        public Vector3 BoundingBoxMax { get; set; }
        public ushort FirstPortalReferenceIndex { get; set; }
        public ushort PortalReferenceCount { get; set; }
        public ushort RenderBatchCountA { get; set; }
        public ushort RenderBatchCountInterior { get; set; }
        public ushort RenderBatchCountExterior { get; set; }
        public ushort Unknown { get; set; }
        public byte[] FogIndices { get; set; } = new byte[4];
        public uint LiquidType { get; set; }
        public uint GroupId { get; set; }
        public uint TerrainFlags { get; set; }
        public uint Unused { get; set; }

        // Geometry data
        public List<Vector3> Vertices { get; set; } = new List<Vector3>();
        public List<ushort> Indices { get; set; } = new List<ushort>();
        public List<Vector3> Normals { get; set; } = new List<Vector3>();
        public List<Vector2> TextureCoordinates { get; set; } = new List<Vector2>();
        public List<Vector2> LightmapCoordinates { get; set; } = new List<Vector2>();
        public List<uint> VertexColors { get; set; } = new List<uint>();
        public List<WmoFaceMaterial> FaceMaterials { get; set; } = new List<WmoFaceMaterial>();
        public List<WmoRenderBatch> RenderBatches { get; set; } = new List<WmoRenderBatch>();

        // Portal/visibility data
        public List<int> PortalReferences { get; set; } = new List<int>();
    }

    /// <summary>
    /// Face material assignment for triangles.
    /// </summary>
    public class WmoFaceMaterial
    {
        public byte Flags { get; set; }
        public byte MaterialId { get; set; }
    }

    /// <summary>
    /// Render batch information.
    /// </summary>
    public class WmoRenderBatch
    {
        public uint StartIndex { get; set; }
        public ushort Count { get; set; }
        public ushort MinVertexIndex { get; set; }
        public ushort MaxVertexIndex { get; set; }
        public byte Flags { get; set; }
        public byte MaterialId { get; set; }
    }

    /// <summary>
    /// Portal definition for visibility culling.
    /// </summary>
    public class WmoPortal
    {
        public Vector3[] Vertices { get; set; } = Array.Empty<Vector3>();
        public Vector4 Plane { get; set; }
        public uint FrontGroup { get; set; }
        public uint BackGroup { get; set; }
        public ushort[] PlaneIndices { get; set; } = Array.Empty<ushort>();
    }

    /// <summary>
    /// WMO lighting data.
    /// </summary>
    public class WmoLight
    {
        public Vector3 Position { get; set; }
        public Vector3 Color { get; set; }
        public float Intensity { get; set; }
        public float Radius { get; set; }
        public uint LightType { get; set; }
        public float AttenuationStart { get; set; }
        public float AttenuationEnd { get; set; }
    }

    /// <summary>
    /// WMO doodad/model placement.
    /// </summary>
    public class WmoDoodad
    {
        public uint NameIndex { get; set; }
        public Vector3 Position { get; set; }
        public Quaternion Rotation { get; set; }
        public float Scale { get; set; }
        public byte Flags { get; set; }
        public byte[] Color { get; set; } = new byte[4];
    }
}