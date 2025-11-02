using System;
using System.Collections.Generic;
using System.Numerics;

namespace WmoBspConverter.Quake3
{
    /// <summary>
    /// Quake 3 BSP data structure (version 46).
    /// Simplified modern C# implementation based on LibBSP and Q3 spec.
    /// </summary>
    public class Q3Bsp
    {
        public string Entities { get; set; } = string.Empty;
        public List<Q3Texture> Textures { get; set; } = new();
        public List<Q3Plane> Planes { get; set; } = new();
        public List<Q3Node> Nodes { get; set; } = new();
        public List<Q3Leaf> Leaves { get; set; } = new();
        public List<int> LeafFaces { get; set; } = new();
        public List<int> LeafBrushes { get; set; } = new();
        public List<Q3Model> Models { get; set; } = new();
        public List<Q3Brush> Brushes { get; set; } = new();
        public List<Q3BrushSide> BrushSides { get; set; } = new();
        public List<Q3Vertex> Vertices { get; set; } = new();
        public List<int> MeshVerts { get; set; } = new();
        public List<Q3Effect> Effects { get; set; } = new();
        public List<Q3Face> Faces { get; set; } = new();
        public List<byte[]> Lightmaps { get; set; } = new();
        public List<Q3LightVol> LightVols { get; set; } = new();
        public byte[]? VisData { get; set; }
    }

    public class Q3Texture
    {
        public string Name { get; set; } = string.Empty;
        public int Flags { get; set; }
        public int Contents { get; set; }
    }

    public class Q3Plane
    {
        public Vector3 Normal { get; set; }
        public float Distance { get; set; }
    }

    public class Q3Node
    {
        public int PlaneIndex { get; set; }
        public int[] Children { get; set; } = new int[2];
        public int[] Mins { get; set; } = new int[3];
        public int[] Maxs { get; set; } = new int[3];
    }

    public class Q3Leaf
    {
        public int Cluster { get; set; }
        public int Area { get; set; }
        public int[] Mins { get; set; } = new int[3];
        public int[] Maxs { get; set; } = new int[3];
        public int FirstLeafFace { get; set; }
        public int NumLeafFaces { get; set; }
        public int FirstLeafBrush { get; set; }
        public int NumLeafBrushes { get; set; }
    }

    public class Q3Model
    {
        public Vector3 Mins { get; set; }
        public Vector3 Maxs { get; set; }
        public int FirstFace { get; set; }
        public int NumFaces { get; set; }
        public int FirstBrush { get; set; }
        public int NumBrushes { get; set; }
    }

    public class Q3Brush
    {
        public int FirstSide { get; set; }
        public int NumSides { get; set; }
        public int TextureIndex { get; set; }
    }

    public class Q3BrushSide
    {
        public int PlaneIndex { get; set; }
        public int TextureIndex { get; set; }
    }

    public class Q3Vertex
    {
        public Vector3 Position { get; set; }
        public Vector2 TexCoord { get; set; }
        public Vector2 LightmapCoord { get; set; }
        public Vector3 Normal { get; set; }
        public uint Color { get; set; } = 0xFFFFFFFF; // RGBA
    }

    public class Q3Effect
    {
        public string Name { get; set; } = string.Empty;
        public int Brush { get; set; }
        public int Unknown { get; set; }
    }

    public class Q3Face
    {
        public int TextureIndex { get; set; }
        public int Effect { get; set; } = -1;
        public int Type { get; set; } = 1; // 1=polygon, 2=patch, 3=mesh, 4=billboard
        public int FirstVertex { get; set; }
        public int NumVertices { get; set; }
        public int FirstMeshVert { get; set; }
        public int NumMeshVerts { get; set; }
        public int LightmapIndex { get; set; } = -1;
        public int[] LightmapStart { get; set; } = new int[2];
        public int[] LightmapSize { get; set; } = new int[2];
        public Vector3 LightmapOrigin { get; set; }
        public Vector3[] LightmapVecs { get; set; } = new Vector3[2];
        public Vector3 Normal { get; set; }
        public int[] PatchSize { get; set; } = new int[2];
    }

    public class Q3LightVol
    {
        public byte[] Ambient { get; set; } = new byte[3];
        public byte[] Directional { get; set; } = new byte[3];
        public byte[] Dir { get; set; } = new byte[2];
    }
}
