using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.Geometry;

namespace WoWViewer.Rendering;

/// <summary>
/// GPU mesh wrapper for procedural geometry, including OpenSCAD (.off and .stl) models
/// and built-in procedural primitives (rings, pins, arrows, reticles).
/// </summary>
public sealed class ProceduralMesh : IDisposable
{
    private readonly GL _gl;
    private uint _vao;
    private uint _vbo;
    private uint _ebo;
    private int _indexCount;
    private bool _disposed;

    public uint Vao => _vao;
    public int IndexCount => _indexCount;
    public Vector3 BoundsMin { get; }
    public Vector3 BoundsMax { get; }

    public ProceduralMesh(GL gl, float[] vertexData, uint[] indices, Vector3 boundsMin, Vector3 boundsMax)
    {
        _gl = gl;
        _indexCount = indices.Length;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;

        _vao = _gl.GenVertexArray();
        _vbo = _gl.GenBuffer();
        _ebo = _gl.GenBuffer();

        _gl.BindVertexArray(_vao);

        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _vbo);
        unsafe
        {
            fixed (float* ptr = vertexData)
            {
                _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(vertexData.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);
            }
        }

        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, _ebo);
        unsafe
        {
            fixed (uint* ptr = indices)
            {
                _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(uint)), ptr, BufferUsageARB.StaticDraw);
            }
        }

        // Layout: vec3 Position (location 0), vec3 Normal (location 1)
        uint stride = 6 * sizeof(float);
        _gl.EnableVertexAttribArray(0);
        unsafe
        {
            _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, (void*)0);
        }

        _gl.EnableVertexAttribArray(1);
        unsafe
        {
            _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, stride, (void*)(3 * sizeof(float)));
        }

        _gl.BindVertexArray(0);
    }

    public void Draw()
    {
        if (_disposed || _indexCount == 0) return;
        _gl.BindVertexArray(_vao);
        unsafe
        {
            _gl.DrawElements(PrimitiveType.Triangles, (uint)_indexCount, DrawElementsType.UnsignedInt, (void*)0);
        }
        _gl.BindVertexArray(0);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_ebo != 0) { _gl.DeleteBuffer(_ebo); _ebo = 0; }
        if (_vbo != 0) { _gl.DeleteBuffer(_vbo); _vbo = 0; }
        if (_vao != 0) { _gl.DeleteVertexArray(_vao); _vao = 0; }
    }
}

/// <summary>
/// Parser and generator for procedural 3D meshes (OpenSCAD .off / .stl and primitives).
/// </summary>
public static class ProceduralMeshLoader
{
    /// <summary>
    /// Parses an OpenSCAD Object File Format (.off) string into a GPU mesh.
    /// </summary>
    public static ProceduralMesh LoadFromOff(GL gl, string offText)
    {
        using var reader = new StringReader(offText);
        return LoadFromOff(gl, reader);
    }

    /// <summary>
    /// Parses an OpenSCAD Object File Format (.off) stream into a GPU mesh.
    /// </summary>
    public static ProceduralMesh LoadFromOff(GL gl, TextReader reader)
    {
        var geom = OffGeometry.Parse(reader);
        return new ProceduralMesh(gl, geom.VertexData, geom.Indices, geom.BoundsMin, geom.BoundsMax);
    }

    /// <summary>
    /// Generates a procedural 3D pointer arrow (head cone + stem cylinder).
    /// </summary>
    public static ProceduralMesh CreatePointerArrow(GL gl)
    {
        const string arrowScad = @"
            $fn = 16;
            union() {
                // Stem
                translate([0, 0, 0])
                cylinder(r1=0.08, r2=0.08, h=0.7, center=false);
                // Head
                translate([0, 0, 0.7])
                cylinder(r1=0.25, r2=0.0, h=0.5, center=false);
            }
        ";

        // Generate synthetic geometry directly:
        return BuildSyntheticPointerMesh(gl);
    }

    /// <summary>
    /// Generates a procedural orbital ring / pedestal (torus) for 3D selection.
    /// </summary>
    public static ProceduralMesh CreateOrbitalRing(GL gl, float radius = 1.0f, float tubeRadius = 0.04f, int radialSegments = 32, int tubeSegments = 8)
    {
        var positions = new List<Vector3>();
        var normals = new List<Vector3>();
        var indices = new List<uint>();

        for (int i = 0; i <= radialSegments; i++)
        {
            float u = (float)i / radialSegments * MathF.PI * 2f;
            float cosU = MathF.Cos(u);
            float sinU = MathF.Sin(u);

            Vector3 center = new(cosU * radius, 0f, sinU * radius);

            for (int j = 0; j <= tubeSegments; j++)
            {
                float v = (float)j / tubeSegments * MathF.PI * 2f;
                float cosV = MathF.Cos(v);
                float sinV = MathF.Sin(v);

                Vector3 normal = new(cosU * cosV, sinV, sinU * cosV);
                Vector3 pos = center + normal * tubeRadius;

                positions.Add(pos);
                normals.Add(normal);
            }
        }

        int stride = tubeSegments + 1;
        for (int i = 0; i < radialSegments; i++)
        {
            for (int j = 0; j < tubeSegments; j++)
            {
                uint i0 = (uint)(i * stride + j);
                uint i1 = (uint)((i + 1) * stride + j);
                uint i2 = (uint)((i + 1) * stride + (j + 1));
                uint i3 = (uint)(i * stride + (j + 1));

                indices.Add(i0); indices.Add(i1); indices.Add(i2);
                indices.Add(i0); indices.Add(i2); indices.Add(i3);
            }
        }

        Vector3 min = new(-radius - tubeRadius, -tubeRadius, -radius - tubeRadius);
        Vector3 max = new(radius + tubeRadius, tubeRadius, radius + tubeRadius);

        float[] vertexData = new float[positions.Count * 6];
        for (int i = 0; i < positions.Count; i++)
        {
            int baseIdx = i * 6;
            vertexData[baseIdx] = positions[i].X;
            vertexData[baseIdx + 1] = positions[i].Y;
            vertexData[baseIdx + 2] = positions[i].Z;
            vertexData[baseIdx + 3] = normals[i].X;
            vertexData[baseIdx + 4] = normals[i].Y;
            vertexData[baseIdx + 5] = normals[i].Z;
        }

        return new ProceduralMesh(gl, vertexData, indices.ToArray(), min, max);
    }

    /// <summary>
    /// Generates a procedural 3D target reticle ring.
    /// </summary>
    public static ProceduralMesh CreateTargetReticle(GL gl, float radius = 0.5f)
    {
        return CreateOrbitalRing(gl, radius, radius * 0.06f, 24, 6);
    }

    private static ProceduralMesh BuildSyntheticPointerMesh(GL gl)
    {
        // 3D Arrow pointing along +Z, tip at (0, 0, 0)
        var positions = new List<Vector3>
        {
            new(0f, 0f, 0f),                 // 0: Tip
            new(-0.25f, -0.25f, -0.6f),      // 1: Cone base left-down
            new(0.25f, -0.25f, -0.6f),       // 2: Cone base right-down
            new(0f, 0.35f, -0.6f),           // 3: Cone base top
            new(-0.08f, -0.08f, -0.6f),      // 4: Stem top left
            new(0.08f, -0.08f, -0.6f),       // 5: Stem top right
            new(0f, 0.12f, -0.6f),           // 6: Stem top back
            new(-0.08f, -0.08f, -1.2f),      // 7: Stem bot left
            new(0.08f, -0.08f, -1.2f),       // 8: Stem bot right
            new(0f, 0.12f, -1.2f)            // 9: Stem bot back
        };

        var indices = new List<uint>
        {
            // Cone sides
            0, 1, 2,
            0, 2, 3,
            0, 3, 1,
            // Cone base cap
            1, 3, 2,
            // Stem sides
            4, 7, 8,  4, 8, 5,
            5, 8, 9,  5, 9, 6,
            6, 9, 7,  6, 7, 4,
            // Stem bot cap
            7, 9, 8
        };

        Vector3 min = new(-0.25f, -0.25f, -1.2f);
        Vector3 max = new(0.25f, 0.35f, 0.0f);

        // Simple normals
        var normals = new Vector3[positions.Count];
        for (int i = 0; i < indices.Count; i += 3)
        {
            uint i0 = indices[i];
            uint i1 = indices[i + 1];
            uint i2 = indices[i + 2];
            Vector3 norm = Vector3.Normalize(Vector3.Cross(positions[(int)i1] - positions[(int)i0], positions[(int)i2] - positions[(int)i0]));
            normals[i0] += norm;
            normals[i1] += norm;
            normals[i2] += norm;
        }

        for (int i = 0; i < normals.Length; i++)
            normals[i] = normals[i].LengthSquared() > 0.001f ? Vector3.Normalize(normals[i]) : Vector3.UnitY;

        float[] vertexData = new float[positions.Count * 6];
        for (int i = 0; i < positions.Count; i++)
        {
            int baseIdx = i * 6;
            vertexData[baseIdx] = positions[i].X;
            vertexData[baseIdx + 1] = positions[i].Y;
            vertexData[baseIdx + 2] = positions[i].Z;
            vertexData[baseIdx + 3] = normals[i].X;
            vertexData[baseIdx + 4] = normals[i].Y;
            vertexData[baseIdx + 5] = normals[i].Z;
        }

        return new ProceduralMesh(gl, vertexData, indices.ToArray(), min, max);
    }
}
