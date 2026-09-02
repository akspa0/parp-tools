using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Numerics;

namespace WowViewer.Core.Geometry;

/// <summary>
/// Headless representation of a 3D polygonal mesh parsed from the OpenSCAD Object File Format (.off).
/// Contains interleaved vertex attributes (Position XYZ, Normal XYZ) and triangulated indices.
/// </summary>
public sealed class OffGeometry
{
    public float[] VertexData { get; }
    public uint[] Indices { get; }
    public Vector3 BoundsMin { get; }
    public Vector3 BoundsMax { get; }
    public int VertexCount { get; }
    public int FaceCount { get; }

    public OffGeometry(float[] vertexData, uint[] indices, Vector3 boundsMin, Vector3 boundsMax, int vertexCount, int faceCount)
    {
        VertexData = vertexData;
        Indices = indices;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        VertexCount = vertexCount;
        FaceCount = faceCount;
    }

    /// <summary>
    /// Parses an OpenSCAD OFF string.
    /// </summary>
    public static OffGeometry Parse(string offText)
    {
        using var reader = new StringReader(offText);
        return Parse(reader);
    }

    /// <summary>
    /// Parses an OpenSCAD OFF stream.
    /// </summary>
    public static OffGeometry Parse(TextReader reader)
    {
        string? line;
        bool headerFound = false;
        int vertexCount = 0;
        int faceCount = 0;

        // Read header
        while ((line = reader.ReadLine()) != null)
        {
            line = line.Trim();
            if (string.IsNullOrEmpty(line) || line.StartsWith('#')) continue;

            if (!headerFound)
            {
                if (line.StartsWith("OFF", StringComparison.OrdinalIgnoreCase))
                {
                    headerFound = true;
                    string remainder = line.Substring(3).Trim();
                    if (!string.IsNullOrEmpty(remainder))
                    {
                        var parts = remainder.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length >= 2)
                        {
                            vertexCount = int.Parse(parts[0], CultureInfo.InvariantCulture);
                            faceCount = int.Parse(parts[1], CultureInfo.InvariantCulture);
                            break;
                        }
                    }
                    continue;
                }
                else
                {
                    throw new FormatException("Invalid OFF file: missing 'OFF' signature header.");
                }
            }
            else
            {
                var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 2)
                {
                    vertexCount = int.Parse(parts[0], CultureInfo.InvariantCulture);
                    faceCount = int.Parse(parts[1], CultureInfo.InvariantCulture);
                    break;
                }
            }
        }

        if (!headerFound)
            throw new FormatException("Premature end of stream while reading OFF header.");

        var positions = new List<Vector3>(vertexCount);
        Vector3 min = new(float.MaxValue);
        Vector3 max = new(float.MinValue);

        // Read vertices
        while (positions.Count < vertexCount && (line = reader.ReadLine()) != null)
        {
            line = line.Trim();
            if (string.IsNullOrEmpty(line) || line.StartsWith('#')) continue;

            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length >= 3)
            {
                float x = float.Parse(parts[0], CultureInfo.InvariantCulture);
                float y = float.Parse(parts[1], CultureInfo.InvariantCulture);
                float z = float.Parse(parts[2], CultureInfo.InvariantCulture);
                Vector3 p = new(x, y, z);
                positions.Add(p);
                min = Vector3.Min(min, p);
                max = Vector3.Max(max, p);
            }
        }

        var indices = new List<uint>(faceCount * 3);
        int parsedFaces = 0;

        // Read faces (triangulate N-gons)
        while (parsedFaces < faceCount && (line = reader.ReadLine()) != null)
        {
            line = line.Trim();
            if (string.IsNullOrEmpty(line) || line.StartsWith('#')) continue;

            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 4) continue;

            int n = int.Parse(parts[0], CultureInfo.InvariantCulture);
            if (parts.Length < n + 1) continue;

            uint i0 = uint.Parse(parts[1], CultureInfo.InvariantCulture);
            for (int k = 1; k < n - 1; k++)
            {
                uint i1 = uint.Parse(parts[k + 1], CultureInfo.InvariantCulture);
                uint i2 = uint.Parse(parts[k + 2], CultureInfo.InvariantCulture);
                indices.Add(i0);
                indices.Add(i1);
                indices.Add(i2);
            }
            parsedFaces++;
        }

        // Calculate smooth normals
        var normals = new Vector3[positions.Count];
        for (int i = 0; i < indices.Count; i += 3)
        {
            uint i0 = indices[i];
            uint i1 = indices[i + 1];
            uint i2 = indices[i + 2];

            Vector3 v0 = positions[(int)i0];
            Vector3 v1 = positions[(int)i1];
            Vector3 v2 = positions[(int)i2];

            Vector3 normal = Vector3.Cross(v1 - v0, v2 - v0);
            if (normal.LengthSquared() > 1e-6f)
            {
                normal = Vector3.Normalize(normal);
                normals[i0] += normal;
                normals[i1] += normal;
                normals[i2] += normal;
            }
        }

        for (int i = 0; i < normals.Length; i++)
        {
            if (normals[i].LengthSquared() > 1e-6f)
                normals[i] = Vector3.Normalize(normals[i]);
            else
                normals[i] = Vector3.UnitZ;
        }

        // Interleave pos + normal: [px, py, pz, nx, ny, nz]
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

        return new OffGeometry(vertexData, indices.ToArray(), min, max, positions.Count, parsedFaces);
    }
}
