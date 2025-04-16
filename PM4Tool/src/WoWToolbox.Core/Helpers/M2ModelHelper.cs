using System;
using System.Collections.Generic;
using System.Numerics;
using Warcraft.NET.Files.M2;
using Warcraft.NET.Files.M2.Chunks;
using Warcraft.NET.Files.M2.Entries;

namespace WoWToolbox.Core.Helpers
{
    /// <summary>
    /// Helper for loading and extracting mesh data from M2 models using Warcraft.NET.
    /// </summary>
    public static class M2ModelHelper
    {
        /// <summary>
        /// Represents a unified mesh extracted from an M2 model, ready for scene assembly.
        /// </summary>
        public class M2Mesh
        {
            public List<Vector3> Vertices = new();
            public List<Vector3> Normals = new();
            public List<Vector2> UVs = new();
            public List<ushort> Indices = new();
            public List<byte> MaterialIds = new(); // Optional, if available
        }

        /// <summary>
        /// Loads an M2 model from a file path and extracts mesh data.
        /// </summary>
        public static M2Mesh LoadMeshFromFile(string filePath, Vector3 position, Vector3 rotationDegrees, float scale)
        {
            var data = System.IO.File.ReadAllBytes(filePath);
            return LoadMeshFromBytes(data, position, rotationDegrees, scale);
        }

        /// <summary>
        /// Loads an M2 model from a byte array and extracts mesh data.
        /// </summary>
        public static M2Mesh LoadMeshFromBytes(byte[] data, Vector3 position, Vector3 rotationDegrees, float scale)
        {
            var model = new Model(data);
            var md21 = model.ModelInformation;
            if (md21 == null || md21.Vertices == null || md21.BoundingTriangles == null)
                throw new Exception("Invalid or incomplete M2 model data.");

            var mesh = new M2Mesh();
            var rot = Quaternion.CreateFromYawPitchRoll(
                MathF.PI / 180f * rotationDegrees.Y,
                MathF.PI / 180f * rotationDegrees.X,
                MathF.PI / 180f * rotationDegrees.Z);

            // Vertices, normals, UVs
            foreach (var v in md21.Vertices)
            {
                var transformed = Vector3.Transform(v.Position * scale, rot) + position;
                mesh.Vertices.Add(transformed);
                mesh.Normals.Add(Vector3.Normalize(Vector3.Transform(v.Normal, rot)));
                mesh.UVs.Add(new Vector2(v.TextureCoordX, v.TextureCoordY));
            }

            // Indices (triangles)
            foreach (var tri in md21.BoundingTriangles)
            {
                mesh.Indices.Add(tri.Index0);
                mesh.Indices.Add(tri.Index1);
                mesh.Indices.Add(tri.Index2);
            }

            // Material IDs (optional, not always available)
            // This can be expanded if batch/material info is needed

            return mesh;
        }
    }
} 