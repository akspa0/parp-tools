// EXPERIMENTAL - Commented out since we're using our own .NET 9 implementation
// This file contained experimental LibBSP integration but we're using our own BspFile instead
// 
// using System;
// using System.Collections.Generic;
// using System.IO;
// using System.Linq;
// // using LibBSP; // Commented out - using our own .NET 9 implementation
// using System.Numerics;
//
// namespace WmoBspConverter.Bsp
// {
//     // EXPERIMENTAL - Commented out since we're using our own .NET 9 implementation
//     /// <summary>
//     /// BSP file implementation using LibBSP for proper Quake 3 format compliance.
//     /// NOTE: This was experimental - using our own BspFile implementation instead
//     /// </summary>
//     public class LibBspFile
//     {
//         private readonly BSP _bsp;
//         
//         public LibBspFile()
//         {
//             // _bsp = new BSP();
//             // _bsp.MapType = MapType.Quake3; // Set to Quake 3 format
//         }
//
//         public void AddVertex(Vector3 position)
//         {
//             var vertex = new Vertex
//             {
//                 position = new Vector3(position.X, position.Z, -position.Y), // Quake coordinate system
//                 normal = Vector3.Zero
//             };
//             
//             _bsp.Vertices = _bsp.Vertices ?? new Lump<Vertex>(null);
//             _bsp.Vertices.Add(vertex);
//         }
//
//         public void AddFace(int firstVertex, int numVertices, int textureIndex, int faceType = 0)
//         {
//             var face = new Face
//             {
//                 firstVertex = firstVertex,
//                 numVertices = numVertices,
//                 texture = textureIndex,
//                 type = faceType, // 0 = normal polygon
//                 // Quake 3 specific fields - using defaults
//                 lightmap = -1,
//                 lmIndex = -1,
//                 lmStart = new Vector2(0, 0),
//                 lmSize = new Vector2(0, 0),
//                 lmOrigin = Vector3.Zero,
//                 // Using identity vectors since we're not generating lightmaps
//                 lmVecs = new[,] { { new Vector3(1, 0, 0), new Vector3(0, 1, 0) } },
//                 normal = Vector3.UnitY,
//                 size = new Vector2(0, 0)
//             };
//
//             _bsp.Faces = _bsp.Faces ?? new Lump<Face>(null);
//             _bsp.Faces.Add(face);
//         }
//
//         public void Save(string filePath)
//         {
//             try
//             {
//                 _bsp.SaveToFile(filePath);
//             }
//             catch (Exception ex)
//             {
//                 throw new InvalidOperationException($"Failed to save BSP file: {ex.Message}", ex);
//             }
//         }
//     }
// }