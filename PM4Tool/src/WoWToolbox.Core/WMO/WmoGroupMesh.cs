using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using System.Linq;
using System.Runtime.InteropServices;
using System.Globalization;
using WoWToolbox.Core.Models; // Added for MeshData

namespace WoWToolbox.Core.WMO
{
    // Represents a single vertex in a WMO group mesh
    public struct WmoVertex
    {
        public Vector3 Position; // (X, Y, Z) after conversion from (X, Z, -Y)
        public Vector3 Normal;
        public Vector2 UV;
    }

    // Represents a triangle in the mesh
    public struct WmoTriangle
    {
        public int Index0, Index1, Index2;
        public byte MaterialId;
        public ushort Flags;
    }

    // Represents a render batch (from MOBA)
    public struct WmoBatch
    {
        public int StartIndex;
        public int IndexCount;
        public int MinVertex;
        public int MaxVertex;
        public byte MaterialId;
    }

    // Main mesh container for a WMO group
    public class WmoGroupMesh
    {
        public List<WmoVertex> Vertices { get; set; } = new();
        public List<WmoTriangle> Triangles { get; set; } = new();
        public List<WmoBatch> Batches { get; set; } = new();

        // --- Internal lists used during loading ---
        // Kept separate to potentially preserve Normal/UV if needed elsewhere later
        private List<Vector3> _loadedPositions = new();
        private List<Vector3> _loadedNormals = new();
        private List<Vector2> _loadedUVs = new();
        private List<(int i0, int i1, int i2)> _loadedIndices = new(); // Store indices directly

        public static void ListChunks(string filePath)
        {
            using var stream = File.OpenRead(filePath);
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
            Console.WriteLine($"[ChunkLister V2] Listing chunks in: {filePath}");
            long fileLen = stream.Length;
            Console.WriteLine($"[ChunkLister V2] File Length: {fileLen} (0x{fileLen:X}) bytes");
            while (stream.Position + 8 <= fileLen) // Need 8 bytes for ID + Size
            {
                long chunkStartOffset = stream.Position;
                var chunkIdBytes = reader.ReadBytes(4);
                if (chunkIdBytes.Length < 4) { Console.WriteLine("[ChunkLister V2] Failed to read chunk ID bytes."); break; }

                uint chunkSize = reader.ReadUInt32();
                long chunkDataStartOffset = stream.Position;
                long expectedChunkEndOffset = chunkDataStartOffset + chunkSize;

                // Reverse ID for display string
                var chunkIdStr = new string(chunkIdBytes.Reverse().Select(b => (char)b).ToArray());

                Console.WriteLine($"  Offset 0x{chunkStartOffset:X6}: Chunk '{chunkIdStr}' (ID Raw: 0x{BitConverter.ToUInt32(chunkIdBytes, 0):X8}), Size: {chunkSize} (0x{chunkSize:X}) bytes. Data starts at 0x{chunkDataStartOffset:X6}, Expected end 0x{expectedChunkEndOffset:X6}");

                // Check if size is plausible
                if (expectedChunkEndOffset > fileLen)
                {
                    Console.WriteLine($"    [WARN] Chunk size exceeds file length! Aborting list.");
                    break;
                }

                // Peek ahead: What's immediately after the header?
                if (chunkSize > 0 && chunkDataStartOffset < fileLen)
                {
                    long peekPosition = chunkDataStartOffset;
                    int bytesToPeek = (int)Math.Min(8, fileLen - peekPosition); // Peek up to 8 bytes
                    if (bytesToPeek > 0)
                    {
                         // Temporarily read ahead without permanently moving position
                        byte[] peekBytes = reader.ReadBytes(bytesToPeek);
                        stream.Position = peekPosition; // Reset position back immediately
                        string peekHex = BitConverter.ToString(peekBytes).Replace("-", " ");
                        Console.WriteLine($"    Peek at 0x{peekPosition:X6}: {peekHex}");
                    }
                }

                // Move to the start of the next chunk header
                stream.Position = expectedChunkEndOffset;
            }
            Console.WriteLine($"[ChunkLister V2] Reached end of listing at offset 0x{stream.Position:X6}");
        }

        public static WmoGroupMesh LoadFromStream(Stream stream, string? debugFilePath = null)
        {
            if (debugFilePath != null)
                Console.WriteLine($"[WMOGroup] Loading file: {debugFilePath}");
            var mesh = new WmoGroupMesh();
            var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

            // --- 3.3.5 WMO group file parsing ---
            // Step 1: Find and skip MVER (version) and locate MOGP (group header)
            stream.Position = 0;
            long fileLen = stream.Length;
            bool foundMOGP = false;
            long mogpStart = 0;
            uint mogpSize = 0;
            while (stream.Position + 8 <= fileLen)
            {
                long chunkStart = stream.Position;
                var chunkIdBytes = reader.ReadBytes(4);
                if (chunkIdBytes.Length < 4) break;
                var chunkIdStr = new string(chunkIdBytes.Reverse().Select(b => (char)b).ToArray());
                var chunkSize = reader.ReadUInt32();
                long chunkEnd = stream.Position + chunkSize;
                if (stream.Position + chunkSize > fileLen)
                {
                    Console.WriteLine($"[ERR] Chunk '{chunkIdStr}' at 0x{chunkStart:X} claims size {chunkSize}, but only {fileLen - stream.Position} bytes remain. Aborting.");
                    break;
                }
                if (chunkIdStr == "MOGP")
                {
                    foundMOGP = true;
                    mogpStart = chunkStart;
                    mogpSize = chunkSize;
                    break;
                }
                stream.Position = chunkEnd;
            }
            if (!foundMOGP)
            {
                Console.WriteLine("[ERR] No MOGP chunk found in group file. Aborting.");
                return mesh;
            }

            // Step 2: Read MOGP struct and handle potential size issues
            long mogpHeaderStart = mogpStart + 8;
            stream.Position = mogpHeaderStart;
            MOGP mogp; // Use the struct defined in WmoChunks.cs
            long structEndPos = -1;
            long reportedEndPos = mogpHeaderStart + mogpSize;
            long nextChunkPos = reportedEndPos; // Default position after MOGP chunk according to its header

            try
            {
                int mogpStructSize = Marshal.SizeOf<MOGP>();
                if (mogpSize < mogpStructSize)
                {
                    Console.WriteLine($"[ERR] MOGP reported chunk size {mogpSize} is smaller than expected struct size {mogpStructSize}. Aborting.");
                    return mesh;
                }

                // Read only the expected struct size first
                var mogpData = reader.ReadBytes(mogpStructSize);
                structEndPos = stream.Position; // Position immediately after reading the struct data

                // Pin the byte array and get a pointer to deserialize the struct
                GCHandle handle = GCHandle.Alloc(mogpData, GCHandleType.Pinned);
                try
                {
                    IntPtr ptr = handle.AddrOfPinnedObject();
                    mogp = Marshal.PtrToStructure<MOGP>(ptr);
                }
                finally
                {
                    handle.Free();
                }

                // Check for size discrepancy
                if (structEndPos < reportedEndPos)
                {
                    Console.WriteLine($"[WARN] MOGP reported size ({mogpSize}) is larger than standard struct size ({mogpStructSize}). Reported end 0x{reportedEndPos:X}, struct end 0x{structEndPos:X}. Assuming subsequent chunks might follow immediately after struct data.");
                    nextChunkPos = structEndPos; // Start looking for MOPY right after the struct data
                }
                else
                {
                     // Reported size matches struct size (or is smaller, handled above), or reading struct reached reported end.
                     // Use the end position calculated from the header size.
                     Console.WriteLine($"[DEBUG] MOGP reported size ({mogpSize}) matches struct size ({mogpStructSize}) or reading struct reached reported end (0x{reportedEndPos:X}). Looking for next chunks after reported end.");
                     nextChunkPos = reportedEndPos;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERR] Failed to read or parse MOGP struct: {ex.Message}. Aborting.");
                return mesh;
            }

            // Step 3: Read subsequent chunks from the determined position
            stream.Position = nextChunkPos; // Position after MOGP chunk data (or struct data if size was inflated)

            // Declare geometry lists
            var mopy = new List<MOPY>();
            var indices = new List<ushort>();
            var positions = new List<Vector3>();
            var normals = new List<Vector3>();
            var uvs = new List<Vector2>();
            var moba = new List<MOBA>(); // Using MOBA struct from WmoChunks.cs

            var expectedChunks = new[] { "MOPY", "MOVI", "MOVT", "MONR", "MOTV", "MOBA" };

            foreach (var expectedChunkId in expectedChunks)
            {
                // Check if enough data remains for a chunk header (name + size)
                if (stream.Position + 8 > fileLen)
                {
                    Console.WriteLine($"[DEBUG] Reached end of file (or < 8 bytes remaining) while expecting chunk '{expectedChunkId}'. Stopping chunk reading.");
                    break; // Stop reading chunks if not enough space for a header
                }

                long chunkHeaderPos = stream.Position;
                var actualChunkIdBytes = reader.ReadBytes(4);
                if (actualChunkIdBytes.Length < 4) break; // Should not happen due to check above, but safety first
                var actualChunkIdStr = new string(actualChunkIdBytes.Reverse().Select(b => (char)b).ToArray());
                var chunkSize = reader.ReadUInt32();
                long chunkDataPos = stream.Position;
                long chunkEndPos = chunkDataPos + chunkSize;

                // Validate chunk size against remaining file length
                if (chunkEndPos > fileLen)
                {
                    Console.WriteLine($"[ERR] Chunk '{actualChunkIdStr}' at 0x{chunkHeaderPos:X} claims size {chunkSize}, which exceeds file length {fileLen}. Stopping chunk reading.");
                    break; // Stop reading if a chunk claims to go past EOF
                }

                // Check if the actual chunk matches the expected one
                if (actualChunkIdStr == expectedChunkId)
                {
                    Console.WriteLine($"[DEBUG] Reading expected chunk '{actualChunkIdStr}' Size: {chunkSize}");

                    try
                    {
                        // Read chunk data based on ID
                        switch (actualChunkIdStr)
                        {
                            case "MOPY":
                                int mopyCount = (int)(chunkSize / Marshal.SizeOf<MOPY>());
                                mopy = MOPY.ReadArray(reader, mopyCount);
                                break;
                            case "MOVI":
                                int indexCount = (int)(chunkSize / sizeof(ushort));
                                indices = MOVI.ReadArray(reader, indexCount);
                                break;
                            case "MOVT":
                                int vertexCount = (int)(chunkSize / (sizeof(float) * 3));
                                positions = MOVT.ReadArray(reader, vertexCount);
                                break;
                            case "MONR":
                                int normalCount = (int)(chunkSize / (sizeof(float) * 3));
                                normals = MONR.ReadArray(reader, normalCount);
                                break;
                            case "MOTV":
                                int uvCount = (int)(chunkSize / (sizeof(float) * 2));
                                uvs = MOTV.ReadArray(reader, uvCount);
                                break;
                            case "MOBA":
                                int batchCount = (int)(chunkSize / Marshal.SizeOf<MOBA>());
                                moba = MOBA.ReadArray(reader, batchCount);
                                break;
                        }
                    }
                    catch (Exception ex)
                    {
                         Console.WriteLine($"[ERR] Failed to read data for chunk '{actualChunkIdStr}': {ex.Message}. Stopping chunk reading.");
                         break; // Abort on read error
                    }

                    // Ensure stream is positioned correctly after reading chunk data
                    if (stream.Position != chunkEndPos)
                    {
                        Console.WriteLine($"[WARN] Stream position 0x{stream.Position:X} does not match expected end 0x{chunkEndPos:X} after reading chunk '{actualChunkIdStr}'. Adjusting position.");
                        stream.Position = chunkEndPos;
                    }
                }
                else
                {
                    // Unexpected chunk found where the expected one should be
                    Console.WriteLine($"[WARN] Expected chunk '{expectedChunkId}' at 0x{chunkHeaderPos:X}, but found '{actualChunkIdStr}'. Skipping '{actualChunkIdStr}'.");
                    // Skip over the unexpected chunk's data
                    stream.Position = chunkEndPos;
                    // Since we skipped the unexpected chunk, we need to reconsider the current 'expectedChunkId' in the next iteration.
                    // However, the current loop structure iterates through expected chunks sequentially.
                    // This means if MOPY is missing, but MOVI is present, this logic won't find MOVI because it's still expecting MOPY.
                    // A more flexible approach might be needed if chunks can be missing *and* followed by later chunks.
                    // For now, this handles the case where the file might end, or contain unexpected data *instead* of the expected chunk.
                    // If the goal is to allow *missing* chunks, we might need to loop through the file reading any chunk header
                    // and process it if it's one of the expected ones, storing data until all expected are found or EOF.

                    // For now, let's just break if the order is wrong, as per the strict requirement, but after logging the skip.
                    // If the file simply ends after MOGP, the initial EOF check handles it.
                    // If there's *different* data where MOPY should be, this warning and break is appropriate.
                    Console.WriteLine("[ERR] Strict chunk order violated. Aborting group parse.");
                    return mesh; // Re-instating strict abort based on updated understanding of requirements
                    // To allow skipping, remove the return above and the loop would continue, expecting the *next* chunk in the sequence.
                }
            }

            // --- Mesh Construction ---
            // Combine vertices, normals, UVs
            int numVertices = positions.Count;
            if (normals.Count != numVertices || uvs.Count != numVertices)
            {
                Console.WriteLine($"[WARN] Vertex ({positions.Count}), Normal ({normals.Count}), or UV ({uvs.Count}) counts mismatch. Using minimum count: {numVertices = Math.Min(positions.Count, Math.Min(normals.Count, uvs.Count))}");
            }

            for (int i = 0; i < numVertices; i++)
            {
                mesh.Vertices.Add(new WmoVertex
                {
                    Position = positions[i],
                    Normal = (i < normals.Count) ? normals[i] : Vector3.UnitY, // Default normal if missing
                    UV = (i < uvs.Count) ? uvs[i] : Vector2.Zero // Default UV if missing
                });
            }

            // Create triangles using indices and MOPY data
            if (indices.Count % 3 != 0)
            {
                Console.WriteLine($"[WARN] Index count ({indices.Count}) is not divisible by 3. Truncating extra indices.");
            }
            int numTriangles = indices.Count / 3;
            if (mopy.Count != numTriangles)
            {
                 Console.WriteLine($"[WARN] MOPY count ({mopy.Count}) does not match triangle count ({numTriangles}). Using minimum count: {numTriangles = Math.Min(mopy.Count, numTriangles)} ");
            }

            for (int i = 0; i < numTriangles; i++)
            {
                int i0 = indices[i * 3 + 0];
                int i1 = indices[i * 3 + 1];
                int i2 = indices[i * 3 + 2];

                // Basic validation against vertex count
                if (i0 >= numVertices || i1 >= numVertices || i2 >= numVertices)
                {
                    Console.WriteLine($"[WARN] Triangle {i} uses out-of-bounds vertex index ({i0}, {i1}, {i2} vs {numVertices}). Skipping triangle.");
                    continue;
                }

                mesh.Triangles.Add(new WmoTriangle
                {
                    Index0 = i0,
                    Index1 = i1,
                    Index2 = i2,
                    Flags = mopy[i].Flags,
                    MaterialId = mopy[i].MaterialId
                });
            }

             // Convert MOBA to WmoBatch (adjust based on actual MOBA structure)
            foreach (var batch in moba)
            {
                mesh.Batches.Add(new WmoBatch
                {
                    StartIndex = (int)batch.StartIndex,
                    IndexCount = batch.Count,
                    MinVertex = batch.MinVertexIndex,
                    MaxVertex = batch.MaxVertexIndex,
                    MaterialId = batch.MaterialId
                });
            }

            if (debugFilePath != null)
                Console.WriteLine($"[OK] Loaded group: {debugFilePath} (Vertices: {mesh.Vertices.Count}, Tris: {mesh.Triangles.Count})");
            return mesh;
        }

        // ADDED: Helper method to save mesh to OBJ format
        public static void SaveToObj(WmoGroupMesh mesh, string filePath)
        {
            Console.WriteLine($"[DEBUG][SaveToObj] Attempting to save OBJ to: {filePath}");
            try
            {
                Console.WriteLine($"[DEBUG][SaveToObj] Inside try block for {filePath}");
                // Ensure the directory exists
                string? directoryPath = Path.GetDirectoryName(filePath);
                if (!string.IsNullOrEmpty(directoryPath) && !Directory.Exists(directoryPath))
                {
                    Console.WriteLine($"[DEBUG][SaveToObj] Creating directory: {directoryPath}");
                    Directory.CreateDirectory(directoryPath);
                }
                Console.WriteLine($"[DEBUG][SaveToObj] About to open StreamWriter for {filePath}");
                using var writer = new StreamWriter(filePath);
                Console.WriteLine($"[DEBUG][SaveToObj] StreamWriter opened for {filePath}");
                writer.WriteLine($"# WoWToolbox WMO Group Mesh Export");
                writer.WriteLine($"# Exported: {DateTime.Now}");
                writer.WriteLine($"# Vertices: {mesh.Vertices.Count}");
                writer.WriteLine($"# Triangles: {mesh.Triangles.Count}");
                writer.WriteLine($"# Batches: {mesh.Batches?.Count ?? 0}");

                if (mesh.Vertices.Count == 0)
                {
                    writer.WriteLine("# WARNING: No vertices found in mesh.");
                    return;
                }

                writer.WriteLine("o WmoGroupMesh");

                // Write Vertices (v x y z)
                foreach (var vertex in mesh.Vertices)
                {
                    // Convert from WoW (X, Y, Z) to OBJ (X, Z, -Y) so +Z is up
                    writer.WriteLine(string.Format(CultureInfo.InvariantCulture, "v {0:F6} {1:F6} {2:F6}", vertex.Position.X, vertex.Position.Z, -vertex.Position.Y));
                }

                // Write Normals (vn x y z)
                // Check if normals are actually present (not default)
                bool hasNormals = mesh.Vertices.Any(v => v.Normal != Vector3.UnitY && v.Normal != Vector3.Zero);
                if (hasNormals)
                {
                    foreach (var vertex in mesh.Vertices)
                    {
                        // Convert normal from (X, Y, Z) to (X, Z, -Y)
                        writer.WriteLine(string.Format(CultureInfo.InvariantCulture, "vn {0:F6} {1:F6} {2:F6}", vertex.Normal.X, vertex.Normal.Z, -vertex.Normal.Y));
                    }
                }

                // Write UVs (vt u v)
                // Check if UVs are actually present (not default)
                bool hasUVs = mesh.Vertices.Any(v => v.UV != Vector2.Zero);
                if (hasUVs)
                {
                    foreach (var vertex in mesh.Vertices)
                    {
                         // OBJ UVs often have V inverted (1-v)
                        writer.WriteLine(string.Format(CultureInfo.InvariantCulture, "vt {0:F6} {1:F6}", vertex.UV.X, 1.0f - vertex.UV.Y));
                    }
                }

                if (mesh.Triangles.Count == 0)
                {
                    writer.WriteLine("# WARNING: No triangles found in mesh.");
                    return;
                }

                // Write Faces (f v/vt/vn)
                // Group faces by Material ID for potential use in viewers
                var groupedTriangles = mesh.Triangles.GroupBy(t => t.MaterialId).OrderBy(g => g.Key);
                foreach (var group in groupedTriangles)
                {
                    writer.WriteLine($"g Material_{group.Key}");
                    writer.WriteLine($"usemtl Material_{group.Key}"); // Placeholder material usage
                    foreach (var tri in group)
                    {
                        // OBJ uses 1-based indices
                        int v1 = tri.Index0 + 1;
                        int v2 = tri.Index1 + 1;
                        int v3 = tri.Index2 + 1;

                        // Format face string based on available data
                        string fStr;
                        if (hasUVs && hasNormals)
                        {
                            fStr = $"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}";
                        }
                        else if (hasUVs)
                        {
                            fStr = $"f {v1}/{v1} {v2}/{v2} {v3}/{v3}";
                        }
                        else if (hasNormals)
                        {
                            fStr = $"f {v1}//{v1} {v2}//{v2} {v3}//{v3}";
                        }
                        else
                        {
                            fStr = $"f {v1} {v2} {v3}";
                        }
                        writer.WriteLine(fStr);
                    }
                }
                Console.WriteLine($"[DEBUG][SaveToObj] Finished writing content for {filePath}");
                writer.Flush();
                Console.WriteLine($"[DEBUG][SaveToObj] StreamWriter flushed for {filePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERR][SaveToObj] Exception caught while saving OBJ ({filePath}): {ex.ToString()}");
            }
            finally
            {
                Console.WriteLine($"[DEBUG][SaveToObj] Reached finally block for {filePath}. Checking file existence...");
                if (File.Exists(filePath))
                {
                    Console.WriteLine($"[DEBUG][SaveToObj] File DOES exist after save attempt: {filePath}");
                }
                else
                {
                    Console.WriteLine($"[WARN][SaveToObj] File DOES NOT exist after save attempt: {filePath}");
                }
            }
        }

        public static WmoGroupMesh MergeMeshes(IEnumerable<WmoGroupMesh> meshes)
        {
            var merged = new WmoGroupMesh();
            int vertexOffset = 0;
            foreach (var mesh in meshes)
            {
                // Copy vertices
                merged.Vertices.AddRange(mesh.Vertices);
                // Copy triangles with index offset
                foreach (var tri in mesh.Triangles)
                {
                    merged.Triangles.Add(new WmoTriangle
                    {
                        Index0 = tri.Index0 + vertexOffset,
                        Index1 = tri.Index1 + vertexOffset,
                        Index2 = tri.Index2 + vertexOffset,
                        MaterialId = tri.MaterialId,
                        Flags = tri.Flags
                    });
                }
                // Optionally, copy batches (not merged)
                if (mesh.Batches != null)
                    merged.Batches.AddRange(mesh.Batches);
                vertexOffset += mesh.Vertices.Count;
            }
            return merged;
        }
    }
} 