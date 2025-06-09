using System;
using System.IO;
using Warcraft.NET.Files;
using Warcraft.NET.Files.Interfaces;
using Warcraft.NET.Attribute;
using WoWToolbox.Core.v2.Models.PM4.Chunks;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using WoWToolbox.Core.v2.Foundation.Data;

namespace WoWToolbox.Core.v2.Foundation.Data
{
    /// <summary>
    /// Optimized PM4 file parser maintaining full compatibility with Warcraft.NET.
    /// Uses Warcraft.NET's reflection-based chunk loading for seamless integration.
    /// </summary>
    public class PM4File : ChunkedFile, IBinarySerializable
    {
        #region Direct Properties - Warcraft.NET uses reflection to populate these
        
        /// <summary>Gets or sets the MVER chunk (Version)</summary>
        [ChunkOptional]
        public MVER? MVER { get; set; }

        /// <summary>Gets or sets the MSHD chunk (Header)</summary>
        [ChunkOptional]
        public MSHDChunk? MSHD { get; set; }

        /// <summary>Gets or sets the MSLK chunk (Scene Graph Links)</summary>
        [ChunkOptional]
        public MSLKChunk? MSLK { get; set; }

        /// <summary>Gets or sets the MSPI chunk (Indices into MSPV)</summary>
        [ChunkOptional]
        public MSPIChunk? MSPI { get; set; }

        /// <summary>Gets or sets the MSPV chunk (Path Vertices)</summary>
        [ChunkOptional]
        public MSPVChunk? MSPV { get; set; }

        /// <summary>Gets or sets the MSVT chunk (Render Vertices)</summary>
        public MSVTChunk? MSVT { get; set; }

        /// <summary>Gets or sets the MSVI chunk (Vertex Indices)</summary>
        public MSVIChunk? MSVI { get; set; }

        /// <summary>Gets or sets the MSUR chunk (Surface Definitions)</summary>
        public MSURChunk? MSUR { get; set; }

        /// <summary>Gets or sets the MSCN chunk (Exterior Vertices)</summary>
        public MSCNChunk? MSCN { get; set; }

        /// <summary>Gets or sets the MSRN chunk (Mesh Surface Referenced Normals)</summary>
        [ChunkOptional]
        public MSRNChunk? MSRN { get; set; }

        /// <summary>Gets or sets the MPRL chunk (Position Data)</summary>
        public MPRLChunk? MPRL { get; set; }

        /// <summary>Gets or sets the MPRR chunk (Reference Data)</summary>
        public MPRRChunk? MPRR { get; set; }

        /// <summary>Gets or sets the MDBH chunk (Destructible Building Header)</summary>
        [ChunkOptional]
        public MDBHChunk? MDBH { get; set; }

        /// <summary>Gets or sets the MDOS chunk (Object Data)</summary>
        [ChunkOptional]
        public MDOSChunk? MDOS { get; set; }

        /// <summary>Gets or sets the MDSF chunk (Structure Data)</summary>
        public MDSFChunk? MDSF { get; set; }

        #endregion

        /// <summary>
        /// Initializes a new instance of the PM4File class.
        /// </summary>
        public PM4File() { }

        /// <summary>
        /// Initializes a new instance of the PM4File class from binary data.
        /// Warcraft.NET automatically populates chunk properties via reflection.
        /// </summary>
        /// <param name="inData">The binary data to parse</param>
        public PM4File(byte[] inData) : base(inData)
        {
            // Base constructor (ChunkedFile) automatically loads chunks
            // into corresponding properties using reflection based on signatures
        }

        /// <summary>
        /// Loads a PM4 file from the specified path with optimized performance.
        /// </summary>
        /// <param name="path">Path to the PM4 file</param>
        /// <returns>PM4File instance with auto-loaded chunks</returns>
        public static PM4File FromFile(string path)
        {
            return new PM4File(File.ReadAllBytes(path));
        }

        /// <summary>
        /// Creates a PM4File from a stream for memory-efficient loading.
        /// </summary>
        /// <param name="stream">Stream containing PM4 data</param>
        /// <returns>PM4File instance</returns>
        public static PM4File FromStream(Stream stream)
        {
            using var memoryStream = new MemoryStream();
            stream.CopyTo(memoryStream);
            return new PM4File(memoryStream.ToArray());
        }

        /// <summary>
        /// Checks chunk availability for analysis and validation.
        /// </summary>
        /// <returns>Chunk availability information</returns>
        public PM4ChunkAvailability GetChunkAvailability()
        {
            return new PM4ChunkAvailability
            {
                HasMSLK = MSLK != null,
                HasMSVT = MSVT != null,
                HasMSUR = MSUR != null,
                HasMDSF = MDSF != null,
                HasMDOS = MDOS != null,
                HasMPRL = MPRL != null,
                HasMPRR = MPRR != null
            };
        }

        /// <inheritdoc/>
        public new byte[] Serialize(long offset = 0)
        {
            return base.Serialize(offset);
        }

        /// <inheritdoc/>
        public override bool IsReverseSignature()
        {
            return true;
        }

        /// <summary>
        /// Returns all triangles as a list of (int, int, int) tuples from MSUR/MSVI.
        /// </summary>
        public List<(int, int, int)> GetAllTriangles()
        {
            var triangles = new List<(int, int, int)>();
            if (MSUR == null || MSVI == null || MSVT == null)
                return triangles;
            var indices = MSVI.Indices;
            foreach (var surface in MSUR.Entries)
            {
                if (!surface.HasValidGeometry) continue;
                int start = (int)surface.MsviFirstIndex;
                int count = (int)surface.IndexCount;
                for (int i = 0; i + 2 < count; i += 3)
                {
                    int a = (int)indices[start + i];
                    int b = (int)indices[start + i + 1];
                    int c = (int)indices[start + i + 2];
                    triangles.Add((a, b, c));
                }
            }
            return triangles;
        }

        /// <summary>
        /// Extracts individual buildings using enhanced root node detection and dual geometry assembly.
        /// Uses multiple strategies for universal PM4 file compatibility.
        /// </summary>
        public List<CompleteWMOModel> ExtractBuildings()
        {
            var buildings = new List<CompleteWMOModel>();
            if (MSLK == null || MSPI == null || MSPV == null || MSUR == null || MSVI == null || MSVT == null)
                return buildings;

            // Strategy 1: Self-referencing root nodes (primary method)
            var rootNodes = MSLK.Entries
                .Select((entry, idx) => (entry, idx))
                .Where(x => x.entry.Unknown_0x04 == (uint)x.idx)
                .ToList();

            bool hasValidRoots = false;
            foreach (var root in rootNodes)
            {
                var model = ExtractBuildingFromRoot(root.entry, root.idx);
                if (model != null && model.Vertices.Count > 0)
                {
                    buildings.Add(model);
                    hasValidRoots = true;
                }
            }

            // Strategy 2: Fallback - Group by Unknown_0x04 if no valid roots found
            if (!hasValidRoots)
            {
                var groupedEntries = MSLK.Entries
                    .Select((entry, idx) => (entry, idx))
                    .Where(x => x.entry.HasGeometry)
                    .GroupBy(x => x.entry.Unknown_0x04)
                    .Where(g => g.Count() > 0);

                foreach (var group in groupedEntries)
                {
                    var model = ExtractBuildingFromGroup(group.ToList());
                    if (model != null && model.Vertices.Count > 0)
                    {
                        buildings.Add(model);
                    }
                }
            }

            return buildings;
        }

        /// <summary>
        /// Extracts a building from a specific root node and its children.
        /// </summary>
        private CompleteWMOModel? ExtractBuildingFromRoot(MSLKEntry rootEntry, int rootIdx)
        {
            var model = new CompleteWMOModel { FileName = $"Building_{rootIdx}" };

            // Find all entries that belong to this root
            var structuralEntries = MSLK.Entries
                .Where(e => (e.Unknown_0x04 == rootIdx || e.Unknown_0x04 == rootEntry.Unknown_0x04) && e.HasGeometry)
                .ToList();

            // Phase 1: Add structural geometry from MSLK/MSPI/MSPV
            AddStructuralGeometry(model, structuralEntries);

            // Phase 2: Add render surfaces from MSUR/MSVI/MSVT
            AddRenderSurfaces(model);

            return model.Vertices.Count > 0 ? model : null;
        }

        /// <summary>
        /// Extracts a building from a group of related entries.
        /// </summary>
        private CompleteWMOModel? ExtractBuildingFromGroup(List<(MSLKEntry entry, int idx)> groupEntries)
        {
            var model = new CompleteWMOModel { FileName = $"Building_Group_{groupEntries.First().entry.Unknown_0x04}" };

            // Convert to MSLK entries
            var structuralEntries = groupEntries.Select(x => x.entry).ToList();

            // Add structural and render geometry
            AddStructuralGeometry(model, structuralEntries);
            AddRenderSurfaces(model);

            return model.Vertices.Count > 0 ? model : null;
        }

        /// <summary>
        /// Adds structural geometry from MSLK/MSPI/MSPV chains.
        /// </summary>
        private void AddStructuralGeometry(CompleteWMOModel model, List<MSLKEntry> entries)
        {
            var vertexMap = new Dictionary<int, int>();

            foreach (var entry in entries)
            {
                int mspiStart = entry.MspiFirstIndex;
                int mspiCount = entry.MspiIndexCount;

                for (int i = 0; i < mspiCount; i++)
                {
                    int mspiIdx = mspiStart + i;
                    if (mspiIdx < 0 || mspiIdx >= MSPI.Indices.Count) continue;

                    uint pvIdx = MSPI.Indices[mspiIdx];
                    if (!vertexMap.ContainsKey((int)pvIdx))
                    {
                        vertexMap[(int)pvIdx] = model.Vertices.Count;
                        if (pvIdx < MSPV.Vertices.Count)
                        {
                            var v = MSPV.Vertices[(int)pvIdx];
                            model.Vertices.Add(new Vector3(v.X, v.Y, v.Z));
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Adds render surfaces from MSUR/MSVI/MSVT with proper spatial filtering.
        /// Only adds surfaces that are spatially near the building's structural elements.
        /// This prevents the bug where ALL surfaces were added to every building causing massive duplicate files.
        /// </summary>
        private void AddRenderSurfaces(CompleteWMOModel model)
        {
            if (MSUR == null || MSVI == null || MSVT == null || model.Vertices.Count == 0)
                return;

            // Calculate bounding box of existing structural vertices
            var bounds = CalculateBuildingBounds(model);
            if (!bounds.HasValue)
                return;

            // Find MSUR surfaces that are spatially near this building
            var nearbySurfaces = FindMSURSurfacesNearBounds(bounds.Value, 50.0f); // 50 unit tolerance

            // Add vertices from nearby MSUR surfaces
            var msvtIndexToLocal = new Dictionary<uint, int>();
            var structuralVertexOffset = model.Vertices.Count;

            // First pass: Add all MSVT vertices referenced by nearby surfaces
            foreach (int surfaceIndex in nearbySurfaces)
            {
                if (surfaceIndex >= MSUR.Entries.Count) continue;
                var surface = MSUR.Entries[surfaceIndex];

                for (int i = 0; i < surface.IndexCount && surface.MsviFirstIndex + i < MSVI.Indices.Count; i++)
                {
                    uint msvtIndex = MSVI.Indices[(int)surface.MsviFirstIndex + i];
                    if (msvtIndex < MSVT.Vertices.Count && !msvtIndexToLocal.ContainsKey(msvtIndex))
                    {
                        msvtIndexToLocal[msvtIndex] = model.Vertices.Count;
                        var vertex = MSVT.Vertices[(int)msvtIndex];
                        // Use proper coordinate transformation
                        var worldCoords = new Vector3(vertex.Y, vertex.X, vertex.Z);
                        model.Vertices.Add(worldCoords);
                    }
                }
            }

            // Second pass: Add triangle faces from nearby surfaces
            foreach (int surfaceIndex in nearbySurfaces)
            {
                if (surfaceIndex >= MSUR.Entries.Count) continue;
                var surface = MSUR.Entries[surfaceIndex];

                if (surface.IndexCount < 3) continue;

                // Generate triangle fan from surface
                for (int i = 0; i < surface.IndexCount - 2; i += 3)
                {
                    if (surface.MsviFirstIndex + i + 2 < MSVI.Indices.Count)
                    {
                        uint v1Index = MSVI.Indices[(int)surface.MsviFirstIndex + i];
                        uint v2Index = MSVI.Indices[(int)surface.MsviFirstIndex + i + 1];
                        uint v3Index = MSVI.Indices[(int)surface.MsviFirstIndex + i + 2];

                        if (msvtIndexToLocal.ContainsKey(v1Index) && 
                            msvtIndexToLocal.ContainsKey(v2Index) && 
                            msvtIndexToLocal.ContainsKey(v3Index))
                        {
                            model.TriangleIndices.Add(msvtIndexToLocal[v1Index]);
                            model.TriangleIndices.Add(msvtIndexToLocal[v2Index]);
                            model.TriangleIndices.Add(msvtIndexToLocal[v3Index]);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Calculates the bounding box of vertices in a building model.
        /// </summary>
        private (Vector3 min, Vector3 max)? CalculateBuildingBounds(CompleteWMOModel model)
        {
            if (model.Vertices.Count == 0)
                return null;

            var minX = model.Vertices.Min(v => v.X);
            var minY = model.Vertices.Min(v => v.Y);
            var minZ = model.Vertices.Min(v => v.Z);
            var maxX = model.Vertices.Max(v => v.X);
            var maxY = model.Vertices.Max(v => v.Y);
            var maxZ = model.Vertices.Max(v => v.Z);

            return (new Vector3(minX, minY, minZ), new Vector3(maxX, maxY, maxZ));
        }

        /// <summary>
        /// Finds MSUR surfaces that are spatially near the given bounds.
        /// </summary>
        private List<int> FindMSURSurfacesNearBounds((Vector3 min, Vector3 max) bounds, float tolerance)
        {
            var nearbySurfaces = new List<int>();

            if (MSUR?.Entries == null || MSVT?.Vertices == null || MSVI?.Indices == null)
                return nearbySurfaces;

            for (int surfaceIndex = 0; surfaceIndex < MSUR.Entries.Count; surfaceIndex++)
            {
                var surface = MSUR.Entries[surfaceIndex];

                // Check if any vertex of this surface is near the bounds
                bool isNearby = false;
                for (int i = (int)surface.MsviFirstIndex; i < surface.MsviFirstIndex + surface.IndexCount && i < MSVI.Indices.Count; i++)
                {
                    uint msvtIndex = MSVI.Indices[i];
                    if (msvtIndex < MSVT.Vertices.Count)
                    {
                        var vertex = MSVT.Vertices[(int)msvtIndex];
                        // Use proper coordinate transformation
                        var worldCoords = new Vector3(vertex.Y, vertex.X, vertex.Z);

                        // Check if vertex is within expanded bounds
                        if (worldCoords.X >= bounds.min.X - tolerance && worldCoords.X <= bounds.max.X + tolerance &&
                            worldCoords.Y >= bounds.min.Y - tolerance && worldCoords.Y <= bounds.max.Y + tolerance &&
                            worldCoords.Z >= bounds.min.Z - tolerance && worldCoords.Z <= bounds.max.Z + tolerance)
                        {
                            isNearby = true;
                            break;
                        }
                    }
                }

                if (isNearby)
                {
                    nearbySurfaces.Add(surfaceIndex);
                }
            }

            return nearbySurfaces;
        }
    }

    /// <summary>
    /// Provides information about which chunks are available in a PM4 file.
    /// </summary>
    public class PM4ChunkAvailability
    {
        public bool HasMSLK { get; set; }
        public bool HasMSVT { get; set; }
        public bool HasMSUR { get; set; }
        public bool HasMDSF { get; set; }
        public bool HasMDOS { get; set; }
        public bool HasMPRL { get; set; }
        public bool HasMPRR { get; set; }
    }
} 