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
        /// Adds render surfaces from MSUR/MSVI/MSVT with improved face generation.
        /// </summary>
        private void AddRenderSurfaces(CompleteWMOModel model)
        {
            var vertexStartIndex = model.Vertices.Count;

            foreach (var surface in MSUR.Entries)
            {
                if (!surface.HasValidGeometry) continue;

                int start = (int)surface.MsviFirstIndex;
                int count = (int)surface.IndexCount;

                // Ensure we have valid triangle count
                if (count < 3 || count % 3 != 0) continue;

                for (int i = 0; i + 2 < count; i += 3)
                {
                    if (start + i + 2 >= MSVI.Indices.Count) break;

                    int a = (int)MSVI.Indices[start + i];
                    int b = (int)MSVI.Indices[start + i + 1];
                    int c = (int)MSVI.Indices[start + i + 2];

                    // Validate vertex indices
                    if (a >= MSVT.Vertices.Count || b >= MSVT.Vertices.Count || c >= MSVT.Vertices.Count) continue;
                    if (a < 0 || b < 0 || c < 0) continue;

                    // Skip degenerate triangles
                    if (a == b || b == c || a == c) continue;

                    // Add vertices with coordinate transformation
                    var va = MSVT.Vertices[a];
                    var vb = MSVT.Vertices[b];
                    var vc = MSVT.Vertices[c];

                    int idxA = model.Vertices.Count;
                    int idxB = model.Vertices.Count + 1;
                    int idxC = model.Vertices.Count + 2;

                    model.Vertices.Add(new Vector3(va.X, va.Y, va.Z));
                    model.Vertices.Add(new Vector3(vb.X, vb.Y, vb.Z));
                    model.Vertices.Add(new Vector3(vc.X, vc.Y, vc.Z));

                    // Add triangle with consistent winding order (counter-clockwise)
                    model.TriangleIndices.Add(idxA);
                    model.TriangleIndices.Add(idxB);
                    model.TriangleIndices.Add(idxC);
                }
            }
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