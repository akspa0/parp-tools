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
        /// Extracts individual buildings using root node detection and dual geometry assembly.
        /// </summary>
        public List<CompleteWMOModel> ExtractBuildings()
        {
            var buildings = new List<CompleteWMOModel>();
            if (MSLK == null || MSPI == null || MSPV == null || MSUR == null || MSVI == null || MSVT == null)
                return buildings;
            // Root node detection: self-referencing nodes
            var rootNodes = MSLK.Entries
                .Select((entry, idx) => new { entry, idx })
                .Where(x => x.entry.Unknown_0x04 == (uint)x.idx)
                .ToList();
            foreach (var root in rootNodes)
            {
                var model = new CompleteWMOModel { FileName = $"Building_{root.idx}" };
                // Phase 1: Structural geometry from MSLK/MSPI/MSPV
                var structuralEntries = MSLK.Entries
                    .Where(e => e.Unknown_0x04 == root.idx && e.HasGeometry)
                    .ToList();
                var vertexMap = new Dictionary<int, int>();
                foreach (var entry in structuralEntries)
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
                // Phase 2: Render surfaces from MSUR/MSVI/MSVT
                foreach (var surface in MSUR.Entries)
                {
                    if (!surface.HasValidGeometry) continue;
                    int start = (int)surface.MsviFirstIndex;
                    int count = (int)surface.IndexCount;
                    for (int i = 0; i + 2 < count; i += 3)
                    {
                        int a = (int)MSVI.Indices[start + i];
                        int b = (int)MSVI.Indices[start + i + 1];
                        int c = (int)MSVI.Indices[start + i + 2];
                        if (a < MSVT.Vertices.Count && b < MSVT.Vertices.Count && c < MSVT.Vertices.Count)
                        {
                            // Add vertices if not already present
                            int va = model.Vertices.Count;
                            int vb = model.Vertices.Count + 1;
                            int vc = model.Vertices.Count + 2;
                            model.Vertices.Add(new Vector3(MSVT.Vertices[a].X, MSVT.Vertices[a].Y, MSVT.Vertices[a].Z));
                            model.Vertices.Add(new Vector3(MSVT.Vertices[b].X, MSVT.Vertices[b].Y, MSVT.Vertices[b].Z));
                            model.Vertices.Add(new Vector3(MSVT.Vertices[c].X, MSVT.Vertices[c].Y, MSVT.Vertices[c].Z));
                            model.TriangleIndices.Add(va);
                            model.TriangleIndices.Add(vb);
                            model.TriangleIndices.Add(vc);
                        }
                    }
                }
                if (model.Vertices.Count > 0 && model.TriangleIndices.Count > 0)
                    buildings.Add(model);
            }
            return buildings;
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