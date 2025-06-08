using System;
using System.IO;
using Warcraft.NET.Files;
using Warcraft.NET.Files.Interfaces;
using Warcraft.NET.Attribute;
using WoWToolbox.Core.v2.Models.PM4.Chunks;

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