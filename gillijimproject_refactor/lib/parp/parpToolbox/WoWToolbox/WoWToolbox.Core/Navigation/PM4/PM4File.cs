using System;
using System.IO;
using Warcraft.NET.Files;
using Warcraft.NET.Files.Interfaces;
using WoWToolbox.Core.Navigation.PM4.Chunks;
using Warcraft.NET.Attribute;
using WoWToolbox.Core.Helpers;

namespace WoWToolbox.Core.Navigation.PM4
{
    /// <summary>
    /// Represents a PM4 file, which is a variant of ADT files used for phasing/pathing data
    /// </summary>
    public class PM4File : ChunkedFile, IBinarySerializable
    {
        /// <summary>
        /// Gets or sets the MVER chunk
        /// </summary>
        [ChunkOptional]
        public MVER? MVER { get; set; }

        /// <summary>
        /// Gets or sets the MSHD chunk (Header).
        /// </summary>
        [ChunkOptional]
        public MSHDChunk? MSHD { get; private set; }

        /// <summary>
        /// Gets or sets the MSLK chunk
        /// </summary>
        [ChunkOptional]
        public MSLK? MSLK { get; set; }

        /// <summary>
        /// Gets or sets the MSPI chunk (Indices into MSPV).
        /// </summary>
        [ChunkOptional]
        public MSPIChunk? MSPI { get; set; }

        /// <summary>
        /// Gets or sets the MSPV chunk (Vertices).
        /// </summary>
        [ChunkOptional]
        public MSPVChunk? MSPV { get; set; }

        /// <summary>
        /// Gets or sets the MSVT chunk (Vertices).
        /// </summary>
        public MSVTChunk? MSVT { get; set; }

        /// <summary>
        /// Gets or sets the MSVI chunk (Vertex Indices).
        /// </summary>
        public MSVIChunk? MSVI { get; set; }

        /// <summary>
        /// Gets or sets the MSUR chunk (Surface Definitions).
        /// </summary>
        public MSURChunk? MSUR { get; set; }

        /// <summary>
        /// Gets or sets the MSCN chunk (Exterior Vertices; previously misinterpreted as normals).
        /// </summary>
        public MSCNChunk? MSCN { get; set; }

        /// <summary>
        /// Gets or sets the MSRN chunk (Mesh Surface Referenced Normals).
        /// </summary>
        [ChunkOptional]
        public MSRNChunk? MSRN { get; set; }

        /// <summary>
        /// Gets or sets the MPRL chunk (Position Data).
        /// </summary>
        public MPRLChunk? MPRL { get; private set; }

        /// <summary>
        /// Gets or sets the MPRR chunk (Reference Data).
        /// </summary>
        public MPRRChunk? MPRR { get; private set; }

        /// <summary>
        /// Gets or sets the MDBH chunk (Destructible Building Header).
        /// </summary>
        [ChunkOptional]
        public MDBHChunk? MDBH { get; private set; }

        /// <summary>
        /// Gets or sets the MDOS chunk (Object Data).
        /// </summary>
        [ChunkOptional]
        public MDOSChunk? MDOS { get; private set; }

        /// <summary>
        /// Gets or sets the MDSF chunk (Structure Data).
        /// </summary>
        public MDSFChunk? MDSF { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4File"/> class.
        /// </summary>
        public PM4File() { }

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4File"/> class.
        /// </summary>
        /// <param name="inData">The binary data.</param>
        public PM4File(byte[] inData) : base(inData)
        {
            // Base constructor (ChunkedFile) automatically loads chunks
            // into corresponding properties using reflection. No need to 
            // manually retrieve them here.

            // Optional Validation/Linking Step can remain if needed
            // ValidateMspIndices();
        }

        /// <summary>
        /// Loads a PM4 file from the specified path
        /// </summary>
        public static PM4File FromFile(string path)
        {
            return new PM4File(File.ReadAllBytes(path));
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

        // Optional: Add validation method
        // private void ValidateMspIndices()
        // {
        //     if (MSPI != null && MSPV != null)
        //     {
        //         if (!MSPI.ValidateIndices(MSPV.Vertices.Count))
        //         {
        //             Console.WriteLine("Warning: MSPI contains indices out of bounds for MSPV.");
        //             // Potentially throw an exception or handle invalid data
        //         }
        //     }
        //     
        //     // Could add MSLK -> MSPI range validation here too
        // }

        // Helper for IFFFile interface if needed (can expose specific chunks)
        // ... existing code ...
    }
} 