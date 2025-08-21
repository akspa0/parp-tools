using System.IO;
using Warcraft.NET.Files;
using Warcraft.NET.Files.Interfaces;
using Warcraft.NET.Attribute;
// PM4 Chunks potentially reusable by PD4
using WoWToolbox.Core.Navigation.PM4.Chunks; 
// PD4 Specific Chunks
using WoWToolbox.Core.Navigation.PD4.Chunks;

namespace WoWToolbox.Core.Navigation.PD4
{
    /// <summary>
    /// Represents a PD4 file, the WMO equivalent of PM4, used server-side.
    /// Based on documentation at https://wowdev.wiki/PD4
    /// </summary>
    public class PD4File : ChunkedFile, IBinarySerializable
    {
        // PD4 Chunks based on wowdev.wiki/PD4

        /// <summary> MVER Chunk </summary>
        public MVER? MVER { get; set; } // Reusing PM4 MVER class

        /// <summary> MCRC Chunk </summary>
        public MCRCChunk? MCRC { get; private set; } // PD4 specific

        /// <summary> MSHD Chunk (Header) </summary>
        public MSHDChunk? MSHD { get; private set; } // Reusing PM4 MSHDChunk class

        /// <summary> MSPV Chunk (Vertices) </summary>
        public MSPVChunk? MSPV { get; private set; } // Reusing PM4 MSPVChunk class

        /// <summary> MSPI Chunk (Indices into MSPV) </summary>
        public MSPIChunk? MSPI { get; private set; } // Reusing PM4 MSPIChunk class

        /// <summary> MSCN Chunk (Unknown purpose, Vector Data) </summary>
        public MSCNChunk? MSCN { get; private set; } // Reusing PM4 MSCNChunk class

        /// <summary> MSLK Chunk </summary>
        public MSLK? MSLK { get; private set; } // Reusing PM4 MSLK class

        /// <summary> MSVT Chunk (Vertices, YXZ order) </summary>
        public MSVTChunk? MSVT { get; private set; } // Reusing PM4 MSVTChunk class

        /// <summary> MSVI Chunk (Vertex Indices into MSVT) </summary>
        public MSVIChunk? MSVI { get; private set; } // Reusing PM4 MSVIChunk class

        /// <summary> MSUR Chunk (Surface Definitions) </summary>
        public MSURChunk? MSUR { get; private set; } // Reusing PM4 MSURChunk class
        
        // Chunks from PM4 NOT listed in PD4.md:
        // MPRL, MPRR, MDBH, MDOS, MDSF, MSRN

        /// <summary>
        /// Initializes a new instance of the <see cref="PD4File"/> class.
        /// </summary>
        public PD4File() { }

        /// <summary>
        /// Initializes a new instance of the <see cref="PD4File"/> class.
        /// </summary>
        /// <param name="inData">The binary data.</param>
        public PD4File(byte[] inData) : base(inData)
        {
            // Base constructor (ChunkedFile) automatically loads chunks
            // into corresponding properties using reflection.
        }

        /// <summary>
        /// Loads a PD4 file from the specified path
        /// </summary>
        public static PD4File FromFile(string path)
        {
            // Check if file exists before reading
            if (!File.Exists(path))
            {
                // Consider throwing a more specific exception or logging
                throw new FileNotFoundException("PD4 file not found.", path);
            }
            return new PD4File(File.ReadAllBytes(path));
        }

        /// <inheritdoc/>
        public new byte[] Serialize(long offset = 0)
        {
            // Important: Serialization needs to be implemented correctly 
            // if saving PD4 files is required. It needs to write only the
            // chunks defined in this class (MVER, MCRC, MSHD, etc.)
            // For now, just calling base serialize might be incorrect.
            // Consider throwing NotImplementedException or implementing properly.
            // return base.Serialize(offset); 
            throw new NotImplementedException("Serialization for PD4File not yet implemented.");
        }

        /// <inheritdoc/>
        public override bool IsReverseSignature()
        {
            // Assuming PD4 uses reverse signatures like PM4
            // Confirm this if issues arise.
            return true;
        }
    }
} 