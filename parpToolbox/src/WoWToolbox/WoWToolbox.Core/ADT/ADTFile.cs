using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Warcraft.NET.Attribute;
using Warcraft.NET.Files;
using Warcraft.NET.Files.Interfaces;
using Warcraft.NET.Files.ADT.Chunks;
using Warcraft.NET.Files.ADT.Chunks.MoP; // For MCRF, potentially others
using Warcraft.NET.Files.ADT.Chunks.Wotlk;
using Warcraft.NET.Files.ADT.Chunks.Cata;
using Warcraft.NET.Files.ADT.Chunks.Legion; // Contains new MCRD
using Warcraft.NET.Files.ADT.Chunks.BfA; // Contains MTCG
using Warcraft.NET.Extensions; // Add this line for BinaryReader extensions
// Add other version-specific chunk namespaces if needed

namespace WoWToolbox.Core.ADT
{
    /// <summary>
    /// Represents an ADT (Area Definition Terrain) file, using Warcraft.NET for chunk loading.
    /// Inherits from ChunkedFile to automatically load recognized chunks based on properties.
    /// </summary>
    public class ADTFile : ChunkedFile
    {
        // --- Core Chunks --- 
        [ChunkOptional]
        public MVER? MVER { get; private set; } // Version
        [ChunkOptional]
        public MHDR? MHDR { get; private set; } // Header (Offsets)
        
        // --- Object References ---
        [ChunkOptional]
        public MTEX? MTEX { get; private set; } // Texture file names
        [ChunkOptional]
        public MMDX? MMDX { get; private set; } // Model (M2) file names
        [ChunkOptional]
        public MMID? MMID { get; private set; } // Model file data IDs (used with MMDX)
        [ChunkOptional]
        public MWMO? MWMO { get; private set; } // WMO file names
        [ChunkOptional]
        public MWID? MWID { get; private set; } // WMO file data IDs (used with MWMO)
        
        // --- Object Placements --- (Crucial for this task)
        [ChunkOptional]
        public MDDF? MDDF { get; private set; } // Model (M2 / Doodad) placements
        [ChunkOptional]
        public MODF? MODF { get; private set; } // WMO placements

        // --- Terrain Chunks (Examples - Add more if needed) ---
        // Note: MCNK is typically handled specially, maybe not as a direct property here
        // [ChunkOptional]
        // public List<MCNK>? MCNKs { get; } 
        [ChunkOptional]
        public MFBO? MFBO { get; private set; } // Max/Min Height Bounding Box
        [ChunkOptional]
        public MH2O? MH2O { get; private set; } // Liquid data
        [ChunkOptional]
        public MTXF? MTXF { get; private set; } // Texture Flags
        // [ChunkOptional]
        // public MTCG? MTCG { get; } // Commented out: Build error - Type not found. Namespace Warcraft.NET.Files.ADT.Chunks.BfA may be incorrect.
        
        // --- Optional / Version Specific (Examples - Add more if needed) ---
        // MCIN, MCVT are likely parts of MCNK, not top-level chunks loaded this way
        // [ChunkOptional]
        // public MCIN[]? MCIN { get; }
        // [ChunkOptional]
        // public MCRF? MCRF { get; } // Removed: Documented as pre-Cata MCNK sub-chunk
        // [ChunkOptional]
        // public MCRD? MCRD { get; } // Commented out: Build error - Type not found. Namespace Warcraft.NET.Files.ADT.Chunks.Legion may be incorrect.
        // ... Add other chunks like MCAL, MCSH, MCLQ, MCLV etc. if needed ... 

        /// <summary>
        /// Initializes a new instance of the <see cref="ADTFile"/> class by loading from a file path.
        /// </summary>
        /// <param name="filePath">Path to the ADT file.</param>
        /// <param name="ignoreChunkErrors">If true, continues loading even if some chunks fail.</param>
        public ADTFile(string filePath, bool ignoreChunkErrors = false) 
            : base(File.ReadAllBytes(filePath)) // Read file to bytes first
        {
            // ignoreChunkErrors is not used by base, handle potential errors during property access if needed
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ADTFile"/> class by loading from a stream.
        /// </summary>
        /// <param name="stream">The stream containing ADT data.</param>
        /// <param name="ignoreChunkErrors">If true, continues loading even if some chunks fail.</param>
        public ADTFile(Stream stream, bool ignoreChunkErrors = false) 
            : base(ReadStreamFully(stream)) // Read stream to bytes first
        {
            // ignoreChunkErrors is not used by base, handle potential errors during property access if needed
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ADTFile"/> class by loading from byte data.
        /// </summary>
        /// <param name="data">Byte array containing ADT data.</param>
        /// <param name="ignoreChunkErrors">If true, continues loading even if some chunks fail.</param>
        public ADTFile(byte[] data, bool ignoreChunkErrors = false) 
            : base(data) // Pass byte array directly. Base constructor attempts reflection loading.
        {
             // ignoreChunkErrors is not used by base, handle potential errors during property access if needed

             // Manually load potentially problematic chunks after base constructor tries reflection,
             // using BinaryReader extensions, to bypass potential reflection setter issues.
             using (var ms = new MemoryStream(data))
             using (var br = new BinaryReader(ms))
             {
                 try
                 {
                     // SeekChunk signature constants are usually public const string in the chunk class
                     if (br.SeekChunk(MDDF.Signature)) 
                     {
                         this.MDDF = br.ReadIFFChunk<MDDF>(false, false); // returnDefault=false, fromBegin=false (already sought)
                     }
                 }
                 catch (Exception ex) 
                 {
                     // Log or handle exception if MDDF is missing or fails to load.
                     // Base constructor might have already logged. This provides redundancy.
                     Console.WriteLine($"Warning: Could not manually load MDDF chunk: {ex.Message}");
                     if (!ignoreChunkErrors) throw; // Rethrow if errors shouldn't be ignored
                 }

                 // Reset stream position or use appropriate SeekChunk parameters if needed between reads.
                 // Since SeekChunk searches from current position by default, resetting might not be necessary
                 // if chunks are expected in order, but safer to handle potential overlaps/re-reads.
                 // ms.Position = 0; // Optional: Reset position if seeking from start each time

                 try
                 {
                     if (br.SeekChunk(MODF.Signature)) 
                     {
                         this.MODF = br.ReadIFFChunk<MODF>(false, false);
                     }
                 }
                 catch (Exception ex)
                 {
                     Console.WriteLine($"Warning: Could not manually load MODF chunk: {ex.Message}");
                     if (!ignoreChunkErrors) throw;
                 }
             }
        }

        // Helper method to read a stream into a byte array
        private static byte[] ReadStreamFully(Stream input)
        {
            using (MemoryStream ms = new MemoryStream())
            {
                input.CopyTo(ms);
                return ms.ToArray();
            }
        }
    }
} 