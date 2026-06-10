using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MWMO chunk - Contains WMO filenames
    /// </summary>
    public class MwmoChunk : ADTChunk
    {
        /// <summary>
        /// The MWMO chunk signature
        /// </summary>
        public const string SIGNATURE = "MWMO";

        /// <summary>
        /// Gets the list of WMO filenames
        /// </summary>
        public List<string> Filenames { get; } = new List<string>();

        /// <summary>
        /// Gets the raw data for the filenames section
        /// </summary>
        public byte[] FilenamesData { get; private set; } = Array.Empty<byte>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MwmoChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MwmoChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            Parse();
        }

        /// <summary>
        /// Parses the MWMO chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MWMO chunk has no data");
                    return;
                }

                // Store the raw data for later use with offsets
                FilenamesData = new byte[Data.Length];
                Array.Copy(Data, FilenamesData, Data.Length);
                
                using var ms = new MemoryStream(Data);
                using var reader = new BinaryReader(ms);
                
                // Read filenames (null-terminated strings)
                while (reader.BaseStream.Position < reader.BaseStream.Length)
                {
                    var filename = ReadNullTerminatedString(reader);
                    if (!string.IsNullOrEmpty(filename))
                    {
                        Filenames.Add(filename);
                    }
                }
                
                Logger?.LogDebug($"MWMO: Read {Filenames.Count} WMO filenames");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MWMO chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Writes the chunk data to a binary writer
        /// </summary>
        /// <param name="writer">The binary writer to write to</param>
        public override void Write(BinaryWriter writer)
        {
            if (writer == null)
            {
                Logger?.LogError("Cannot write MWMO chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                
                // Calculate the total size of all filenames with null terminators
                int totalSize = 0;
                foreach (var filename in Filenames)
                {
                    totalSize += Encoding.ASCII.GetByteCount(filename) + 1; // +1 for null terminator
                }
                
                writer.Write(totalSize);
                
                // Write each filename with a null terminator
                foreach (var filename in Filenames)
                {
                    byte[] filenameBytes = Encoding.ASCII.GetBytes(filename);
                    writer.Write(filenameBytes);
                    writer.Write((byte)0); // Null terminator
                }
                
                Logger?.LogDebug($"MWMO: Wrote {Filenames.Count} WMO filenames");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MWMO chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Gets the filename at the specified offset in the raw data
        /// </summary>
        /// <param name="offset">The offset into the raw data</param>
        /// <returns>The filename at the offset, or null if the offset is invalid</returns>
        public string GetFilenameAtOffset(int offset)
        {
            if (FilenamesData == null || offset < 0 || offset >= FilenamesData.Length)
            {
                return null;
            }
            
            try
            {
                List<byte> filenameBytes = new List<byte>();
                int i = offset;
                
                // Read bytes until we hit a null terminator or the end of the data
                while (i < FilenamesData.Length && FilenamesData[i] != 0)
                {
                    filenameBytes.Add(FilenamesData[i]);
                    i++;
                }
                
                return Encoding.ASCII.GetString(filenameBytes.ToArray());
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error getting filename at offset {offset}: {ex.Message}");
                return null;
            }
        }
        
        /// <summary>
        /// Reads a null-terminated string from the binary reader
        /// </summary>
        /// <param name="reader">The binary reader to read from</param>
        /// <returns>The string that was read</returns>
        private static string ReadNullTerminatedString(BinaryReader reader)
        {
            var bytes = new List<byte>();
            byte b;
            
            // Read bytes until we hit a null terminator or the end of the stream
            while (reader.BaseStream.Position < reader.BaseStream.Length && (b = reader.ReadByte()) != 0)
            {
                bytes.Add(b);
            }
            
            return Encoding.ASCII.GetString(bytes.ToArray());
        }
    }
} 