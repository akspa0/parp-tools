using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Common.Extensions;
using WCAnalyzer.Core.Files.Interfaces;

namespace WCAnalyzer.Core.Files.PM4.Chunks
{
    /// <summary>
    /// MDBH chunk - Destructible building header with MDBI and MDBF sub-chunks
    /// Format:
    /// - Entry count (uint32)
    /// - For each entry:
    ///   - MDBI chunk (contains index)
    ///   - 3x MDBF chunks (containing filenames for destroyed, damaged, and intact states)
    /// </summary>
    public class MdbhChunk : PM4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MDBH")
        /// </summary>
        public const string SIGNATURE = "MDBH";
        
        /// <summary>
        /// MDBI sub-chunk signature
        /// </summary>
        private const string MDBI_SIGNATURE = "MDBI";
        
        /// <summary>
        /// MDBF sub-chunk signature
        /// </summary>
        private const string MDBF_SIGNATURE = "MDBF";
        
        /// <summary>
        /// Entry count field size
        /// </summary>
        private const int ENTRY_COUNT_SIZE = 4;
        
        /// <summary>
        /// Collection of parsed entries
        /// </summary>
        private List<DestructibleBuildingEntry> _entries = new List<DestructibleBuildingEntry>();
        
        /// <summary>
        /// Class representing a single destructible building entry
        /// </summary>
        public class DestructibleBuildingEntry
        {
            /// <summary>
            /// Index value from MDBI chunk
            /// </summary>
            public uint Index { get; set; }
            
            /// <summary>
            /// Filename for the destroyed state (from first MDBF chunk)
            /// </summary>
            public string DestroyedFilename { get; set; } = string.Empty;
            
            /// <summary>
            /// Filename for the damaged state (from second MDBF chunk)
            /// </summary>
            public string DamagedFilename { get; set; } = string.Empty;
            
            /// <summary>
            /// Filename for the intact state (from third MDBF chunk)
            /// </summary>
            public string IntactFilename { get; set; } = string.Empty;
        }
        
        /// <summary>
        /// Creates a new MDBH chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MdbhChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
            ParseEntries();
        }
        
        /// <summary>
        /// Creates a new empty MDBH chunk
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public MdbhChunk(ILogger? logger = null)
            : base(SIGNATURE, new byte[0], logger)
        {
        }
        
        /// <summary>
        /// Parses the entries in the chunk
        /// </summary>
        private void ParseEntries()
        {
            _entries.Clear();
            
            // Need at least 4 bytes for entry count
            if (_data.Length < ENTRY_COUNT_SIZE)
            {
                return;
            }
            
            uint entryCount = BitConverter.ToUInt32(_data, 0);
            int offset = ENTRY_COUNT_SIZE;
            
            for (int i = 0; i < entryCount && offset < _data.Length; i++)
            {
                DestructibleBuildingEntry entry = new DestructibleBuildingEntry();
                bool validEntry = true;
                
                // Parse MDBI chunk for index
                if (!ParseMdbiChunk(ref offset, ref entry.Index))
                {
                    validEntry = false;
                }
                
                // Parse first MDBF chunk for destroyed filename
                if (validEntry && !ParseMdbfChunk(ref offset, ref entry.DestroyedFilename))
                {
                    validEntry = false;
                }
                
                // Parse second MDBF chunk for damaged filename
                if (validEntry && !ParseMdbfChunk(ref offset, ref entry.DamagedFilename))
                {
                    validEntry = false;
                }
                
                // Parse third MDBF chunk for intact filename
                if (validEntry && !ParseMdbfChunk(ref offset, ref entry.IntactFilename))
                {
                    validEntry = false;
                }
                
                if (validEntry)
                {
                    _entries.Add(entry);
                }
            }
        }
        
        /// <summary>
        /// Parses an MDBI sub-chunk and extracts the index
        /// </summary>
        /// <param name="offset">Current offset, updated after parsing</param>
        /// <param name="index">Output index value</param>
        /// <returns>True if successful, false otherwise</returns>
        private bool ParseMdbiChunk(ref int offset, ref uint index)
        {
            // Check if we have enough data for signature (4) + size (4) + index (4)
            if (offset + 12 > _data.Length)
            {
                Logger?.LogWarning($"Not enough data to parse MDBI chunk at offset {offset}");
                return false;
            }
            
            // Verify signature
            string signature = Encoding.ASCII.GetString(_data, offset, 4);
            if (signature != MDBI_SIGNATURE)
            {
                Logger?.LogWarning($"Expected MDBI signature but found {signature} at offset {offset}");
                return false;
            }
            offset += 4;
            
            // Get size
            uint size = BitConverter.ToUInt32(_data, offset);
            offset += 4;
            
            // Size should be 4 for a uint32
            if (size != 4)
            {
                Logger?.LogWarning($"Expected MDBI data size of 4 but found {size} at offset {offset - 4}");
                return false;
            }
            
            // Read index
            index = BitConverter.ToUInt32(_data, offset);
            offset += 4;
            
            return true;
        }
        
        /// <summary>
        /// Parses an MDBF sub-chunk and extracts the filename
        /// </summary>
        /// <param name="offset">Current offset, updated after parsing</param>
        /// <param name="filename">Output filename</param>
        /// <returns>True if successful, false otherwise</returns>
        private bool ParseMdbfChunk(ref int offset, ref string filename)
        {
            // Check if we have enough data for signature (4) + size (4)
            if (offset + 8 > _data.Length)
            {
                Logger?.LogWarning($"Not enough data to parse MDBF chunk at offset {offset}");
                return false;
            }
            
            // Verify signature
            string signature = Encoding.ASCII.GetString(_data, offset, 4);
            if (signature != MDBF_SIGNATURE)
            {
                Logger?.LogWarning($"Expected MDBF signature but found {signature} at offset {offset}");
                return false;
            }
            offset += 4;
            
            // Get size
            uint size = BitConverter.ToUInt32(_data, offset);
            offset += 4;
            
            // Check if we have enough data for the string
            if (offset + size > _data.Length)
            {
                Logger?.LogWarning($"Not enough data to read MDBF string of size {size} at offset {offset}");
                return false;
            }
            
            // Read null-terminated string
            int maxLength = (int)size;
            int stringLength = 0;
            while (stringLength < maxLength && _data[offset + stringLength] != 0)
            {
                stringLength++;
            }
            
            filename = Encoding.ASCII.GetString(_data, offset, stringLength);
            offset += (int)size; // Skip the entire size including null terminator
            
            return true;
        }
        
        /// <summary>
        /// Gets the number of entries in the chunk
        /// </summary>
        /// <returns>Number of entries</returns>
        public int GetEntryCount()
        {
            return _entries.Count;
        }
        
        /// <summary>
        /// Gets an entry at the specified index
        /// </summary>
        /// <param name="index">Entry index (0-based)</param>
        /// <returns>Destructible building entry</returns>
        /// <exception cref="ArgumentOutOfRangeException">If index is out of range</exception>
        public DestructibleBuildingEntry GetEntry(int index)
        {
            if (index < 0 || index >= _entries.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(index), $"Entry index must be between 0 and {_entries.Count - 1}");
            }
            
            return _entries[index];
        }
        
        /// <summary>
        /// Gets a formatted string representation of this chunk
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            return $"{SIGNATURE} Chunk: {_entries.Count} destructible building entries";
        }
        
        /// <summary>
        /// Writes this chunk to a byte array
        /// </summary>
        /// <returns>Byte array containing chunk data</returns>
        public override byte[] Write()
        {
            return _data;
        }
    }
} 