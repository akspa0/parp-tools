using System;
using System.Text;
using Microsoft.Extensions.Logging;

namespace WCAnalyzer.Core.Files.PM4.Chunks
{
    /// <summary>
    /// MDBF chunk - Destructible Building Filename, a sub-chunk of MDBH
    /// Contains filenames associated with destructible buildings
    /// </summary>
    public class MdbfChunk : PM4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MDBF")
        /// </summary>
        public const string SIGNATURE = "MDBF";
        
        /// <summary>
        /// The filename stored in this chunk
        /// </summary>
        private string _filename;
        
        /// <summary>
        /// Creates a new MDBF chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MdbfChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
            // Extract the filename from the data
            _filename = ExtractFilename();
        }
        
        /// <summary>
        /// Creates a new MDBF chunk with the specified filename
        /// </summary>
        /// <param name="filename">The filename to store</param>
        /// <param name="logger">Optional logger</param>
        public MdbfChunk(string filename, ILogger? logger = null)
            : base(SIGNATURE, Array.Empty<byte>(), logger)
        {
            _filename = filename;
            
            // Update the data with the filename
            _data = Encoding.ASCII.GetBytes(filename);
            
            // Ensure null-termination
            if (_data.Length == 0 || _data[_data.Length - 1] != 0)
            {
                byte[] newData = new byte[_data.Length + 1];
                Array.Copy(_data, 0, newData, 0, _data.Length);
                newData[_data.Length] = 0; // Null terminator
                _data = newData;
            }
        }
        
        /// <summary>
        /// Extracts the filename from the raw data
        /// </summary>
        /// <returns>The extracted filename</returns>
        private string ExtractFilename()
        {
            if (_data.Length == 0)
            {
                return string.Empty;
            }
            
            // Find the null terminator
            int nullTermPos = Array.IndexOf(_data, (byte)0);
            if (nullTermPos == -1)
            {
                // No null terminator found, use the entire data
                nullTermPos = _data.Length;
                Logger?.LogWarning("MDBF chunk data is not null-terminated.");
            }
            
            // Extract the filename
            return Encoding.ASCII.GetString(_data, 0, nullTermPos);
        }
        
        /// <summary>
        /// Gets the filename stored in this chunk
        /// </summary>
        /// <returns>The filename</returns>
        public string GetFilename()
        {
            return _filename;
        }
        
        /// <summary>
        /// Sets the filename stored in this chunk
        /// </summary>
        /// <param name="filename">The new filename</param>
        public void SetFilename(string filename)
        {
            _filename = filename;
            
            // Update the data with the new filename
            byte[] filenameBytes = Encoding.ASCII.GetBytes(filename);
            _data = new byte[filenameBytes.Length + 1]; // +1 for null terminator
            Array.Copy(filenameBytes, 0, _data, 0, filenameBytes.Length);
            _data[filenameBytes.Length] = 0; // Null terminator
        }
        
        /// <summary>
        /// Gets a formatted string representation of this chunk
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            return $"{SIGNATURE} Chunk: Filename \"{_filename}\"";
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