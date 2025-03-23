using System;
using System.Numerics;
using Microsoft.Extensions.Logging;

namespace WCAnalyzer.Core.Files.PM4.Chunks
{
    /// <summary>
    /// MPRL chunk - Contains position data for the PM4 format
    /// Structure (24 bytes per entry):
    /// - uint16_t _0x00; // Always 0 in version_??.u
    /// - int16_t _0x02; // Always -1 in version_??.u
    /// - uint16_t _0x04;
    /// - uint16_t _0x06;
    /// - C3Vectori position; // 12 bytes (3 * 4 bytes)
    /// - int16_t _0x14;
    /// - uint16_t _0x16;
    /// </summary>
    public class MprlChunk : PM4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MPRL")
        /// </summary>
        public const string SIGNATURE = "MPRL";
        
        /// <summary>
        /// Size of a single entry in bytes
        /// </summary>
        private const int ENTRY_SIZE = 24;
        
        /// <summary>
        /// Creates a new MPRL chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MprlChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
            // Check if data size is multiple of ENTRY_SIZE
            if (_data.Length % ENTRY_SIZE != 0)
            {
                Logger?.LogWarning($"MPRL chunk has irregular size: {_data.Length} bytes. Not a multiple of {ENTRY_SIZE} bytes per entry.");
            }
        }
        
        /// <summary>
        /// Creates a new empty MPRL chunk
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public MprlChunk(ILogger? logger = null)
            : base(SIGNATURE, Array.Empty<byte>(), logger)
        {
        }
        
        /// <summary>
        /// Gets the number of entries in this chunk
        /// </summary>
        /// <returns>Number of entries</returns>
        public int GetEntryCount()
        {
            return _data.Length / ENTRY_SIZE;
        }
        
        /// <summary>
        /// Gets a position from the entry at the specified index
        /// </summary>
        /// <param name="index">Index of the entry</param>
        /// <returns>Vector3 containing position data</returns>
        public Vector3 GetPosition(int index)
        {
            int entryCount = GetEntryCount();
            
            if (index < 0 || index >= entryCount)
            {
                Logger?.LogWarning($"Entry index {index} out of range [0-{entryCount - 1}]. Returning Zero vector.");
                return Vector3.Zero;
            }
            
            int offset = index * ENTRY_SIZE + 8; // Position starts at offset 8 in the entry
            
            float x = BitConverter.ToSingle(_data, offset);
            float y = BitConverter.ToSingle(_data, offset + 4);
            float z = BitConverter.ToSingle(_data, offset + 8);
            
            return new Vector3(x, y, z);
        }
        
        /// <summary>
        /// Gets the data fields from the entry at the specified index
        /// </summary>
        /// <param name="index">Index of the entry</param>
        /// <returns>Tuple containing all fields of the entry</returns>
        public (ushort Field0, short Field2, ushort Field4, ushort Field6, Vector3 Position, short Field20, ushort Field22) GetEntry(int index)
        {
            int entryCount = GetEntryCount();
            
            if (index < 0 || index >= entryCount)
            {
                Logger?.LogWarning($"Entry index {index} out of range [0-{entryCount - 1}]. Returning default values.");
                return (0, -1, 0, 0, Vector3.Zero, 0, 0);
            }
            
            int offset = index * ENTRY_SIZE;
            
            ushort field0 = BitConverter.ToUInt16(_data, offset);
            short field2 = BitConverter.ToInt16(_data, offset + 2);
            ushort field4 = BitConverter.ToUInt16(_data, offset + 4);
            ushort field6 = BitConverter.ToUInt16(_data, offset + 6);
            
            Vector3 position = new Vector3(
                BitConverter.ToSingle(_data, offset + 8),
                BitConverter.ToSingle(_data, offset + 12),
                BitConverter.ToSingle(_data, offset + 16)
            );
            
            short field20 = BitConverter.ToInt16(_data, offset + 20);
            ushort field22 = BitConverter.ToUInt16(_data, offset + 22);
            
            return (field0, field2, field4, field6, position, field20, field22);
        }
        
        /// <summary>
        /// Adds an entry to the chunk
        /// </summary>
        /// <param name="field0">Field at offset 0</param>
        /// <param name="field2">Field at offset 2</param>
        /// <param name="field4">Field at offset 4</param>
        /// <param name="field6">Field at offset 6</param>
        /// <param name="position">Position vector</param>
        /// <param name="field20">Field at offset 20</param>
        /// <param name="field22">Field at offset 22</param>
        public void AddEntry(ushort field0, short field2, ushort field4, ushort field6, Vector3 position, short field20, ushort field22)
        {
            byte[] newData = new byte[_data.Length + ENTRY_SIZE];
            
            // Copy existing data
            Array.Copy(_data, 0, newData, 0, _data.Length);
            
            // Add new entry data
            int offset = _data.Length;
            Array.Copy(BitConverter.GetBytes(field0), 0, newData, offset, 2);
            Array.Copy(BitConverter.GetBytes(field2), 0, newData, offset + 2, 2);
            Array.Copy(BitConverter.GetBytes(field4), 0, newData, offset + 4, 2);
            Array.Copy(BitConverter.GetBytes(field6), 0, newData, offset + 6, 2);
            Array.Copy(BitConverter.GetBytes(position.X), 0, newData, offset + 8, 4);
            Array.Copy(BitConverter.GetBytes(position.Y), 0, newData, offset + 12, 4);
            Array.Copy(BitConverter.GetBytes(position.Z), 0, newData, offset + 16, 4);
            Array.Copy(BitConverter.GetBytes(field20), 0, newData, offset + 20, 2);
            Array.Copy(BitConverter.GetBytes(field22), 0, newData, offset + 22, 2);
            
            // Update data
            _data = newData;
        }
        
        /// <summary>
        /// Gets a formatted string representation of this chunk
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            return $"{SIGNATURE} Chunk: {GetEntryCount()} entries";
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