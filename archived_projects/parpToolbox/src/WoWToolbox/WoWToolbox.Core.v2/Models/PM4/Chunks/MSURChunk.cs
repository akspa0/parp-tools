using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    /// <summary>
    /// Optimized MSUR entry with decoded surface normal and height fields.
    /// Represents surface definitions with complete geometric and material information.
    /// </summary>
    public class MsurEntry : IBinarySerializable
    {
        #region Core Fields - DECODED THROUGH STATISTICAL ANALYSIS

        // Flags and index data (first 4 bytes)
        public byte FlagsOrUnknown_0x00 { get; set; }          // _0x00 - Meaning TBD. Might be flags.
        public byte IndexCount { get; set; }                   // _0x01 - Number of indices in MSVI used by this surface.
        public byte Unknown_0x02 { get; set; }                 // _0x02 - Meaning TBD.
        public byte Padding_0x03 { get; set; }                 // _0x03 - Likely padding.
        
        // DECODED FIELDS - Surface Normal and Height System
        public float SurfaceNormalX { get; set; }              // _0x04 - Surface Normal X
        public float SurfaceNormalY { get; set; }              // _0x08 - Surface Normal Y  
        public float SurfaceNormalZ { get; set; }              // _0x0C - Surface Normal Z
        public float SurfaceHeight { get; set; }               // _0x10 - Surface Height/Y-coordinate

        // Geometry references (last 12 bytes)
        public uint MsviFirstIndex { get; set; }

        // Shim for legacy code expecting FirstIndex
        public uint FirstIndex => MsviFirstIndex;               // _0x14 - Starting index in MSVI for this surface
        public uint MdosIndex { get; set; }                    // _0x18 - Index into MDOS (Destructible Object States)
        public uint Unknown_0x1C { get; set; }                 // _0x1C - Meaning TBD.

        #endregion

        #region Decoded Property Accessors

        /// <summary>Gets the surface normal as a Vector3</summary>
        public Vector3 SurfaceNormal => new Vector3(SurfaceNormalX, SurfaceNormalY, SurfaceNormalZ);

        /// <summary>Gets the normalized surface normal vector</summary>
        public Vector3 NormalizedSurfaceNormal => Vector3.Normalize(SurfaceNormal);

        /// <summary>Checks if the surface normal is properly normalized</summary>
        public bool IsNormalValid
        {
            get
            {
                var magnitude = SurfaceNormal.Length();
                return Math.Abs(magnitude - 1.0f) < 0.01f; // Allow small floating-point errors
            }
        }

        /// <summary>Checks if this surface has valid geometry references</summary>
        public bool HasValidGeometry => IndexCount >= 3; // At least one triangle

        /// <summary>Gets the triangle count for this surface</summary>
        public int TriangleCount => (int)(IndexCount / 3);

        #endregion

        public const int BaseStructSize = 32; // FIXED: Must be 32 bytes to match Core implementation

        /// <summary>Initializes a new instance of the MsurEntry class</summary>
        public MsurEntry() { }

        #region IBinarySerializable Implementation

        /// <inheritdoc/>
        public void LoadBinaryData(byte[] inData)
        {
            if (inData == null || inData.Length < BaseStructSize)
                throw new ArgumentException($"Input data must be at least {BaseStructSize} bytes.", nameof(inData));

            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        /// <inheritdoc/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Load(BinaryReader br)
        {
            // Read flags and index count (4 bytes)
            FlagsOrUnknown_0x00 = br.ReadByte();
            IndexCount = br.ReadByte();
            Unknown_0x02 = br.ReadByte();
            Padding_0x03 = br.ReadByte();
            
            // Read surface normal and height (16 bytes)
            SurfaceNormalX = br.ReadSingle();
            SurfaceNormalY = br.ReadSingle();
            SurfaceNormalZ = br.ReadSingle();
            SurfaceHeight = br.ReadSingle();
            
            // Read geometry references (12 bytes)
            MsviFirstIndex = br.ReadUInt32();
            MdosIndex = br.ReadUInt32();
            Unknown_0x1C = br.ReadUInt32();
        }

        /// <inheritdoc/>
        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream(BaseStructSize);
            using var bw = new BinaryWriter(ms);
            Write(bw);
            return ms.ToArray();
        }

        /// <summary>Writes the entry to a BinaryWriter</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Write(BinaryWriter bw)
        {
            // Write flags and index count (4 bytes)
            bw.Write(FlagsOrUnknown_0x00);
            bw.Write(IndexCount);
            bw.Write(Unknown_0x02);
            bw.Write(Padding_0x03);
            
            // Write surface normal and height (16 bytes)
            bw.Write(SurfaceNormalX);
            bw.Write(SurfaceNormalY);
            bw.Write(SurfaceNormalZ);
            bw.Write(SurfaceHeight);
            
            // Write geometry references (12 bytes)
            bw.Write(MsviFirstIndex);
            bw.Write(MdosIndex);
            bw.Write(Unknown_0x1C);
        }

        /// <inheritdoc/>
        public uint GetSize() => BaseStructSize;

        #endregion

        #region Surface Analysis Methods

        /// <summary>
        /// Creates a signature for duplicate surface detection.
        /// </summary>
        /// <returns>Unique signature string for this surface</returns>
        public string CreateSignature()
        {
            return $"{MsviFirstIndex}_{IndexCount}_{SurfaceNormalX:F3}_{SurfaceNormalY:F3}_{SurfaceNormalZ:F3}_{SurfaceHeight:F3}";
        }

        /// <summary>
        /// Checks if this surface is likely a duplicate of another.
        /// </summary>
        /// <param name="other">Other surface to compare</param>
        /// <param name="tolerance">Floating-point comparison tolerance</param>
        /// <returns>True if surfaces appear to be duplicates</returns>
        public bool IsLikelyDuplicate(MsurEntry other, float tolerance = 0.001f)
        {
            return MsviFirstIndex == other.MsviFirstIndex &&
                   IndexCount == other.IndexCount &&
                   Math.Abs(SurfaceNormalX - other.SurfaceNormalX) < tolerance &&
                   Math.Abs(SurfaceNormalY - other.SurfaceNormalY) < tolerance &&
                   Math.Abs(SurfaceNormalZ - other.SurfaceNormalZ) < tolerance &&
                   Math.Abs(SurfaceHeight - other.SurfaceHeight) < tolerance;
        }

        /// <summary>
        /// Calculates the area approximation based on normal magnitude and triangle count.
        /// </summary>
        /// <returns>Estimated surface area</returns>
        public float EstimateSurfaceArea()
        {
            var normalMagnitude = SurfaceNormal.Length();
            var triangleCount = TriangleCount;
            
            // Simple area estimation based on normal magnitude and triangle count
            return normalMagnitude * triangleCount * 10.0f; // Scaling factor may need adjustment
        }

        #endregion

        public override string ToString()
        {
            return $"MSUR Entry [MSVI:{MsviFirstIndex}+{IndexCount}, Normal:({SurfaceNormalX:F2},{SurfaceNormalY:F2},{SurfaceNormalZ:F2}), Height:{SurfaceHeight:F1}]";
        }
    }

    /// <summary>
    /// Optimized MSUR chunk with performance improvements and advanced surface analysis.
    /// Contains surface definitions with decoded normal vectors and height information.
    /// </summary>
    public class MSURChunk : IIFFChunk, IBinarySerializable, IDisposable
    {
        /// <summary>The chunk signature</summary>
        public const string Signature = "MSUR";

        private List<MsurEntry>? _entries;

        /// <summary>Gets the entries in this chunk with lazy initialization</summary>
        public List<MsurEntry> Entries => _entries ??= new List<MsurEntry>();

        /// <summary>Gets the entry count efficiently</summary>
        public int EntryCount => _entries?.Count ?? 0;

        /// <summary>Checks if the chunk has any entries</summary>
        public bool HasEntries => EntryCount > 0;

        /// <summary>Initializes a new instance of the MSURChunk class</summary>
        public MSURChunk() { }

        /// <summary>Initializes a new instance from binary data</summary>
        public MSURChunk(byte[] inData)
        {
            LoadBinaryData(inData);
        }

        #region Efficient Access Methods

        /// <summary>
        /// Gets surfaces with valid geometry efficiently.
        /// </summary>
        /// <returns>Surfaces that have valid triangle data</returns>
        public IEnumerable<MsurEntry> GetValidSurfaces()
        {
            if (_entries == null) yield break;
            
            for (int i = 0; i < _entries.Count; i++)
            {
                var entry = _entries[i];
                if (entry.HasValidGeometry)
                    yield return entry;
            }
        }

        /// <summary>
        /// Finds surfaces by height range efficiently.
        /// </summary>
        /// <param name="minHeight">Minimum height</param>
        /// <param name="maxHeight">Maximum height</param>
        /// <returns>Surfaces within the height range</returns>
        public IEnumerable<(int index, MsurEntry entry)> GetSurfacesByHeight(float minHeight, float maxHeight)
        {
            if (_entries == null) yield break;
            
            for (int i = 0; i < _entries.Count; i++)
            {
                var entry = _entries[i];
                if (entry.SurfaceHeight >= minHeight && entry.SurfaceHeight <= maxHeight)
                    yield return (i, entry);
            }
        }

        /// <summary>
        /// Detects duplicate surfaces using signature comparison.
        /// </summary>
        /// <returns>Groups of duplicate surface indices</returns>
        public IEnumerable<int[]> DetectDuplicateSurfaces()
        {
            if (_entries == null || _entries.Count < 2) yield break;

            var signatureGroups = new Dictionary<string, List<int>>();
            
            for (int i = 0; i < _entries.Count; i++)
            {
                var signature = _entries[i].CreateSignature();
                if (!signatureGroups.ContainsKey(signature))
                    signatureGroups[signature] = new List<int>();
                signatureGroups[signature].Add(i);
            }

            foreach (var group in signatureGroups.Values)
            {
                if (group.Count > 1)
                    yield return group.ToArray();
            }
        }

        /// <summary>
        /// Validates surface normals across all entries.
        /// </summary>
        /// <returns>Validation statistics</returns>
        public SurfaceValidationStats ValidateNormals()
        {
            var stats = new SurfaceValidationStats();
            
            if (_entries == null) return stats;
            
            for (int i = 0; i < _entries.Count; i++)
            {
                var entry = _entries[i];
                stats.TotalSurfaces++;
                
                if (entry.IsNormalValid)
                    stats.ValidNormals++;
                else
                    stats.InvalidNormals++;
                    
                if (entry.HasValidGeometry)
                    stats.ValidGeometry++;
            }
            
            return stats;
        }

        /// <summary>
        /// Pre-allocates capacity for known entry count.
        /// </summary>
        /// <param name="entryCount">Expected number of entries</param>
        public void PreAllocate(int entryCount)
        {
            if (entryCount > 0)
            {
                _entries ??= new List<MsurEntry>(entryCount);
                if (_entries.Capacity < entryCount)
                    _entries.Capacity = entryCount;
            }
        }

        /// <summary>
        /// Validates that all index ranges defined by the entries are within the bounds 
        /// of the provided MSVI index count.
        /// </summary>
        /// <param name="msviIndexCount">The total number of indices available in the corresponding MSVI chunk.</param>
        /// <returns>True if all ranges are valid, false otherwise.</returns>
        public bool ValidateIndices(int msviIndexCount)
        {
            if (_entries == null || msviIndexCount <= 0) return EntryCount == 0;
            for (int i = 0; i < _entries.Count; i++)
            {
                var entry = _entries[i];
                if (entry.MsviFirstIndex >= msviIndexCount)
                    return false;
                if ((long)entry.MsviFirstIndex + entry.IndexCount > msviIndexCount)
                    return false;
            }
            return true;
        }

        #endregion

        #region IBinarySerializable Implementation

        /// <inheritdoc/>
        public void LoadBinaryData(byte[] inData)
        {
            if (inData == null)
                throw new ArgumentNullException(nameof(inData));

            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        /// <inheritdoc/>
        public void Load(BinaryReader br)
        {
            var chunkSize = br.BaseStream.Length - br.BaseStream.Position;
            if (chunkSize > 0 && MsurEntry.BaseStructSize > 0)
            {
                var numEntries = (int)(chunkSize / MsurEntry.BaseStructSize);
                PreAllocate(numEntries);
                for (var i = 0; i < numEntries; i++)
                {
                    var entry = new MsurEntry();
                    entry.Load(br);
                    Entries.Add(entry);
                }
            }
        }

        /// <inheritdoc/>
        public byte[] Serialize(long offset = 0)
        {
            if (_entries == null || _entries.Count == 0)
                return Array.Empty<byte>();

            var totalSize = _entries.Count * MsurEntry.BaseStructSize;
            using var ms = new MemoryStream(totalSize);
            using var bw = new BinaryWriter(ms);
            
            foreach (var entry in _entries)
            {
                entry.Write(bw);
            }
            
            return ms.ToArray();
        }

        /// <inheritdoc/>
        public string GetSignature() => Signature;

        /// <inheritdoc/>
        public uint GetSize()
        {
            return (uint)(EntryCount * MsurEntry.BaseStructSize);
        }

        #endregion

        #region IDisposable

        /// <summary>Disposes resources and clears entries</summary>
        public void Dispose()
        {
            _entries?.Clear();
            _entries = null;
            GC.SuppressFinalize(this);
        }

        #endregion

        public override string ToString()
        {
            return $"MSUR Chunk [{EntryCount} surfaces, {GetSize()} bytes]";
        }
    }

    /// <summary>
    /// Statistics for surface validation.
    /// </summary>
    public struct SurfaceValidationStats
    {
        public int TotalSurfaces;
        public int ValidNormals;
        public int InvalidNormals;
        public int ValidGeometry;
        
        public float NormalValidationRate => TotalSurfaces > 0 ? (float)ValidNormals / TotalSurfaces : 0f;
        public float GeometryValidationRate => TotalSurfaces > 0 ? (float)ValidGeometry / TotalSurfaces : 0f;
        public bool AllNormalsValid => InvalidNormals == 0 && TotalSurfaces > 0;
    }
} 