using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// Represents an MCRF chunk in an ADT file, containing references to doodad (M2 model) and WMO placements.
    /// </summary>
    public class McrfChunk : ADTChunk
    {
        /// <summary>
        /// The MCRF chunk signature
        /// </summary>
        public const string SIGNATURE = "MCRF";
        
        /// <summary>
        /// Gets the doodad (M2 model) references. These are indices into the MMID chunk.
        /// </summary>
        public List<uint> ModelReferences { get; private set; } = new List<uint>();
        
        /// <summary>
        /// Gets the WMO references. These are indices into the MWID chunk.
        /// </summary>
        public List<uint> WorldObjectReferences { get; private set; } = new List<uint>();
        
        /// <summary>
        /// Gets the raw reference data as bytes.
        /// </summary>
        public byte[] RawReferenceData { get; private set; } = Array.Empty<byte>();
        
        /// <summary>
        /// The count of model references expected in the chunk.
        /// </summary>
        private readonly uint _modelReferenceCount;
        
        /// <summary>
        /// The count of world object references expected in the chunk.
        /// </summary>
        private readonly uint _worldObjectReferenceCount;

        /// <summary>
        /// Initializes a new instance of the <see cref="McrfChunk"/> class with model reference count only.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        /// <param name="modelReferenceCount">The expected number of model references.</param>
        /// <param name="logger">Optional logger.</param>
        public McrfChunk(byte[] data, uint modelReferenceCount, ILogger? logger = null)
            : this(data, modelReferenceCount, 0, logger)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="McrfChunk"/> class with both model and world object reference counts.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        /// <param name="modelReferenceCount">The expected number of model references.</param>
        /// <param name="worldObjectReferenceCount">The expected number of world object references.</param>
        /// <param name="logger">Optional logger.</param>
        public McrfChunk(byte[] data, uint modelReferenceCount, uint worldObjectReferenceCount, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
            _modelReferenceCount = modelReferenceCount;
            _worldObjectReferenceCount = worldObjectReferenceCount;
        }

        /// <summary>
        /// Parses the chunk data
        /// </summary>
        public override void Parse()
        {
            if (Data == null || Data.Length == 0)
            {
                AddError("No data to parse for MCRF chunk");
                return;
            }
            
            try
            {
                using (var ms = new MemoryStream(Data))
                using (var reader = new BinaryReader(ms))
                {
                    // Store raw data
                    RawReferenceData = Data;
                    ModelReferences.Clear();
                    WorldObjectReferences.Clear();
                    
                    // Read model references
                    for (int i = 0; i < _modelReferenceCount && ms.Position < ms.Length; i++)
                    {
                        ModelReferences.Add(reader.ReadUInt32());
                    }
                    
                    // Read world object references if any
                    for (int i = 0; i < _worldObjectReferenceCount && ms.Position < ms.Length; i++)
                    {
                        WorldObjectReferences.Add(reader.ReadUInt32());
                    }
                    
                    Logger?.LogDebug($"MCRF: Parsed {ModelReferences.Count} model references and {WorldObjectReferences.Count} world object references");
                }
            }
            catch (Exception ex)
            {
                AddError($"Failed to parse MCRF chunk: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Writes the chunk data to the specified writer
        /// </summary>
        /// <param name="writer">The binary writer to write to</param>
        public override void Write(BinaryWriter writer)
        {
            if (writer == null)
            {
                AddError("Cannot write to null writer");
                return;
            }
            
            try
            {
                // Write model references
                foreach (var reference in ModelReferences)
                {
                    writer.Write(reference);
                }
                
                // Write world object references
                foreach (var reference in WorldObjectReferences)
                {
                    writer.Write(reference);
                }
                
                Logger?.LogDebug($"MCRF: Wrote {ModelReferences.Count} model references and {WorldObjectReferences.Count} world object references");
            }
            catch (Exception ex)
            {
                AddError($"Error writing MCRF chunk: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Gets a model reference index at the specified position.
        /// </summary>
        /// <param name="index">The index of the reference to retrieve.</param>
        /// <returns>The model reference index or 0 if the index is out of range.</returns>
        public uint GetModelReference(int index)
        {
            if (index < 0 || index >= ModelReferences.Count)
            {
                AddError($"Model reference index must be between 0 and {ModelReferences.Count - 1}, got {index}");
                return 0;
            }
            
            return ModelReferences[index];
        }
        
        /// <summary>
        /// Gets a world object reference index at the specified position.
        /// </summary>
        /// <param name="index">The index of the reference to retrieve.</param>
        /// <returns>The world object reference index or 0 if the index is out of range.</returns>
        public uint GetWorldObjectReference(int index)
        {
            if (index < 0 || index >= WorldObjectReferences.Count)
            {
                AddError($"World object reference index must be between 0 and {WorldObjectReferences.Count - 1}, got {index}");
                return 0;
            }
            
            return WorldObjectReferences[index];
        }
        
        /// <summary>
        /// Gets the number of model references in this chunk
        /// </summary>
        public int ModelReferenceCount => ModelReferences.Count;
        
        /// <summary>
        /// Gets the number of world object references in this chunk
        /// </summary>
        public int WorldObjectReferenceCount => WorldObjectReferences.Count;
    }
} 