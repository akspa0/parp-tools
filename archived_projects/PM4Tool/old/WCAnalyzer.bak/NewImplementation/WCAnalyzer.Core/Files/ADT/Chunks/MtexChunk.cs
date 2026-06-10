using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using WCAnalyzer.Core.Common.Interfaces;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MTEX chunk - Contains a list of all texture filenames referenced in the ADT (pre-8.1.0.28294)
    /// Replaced by MDID and MHID chunks in newer versions
    /// </summary>
    public class MtexChunk : ADTChunk
    {
        /// <summary>
        /// The MTEX chunk signature
        /// </summary>
        public const string SIGNATURE = "MTEX";

        /// <summary>
        /// Gets the list of texture filenames referenced in this ADT
        /// </summary>
        public List<string> Textures { get; } = new List<string>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MtexChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MtexChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
        }

        /// <summary>
        /// Parses the texture filenames from the chunk data
        /// </summary>
        public override void Parse()
        {
            if (Data == null || Data.Length == 0)
            {
                AddError("No data to parse for MTEX chunk");
                return;
            }

            try
            {
                // The MTEX chunk contains a sequence of null-terminated strings
                Textures.Clear();
                var currentString = new List<byte>();
                
                for (int i = 0; i < Data.Length; i++)
                {
                    byte b = Data[i];
                    
                    if (b == 0)
                    {
                        // Found null terminator, add the string to the list
                        if (currentString.Count > 0)
                        {
                            string texture = Encoding.ASCII.GetString(currentString.ToArray());
                            Textures.Add(texture);
                            Logger?.LogDebug($"MTEX: Found texture: {texture}");
                            currentString.Clear();
                        }
                    }
                    else
                    {
                        currentString.Add(b);
                    }
                }
                
                // Add the last string if it wasn't null-terminated
                if (currentString.Count > 0)
                {
                    string texture = Encoding.ASCII.GetString(currentString.ToArray());
                    Textures.Add(texture);
                    Logger?.LogDebug($"MTEX: Found texture: {texture}");
                }
                
                Logger?.LogDebug($"MTEX: Parsed {Textures.Count} texture filenames");
            }
            catch (Exception ex)
            {
                AddError($"Error parsing MTEX chunk: {ex.Message}");
            }
        }

        /// <summary>
        /// Gets a texture filename by its index
        /// </summary>
        /// <param name="index">The index of the texture in the MTEX chunk</param>
        /// <returns>The texture filename at the specified index, or null if out of range</returns>
        public string? GetTexture(int index)
        {
            if (index < 0 || index >= Textures.Count)
            {
                AddError($"Texture index {index} is out of range (0-{Textures.Count - 1})");
                return null;
            }
            
            return Textures[index];
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
                // Write each texture as a null-terminated string
                foreach (var texture in Textures)
                {
                    byte[] textureBytes = Encoding.ASCII.GetBytes(texture);
                    writer.Write(textureBytes);
                    writer.Write((byte)0); // Null terminator
                }
            }
            catch (Exception ex)
            {
                AddError($"Error writing MTEX chunk: {ex.Message}");
            }
        }
    }
} 