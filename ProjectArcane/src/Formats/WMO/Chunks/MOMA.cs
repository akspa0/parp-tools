using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MOMA chunk - Material attributes for v14 (Alpha) WMO
    /// Contains additional material properties not present in MOMT
    /// </summary>
    public class MOMA : IChunk
    {
        /// <summary>
        /// Size of each material attribute entry in bytes
        /// </summary>
        private const int MATERIAL_ATTR_SIZE = 16;

        /// <summary>
        /// Gets the list of material attributes.
        /// </summary>
        public List<MaterialAttributes> MaterialAttributesList { get; } = new();

        public class MaterialAttributes
        {
            /// <summary>
            /// Gets or sets the flags for this material.
            /// </summary>
            public uint Flags { get; set; }

            /// <summary>
            /// Gets or sets the shader index for this material.
            /// </summary>
            public uint ShaderIndex { get; set; }

            /// <summary>
            /// Gets or sets the blend mode for this material.
            /// </summary>
            public uint BlendMode { get; set; }

            /// <summary>
            /// Gets or sets additional material data.
            /// </summary>
            public uint MaterialData { get; set; }
        }

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Calculate number of material attributes
            var materialCount = (int)size / MATERIAL_ATTR_SIZE;

            // Clear existing data
            MaterialAttributesList.Clear();

            // Read material attributes
            for (int i = 0; i < materialCount; i++)
            {
                var attr = new MaterialAttributes
                {
                    Flags = reader.ReadUInt32(),
                    ShaderIndex = reader.ReadUInt32(),
                    BlendMode = reader.ReadUInt32(),
                    MaterialData = reader.ReadUInt32()
                };
                MaterialAttributesList.Add(attr);
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var attr in MaterialAttributesList)
            {
                writer.Write(attr.Flags);
                writer.Write(attr.ShaderIndex);
                writer.Write(attr.BlendMode);
                writer.Write(attr.MaterialData);
            }
        }

        /// <summary>
        /// Gets material attributes for a specific material index.
        /// </summary>
        /// <param name="materialIndex">Index of the material.</param>
        /// <returns>Material attributes if found, null otherwise.</returns>
        public MaterialAttributes GetMaterialAttributes(int materialIndex)
        {
            if (materialIndex < 0 || materialIndex >= MaterialAttributesList.Count)
                return null;

            return MaterialAttributesList[materialIndex];
        }

        /// <summary>
        /// Validates material attributes against MOMT chunk.
        /// </summary>
        /// <param name="momtMaterialCount">Number of materials in MOMT chunk.</param>
        /// <returns>True if material counts match, false otherwise.</returns>
        public bool ValidateMaterialCount(int momtMaterialCount)
        {
            return MaterialAttributesList.Count == momtMaterialCount;
        }

        /// <summary>
        /// Gets a validation report for the material attributes.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MOMA Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Materials: {MaterialAttributesList.Count}");
            report.AppendLine();

            // Analyze shader indices
            var shaderCounts = new Dictionary<uint, int>();
            foreach (var attr in MaterialAttributesList)
            {
                if (!shaderCounts.ContainsKey(attr.ShaderIndex))
                    shaderCounts[attr.ShaderIndex] = 0;
                shaderCounts[attr.ShaderIndex]++;
            }

            report.AppendLine("Shader Usage:");
            foreach (var kvp in shaderCounts)
            {
                report.AppendLine($"  Shader {kvp.Key}: {kvp.Value} materials");
            }

            // Analyze blend modes
            var blendCounts = new Dictionary<uint, int>();
            foreach (var attr in MaterialAttributesList)
            {
                if (!blendCounts.ContainsKey(attr.BlendMode))
                    blendCounts[attr.BlendMode] = 0;
                blendCounts[attr.BlendMode]++;
            }

            report.AppendLine();
            report.AppendLine("Blend Mode Usage:");
            foreach (var kvp in blendCounts)
            {
                report.AppendLine($"  Mode {kvp.Key}: {kvp.Value} materials");
            }

            // Flag analysis
            report.AppendLine();
            report.AppendLine("Flag Analysis:");
            var uniqueFlags = new HashSet<uint>();
            foreach (var attr in MaterialAttributesList)
            {
                uniqueFlags.Add(attr.Flags);
            }
            report.AppendLine($"  Unique Flag Combinations: {uniqueFlags.Count}");
            foreach (var flag in uniqueFlags)
            {
                var count = MaterialAttributesList.Count(m => m.Flags == flag);
                report.AppendLine($"  Flag 0x{flag:X8}: {count} materials");
            }

            return report.ToString();
        }
    }
} 