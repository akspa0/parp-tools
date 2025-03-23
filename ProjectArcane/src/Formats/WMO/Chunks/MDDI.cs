using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MDDI chunk - Map Object Doodad Additional Info
    /// Contains additional information for doodads, added in 8.3.0.32044
    /// </summary>
    public class MDDI : IChunk
    {
        /// <summary>
        /// Gets the list of color multipliers for doodads.
        /// </summary>
        public List<float> ColorMultipliers { get; } = new();

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each color multiplier is 4 bytes (float)
            var count = (int)size / 4;

            // Clear existing data
            ColorMultipliers.Clear();

            // Read color multipliers
            for (int i = 0; i < count; i++)
            {
                ColorMultipliers.Add(reader.ReadSingle());
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var multiplier in ColorMultipliers)
            {
                writer.Write(multiplier);
            }
        }

        /// <summary>
        /// Gets a color multiplier for a specific doodad.
        /// </summary>
        /// <param name="doodadIndex">Index of the doodad.</param>
        /// <returns>Color multiplier if found, 1.0f otherwise.</returns>
        public float GetColorMultiplier(int doodadIndex)
        {
            if (doodadIndex < 0 || doodadIndex >= ColorMultipliers.Count)
                return 1.0f;

            return ColorMultipliers[doodadIndex];
        }

        /// <summary>
        /// Validates doodad count against MOHD.
        /// </summary>
        /// <param name="mohdDoodadCount">Number of doodads specified in MOHD chunk.</param>
        /// <returns>True if counts match, false otherwise.</returns>
        public bool ValidateDoodadCount(int mohdDoodadCount)
        {
            return ColorMultipliers.Count == mohdDoodadCount;
        }

        /// <summary>
        /// Gets a validation report for the doodad color multipliers.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MDDI Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Color Multipliers: {ColorMultipliers.Count}");
            report.AppendLine();

            // Analyze multiplier values
            var nonDefaultMultipliers = 0;
            var minMultiplier = float.MaxValue;
            var maxMultiplier = float.MinValue;

            foreach (var multiplier in ColorMultipliers)
            {
                if (multiplier != 1.0f)
                {
                    nonDefaultMultipliers++;
                    minMultiplier = System.Math.Min(minMultiplier, multiplier);
                    maxMultiplier = System.Math.Max(maxMultiplier, multiplier);
                }
            }

            report.AppendLine($"Modified Doodads: {nonDefaultMultipliers}");
            report.AppendLine($"Percentage Modified: {(nonDefaultMultipliers * 100.0f / ColorMultipliers.Count):F1}%");

            if (nonDefaultMultipliers > 0)
            {
                report.AppendLine();
                report.AppendLine("Multiplier Range:");
                report.AppendLine($"  Min: {minMultiplier:F3}");
                report.AppendLine($"  Max: {maxMultiplier:F3}");
            }

            return report.ToString();
        }
    }
} 