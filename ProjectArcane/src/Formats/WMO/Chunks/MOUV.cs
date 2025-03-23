using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Types;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MOUV chunk - Map Object UV Animation
    /// Contains UV animation data for materials, added in 7.3.0.24473
    /// </summary>
    public class MOUV : IChunk
    {
        /// <summary>
        /// Gets the list of UV animations.
        /// </summary>
        public List<UVAnimation> UVAnimations { get; } = new();

        public class UVAnimation
        {
            /// <summary>
            /// Gets or sets the translation speeds for two texture layers.
            /// </summary>
            public C2Vector[] TranslationSpeed { get; } = new C2Vector[2];
        }

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each UV animation is 16 bytes (2 C2Vectors)
            var animationCount = (int)size / 16;

            // Clear existing data
            UVAnimations.Clear();

            // Read UV animations
            for (int i = 0; i < animationCount; i++)
            {
                var animation = new UVAnimation();
                for (int j = 0; j < 2; j++)
                {
                    animation.TranslationSpeed[j] = new C2Vector
                    {
                        X = reader.ReadSingle(),
                        Y = reader.ReadSingle()
                    };
                }
                UVAnimations.Add(animation);
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var animation in UVAnimations)
            {
                foreach (var speed in animation.TranslationSpeed)
                {
                    writer.Write(speed.X);
                    writer.Write(speed.Y);
                }
            }
        }

        /// <summary>
        /// Gets UV animation data for a specific material.
        /// </summary>
        /// <param name="materialIndex">Index of the material.</param>
        /// <returns>UV animation if found, null otherwise.</returns>
        public UVAnimation GetUVAnimation(int materialIndex)
        {
            if (materialIndex < 0 || materialIndex >= UVAnimations.Count)
                return null;

            return UVAnimations[materialIndex];
        }

        /// <summary>
        /// Validates UV animation data against material count.
        /// </summary>
        /// <param name="materialCount">Number of materials in MOMT chunk.</param>
        /// <returns>True if UV animation count matches material count, false otherwise.</returns>
        public bool ValidateMaterialCount(int materialCount)
        {
            return UVAnimations.Count == materialCount;
        }

        /// <summary>
        /// Gets a validation report for the UV animations.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MOUV Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total UV Animations: {UVAnimations.Count}");
            report.AppendLine();

            // Analyze translation speeds
            var nonZeroAnimations = 0;
            for (int i = 0; i < UVAnimations.Count; i++)
            {
                var animation = UVAnimations[i];
                bool hasAnimation = false;
                for (int j = 0; j < 2; j++)
                {
                    if (animation.TranslationSpeed[j].X != 0 || animation.TranslationSpeed[j].Y != 0)
                    {
                        hasAnimation = true;
                        report.AppendLine($"Material {i}, Layer {j}:");
                        report.AppendLine($"  Translation Speed: ({animation.TranslationSpeed[j].X:F3}, {animation.TranslationSpeed[j].Y:F3})");
                    }
                }
                if (hasAnimation)
                    nonZeroAnimations++;
            }

            report.AppendLine();
            report.AppendLine($"Materials with UV Animation: {nonZeroAnimations}");
            report.AppendLine($"Percentage: {(nonZeroAnimations * 100.0f / UVAnimations.Count):F1}%");

            return report.ToString();
        }
    }
} 