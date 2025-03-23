using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Types;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MOLS chunk - Map Object Spot Lights
    /// Contains spot light definitions.
    /// </summary>
    public class MOLS : IChunk
    {
        /// <summary>
        /// Gets the list of spot lights.
        /// </summary>
        public List<SpotLight> SpotLights { get; } = new();

        public class SpotLight
        {
            /// <summary>
            /// Gets or sets the light color.
            /// </summary>
            public CImVector Color { get; set; }

            /// <summary>
            /// Gets or sets the position vector (X, Y, Z).
            /// </summary>
            public C3Vector Position { get; set; }

            /// <summary>
            /// Gets or sets the intensity of the light.
            /// </summary>
            public float Intensity { get; set; }

            /// <summary>
            /// Gets or sets the start range of the light.
            /// </summary>
            public float StartRange { get; set; }

            /// <summary>
            /// Gets or sets the end range of the light.
            /// </summary>
            public float EndRange { get; set; }

            /// <summary>
            /// Gets or sets the light info.
            /// </summary>
            public uint Info { get; set; }

            /// <summary>
            /// Gets or sets whether the light is used.
            /// </summary>
            public bool IsUsed { get; set; }

            /// <summary>
            /// Gets or sets the direction vector (X, Y, Z).
            /// </summary>
            public C3Vector Direction { get; set; }

            /// <summary>
            /// Gets or sets the inner radius of the spot light.
            /// </summary>
            public float InnerRadius { get; set; }

            /// <summary>
            /// Gets or sets the outer radius of the spot light.
            /// </summary>
            public float OuterRadius { get; set; }

            /// <summary>
            /// Gets or sets the attenuation start.
            /// </summary>
            public float AttenuationStart { get; set; }

            /// <summary>
            /// Gets or sets the attenuation end.
            /// </summary>
            public float AttenuationEnd { get; set; }
        }

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each spot light is 56 bytes:
            // 4 bytes for color
            // 12 bytes for position (3 floats)
            // 4 bytes for intensity
            // 4 bytes for start range
            // 4 bytes for end range
            // 4 bytes for info
            // 1 byte for isUsed
            // 12 bytes for direction (3 floats)
            // 4 bytes for inner radius
            // 4 bytes for outer radius
            // 4 bytes for attenuation start
            // 4 bytes for attenuation end
            var lightCount = (int)size / 56;

            // Clear existing data
            SpotLights.Clear();

            // Read spot lights
            for (int i = 0; i < lightCount; i++)
            {
                var light = new SpotLight
                {
                    Color = new CImVector { Value = reader.ReadUInt32() },
                    Position = new C3Vector
                    {
                        X = reader.ReadSingle(),
                        Y = reader.ReadSingle(),
                        Z = reader.ReadSingle()
                    },
                    Intensity = reader.ReadSingle(),
                    StartRange = reader.ReadSingle(),
                    EndRange = reader.ReadSingle(),
                    Info = reader.ReadUInt32(),
                    IsUsed = reader.ReadBoolean(),
                    Direction = new C3Vector
                    {
                        X = reader.ReadSingle(),
                        Y = reader.ReadSingle(),
                        Z = reader.ReadSingle()
                    },
                    InnerRadius = reader.ReadSingle(),
                    OuterRadius = reader.ReadSingle(),
                    AttenuationStart = reader.ReadSingle(),
                    AttenuationEnd = reader.ReadSingle()
                };

                SpotLights.Add(light);
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var light in SpotLights)
            {
                writer.Write(light.Color.Value);
                writer.Write(light.Position.X);
                writer.Write(light.Position.Y);
                writer.Write(light.Position.Z);
                writer.Write(light.Intensity);
                writer.Write(light.StartRange);
                writer.Write(light.EndRange);
                writer.Write(light.Info);
                writer.Write(light.IsUsed);
                writer.Write(light.Direction.X);
                writer.Write(light.Direction.Y);
                writer.Write(light.Direction.Z);
                writer.Write(light.InnerRadius);
                writer.Write(light.OuterRadius);
                writer.Write(light.AttenuationStart);
                writer.Write(light.AttenuationEnd);
            }
        }

        /// <summary>
        /// Gets a validation report for the spot lights.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MOLS Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Spot Lights: {SpotLights.Count}");
            report.AppendLine();

            // Track statistics
            var usedLights = 0;
            var minIntensity = float.MaxValue;
            var maxIntensity = float.MinValue;
            var minRange = float.MaxValue;
            var maxRange = float.MinValue;
            var minRadius = float.MaxValue;
            var maxRadius = float.MinValue;

            foreach (var light in SpotLights)
            {
                if (light.IsUsed) usedLights++;
                minIntensity = Math.Min(minIntensity, light.Intensity);
                maxIntensity = Math.Max(maxIntensity, light.Intensity);
                minRange = Math.Min(minRange, light.StartRange);
                maxRange = Math.Max(maxRange, light.EndRange);
                minRadius = Math.Min(minRadius, light.InnerRadius);
                maxRadius = Math.Max(maxRadius, light.OuterRadius);
            }

            // Report statistics
            report.AppendLine("Light Statistics:");
            report.AppendLine($"  Used Lights: {usedLights} of {SpotLights.Count}");
            report.AppendLine($"  Intensity Range: {minIntensity:F2} to {maxIntensity:F2}");
            report.AppendLine($"  Distance Range: {minRange:F2} to {maxRange:F2}");
            report.AppendLine($"  Radius Range: {minRadius:F2} to {maxRadius:F2}");

            // Analyze directions
            report.AppendLine();
            report.AppendLine("Direction Analysis:");
            for (int i = 0; i < SpotLights.Count; i++)
            {
                var light = SpotLights[i];
                var dirLength = Math.Sqrt(
                    light.Direction.X * light.Direction.X +
                    light.Direction.Y * light.Direction.Y +
                    light.Direction.Z * light.Direction.Z
                );

                if (Math.Abs(dirLength - 1.0) > 0.001)
                {
                    report.AppendLine($"  Light {i} direction vector length: {dirLength:F3} (not normalized)");
                }
            }

            return report.ToString();
        }
    }
} 