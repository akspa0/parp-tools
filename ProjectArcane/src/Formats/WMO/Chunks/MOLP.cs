using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Types;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MOLP chunk - Map Object Light Points
    /// Contains light point definitions.
    /// </summary>
    public class MOLP : IChunk
    {
        /// <summary>
        /// Gets the list of light points.
        /// </summary>
        public List<LightPoint> LightPoints { get; } = new();

        public class LightPoint
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
            /// Gets or sets the attenuation start.
            /// </summary>
            public float AttenuationStart { get; set; }

            /// <summary>
            /// Gets or sets the attenuation end.
            /// </summary>
            public float AttenuationEnd { get; set; }

            /// <summary>
            /// Gets or sets whether the light is used.
            /// </summary>
            public bool IsUsed { get; set; }

            /// <summary>
            /// Gets or sets the light info.
            /// </summary>
            public uint Info { get; set; }
        }

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each light point is 28 bytes:
            // 4 bytes for color
            // 12 bytes for position (3 floats)
            // 4 bytes for intensity
            // 4 bytes for attenuation start
            // 4 bytes for attenuation end
            // 1 byte for isUsed
            // 4 bytes for info
            var lightCount = (int)size / 28;

            // Clear existing data
            LightPoints.Clear();

            // Read light points
            for (int i = 0; i < lightCount; i++)
            {
                var light = new LightPoint
                {
                    Color = new CImVector { Value = reader.ReadUInt32() },
                    Position = new C3Vector
                    {
                        X = reader.ReadSingle(),
                        Y = reader.ReadSingle(),
                        Z = reader.ReadSingle()
                    },
                    Intensity = reader.ReadSingle(),
                    AttenuationStart = reader.ReadSingle(),
                    AttenuationEnd = reader.ReadSingle(),
                    IsUsed = reader.ReadBoolean(),
                    Info = reader.ReadUInt32()
                };

                LightPoints.Add(light);
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var light in LightPoints)
            {
                writer.Write(light.Color.Value);
                writer.Write(light.Position.X);
                writer.Write(light.Position.Y);
                writer.Write(light.Position.Z);
                writer.Write(light.Intensity);
                writer.Write(light.AttenuationStart);
                writer.Write(light.AttenuationEnd);
                writer.Write(light.IsUsed);
                writer.Write(light.Info);
            }
        }

        /// <summary>
        /// Gets a validation report for the light points.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MOLP Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Light Points: {LightPoints.Count}");
            report.AppendLine();

            // Track statistics
            var usedLights = 0;
            var minIntensity = float.MaxValue;
            var maxIntensity = float.MinValue;
            var minAttenuation = float.MaxValue;
            var maxAttenuation = float.MinValue;

            foreach (var light in LightPoints)
            {
                if (light.IsUsed) usedLights++;
                minIntensity = Math.Min(minIntensity, light.Intensity);
                maxIntensity = Math.Max(maxIntensity, light.Intensity);
                minAttenuation = Math.Min(minAttenuation, light.AttenuationStart);
                maxAttenuation = Math.Max(maxAttenuation, light.AttenuationEnd);
            }

            // Report statistics
            report.AppendLine("Light Statistics:");
            report.AppendLine($"  Used Lights: {usedLights} of {LightPoints.Count}");
            report.AppendLine($"  Intensity Range: {minIntensity:F2} to {maxIntensity:F2}");
            report.AppendLine($"  Attenuation Range: {minAttenuation:F2} to {maxAttenuation:F2}");

            // Analyze positions
            var positions = LightPoints.Select(l => l.Position).ToList();
            if (positions.Any())
            {
                var minX = positions.Min(p => p.X);
                var maxX = positions.Max(p => p.X);
                var minY = positions.Min(p => p.Y);
                var maxY = positions.Max(p => p.Y);
                var minZ = positions.Min(p => p.Z);
                var maxZ = positions.Max(p => p.Z);

                report.AppendLine();
                report.AppendLine("Position Bounds:");
                report.AppendLine($"  X: {minX:F2} to {maxX:F2}");
                report.AppendLine($"  Y: {minY:F2} to {maxY:F2}");
                report.AppendLine($"  Z: {minZ:F2} to {maxZ:F2}");
            }

            return report.ToString();
        }
    }
} 