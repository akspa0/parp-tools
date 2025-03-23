using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MNLD chunk - Map Object New Light Definitions
    /// Contains dynamic light definitions added in Shadowlands. Used for everything from torch fires 
    /// to projecting light/shadow on the ground to make it look like light is coming through windows.
    /// </summary>
    public class MNLD : IChunk
    {
        /// <summary>
        /// Gets the list of new light definitions.
        /// </summary>
        public List<NewLightDefinition> NewLightDefinitions { get; } = new();

        public class NewLightDefinition
        {
            /// <summary>
            /// Gets or sets the type of light (0 = Point light (sphere), 1 = Spot light (cone)).
            /// </summary>
            public int Type { get; set; }

            /// <summary>
            /// Gets or sets the light index (appears to be same as index in mapobject_new_light_defs[]).
            /// </summary>
            public int LightIndex { get; set; }

            /// <summary>
            /// Gets or sets whether to enable color gradient (0 = false (use only startColor), 1 = true (use start and end color gradient)).
            /// </summary>
            public int EnableColorGradient { get; set; }

            /// <summary>
            /// Gets or sets the doodad set this light belongs to.
            /// </summary>
            public int DoodadSet { get; set; }

            /// <summary>
            /// Gets or sets the start color.
            /// </summary>
            public CImVector StartColor { get; set; }

            /// <summary>
            /// Gets or sets the light position in WMO.
            /// </summary>
            public Vector3 Position { get; set; }

            /// <summary>
            /// Gets or sets the Euler rotation in radians. For spot light rotates the light, for point light rotates the light cookie.
            /// </summary>
            public Vector3 Rotation { get; set; }

            /// <summary>
            /// Gets or sets the start attenuation distance.
            /// </summary>
            public float AttenuationStart { get; set; }

            /// <summary>
            /// Gets or sets the end attenuation distance.
            /// </summary>
            public float AttenuationEnd { get; set; }

            /// <summary>
            /// Gets or sets the light intensity.
            /// </summary>
            public float Intensity { get; set; }

            /// <summary>
            /// Gets or sets the end color for gradient.
            /// </summary>
            public CImVector EndColor { get; set; }

            /// <summary>
            /// Gets or sets the gradient start distance from emitter position, for mixing start and end color.
            /// </summary>
            public float ColorBlendStart { get; set; }

            /// <summary>
            /// Gets or sets the gradient end distance from emitter position, for mixing start and end color.
            /// </summary>
            public float ColorBlendEnd { get; set; }

            /// <summary>
            /// Gets or sets the flickering light intensity.
            /// </summary>
            public float FlickerIntensity { get; set; }

            /// <summary>
            /// Gets or sets the flickering light speed.
            /// </summary>
            public float FlickerSpeed { get; set; }

            /// <summary>
            /// Gets or sets the flicker mode (0 = off, 1 = sine curve, 2 = noise curve, 3 = noise step curve).
            /// </summary>
            public int FlickerMode { get; set; }

            /// <summary>
            /// Gets or sets field_54 (only found 0's so far).
            /// </summary>
            public Vector3 Field54 { get; set; }

            /// <summary>
            /// Gets or sets the file ID for light cookie texture. For point light it's a cube map.
            /// </summary>
            public uint LightCookieFileId { get; set; }

            /// <summary>
            /// Gets or sets the overall radius of the spot light, in radians.
            /// </summary>
            public float SpotlightRadius { get; set; }

            /// <summary>
            /// Gets or sets the start of drop-off gradient, in radians. Starts at center, ends at edge.
            /// Controls the rate at which light intensity decreases from the center to the edge of the spot light beam.
            /// </summary>
            public float SpotlightDropoffStart { get; set; }

            /// <summary>
            /// Gets or sets the end of drop-off gradient, in radians.
            /// Both start and end drop-off angles have to be smaller than radius else sharp edge.
            /// </summary>
            public float SpotlightDropoffEnd { get; set; }

            /// <summary>
            /// Gets or sets unknown value (14336 - power of 2).
            /// </summary>
            public uint Unknown0 { get; set; }

            /// <summary>
            /// Gets or sets field_50 (only found 0's so far).
            /// </summary>
            public byte Field50 { get; set; }

            /// <summary>
            /// Gets or sets unknown values (only found 0's so far).
            /// </summary>
            public byte[] Unknown1 { get; set; } = new byte[2];
        }

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each light definition is 128 bytes
            var lightCount = (int)size / 128;

            // Clear existing data
            NewLightDefinitions.Clear();

            // Read light definitions
            for (int i = 0; i < lightCount; i++)
            {
                var light = new NewLightDefinition
                {
                    Type = reader.ReadInt32(),
                    LightIndex = reader.ReadInt32(),
                    EnableColorGradient = reader.ReadInt32(),
                    DoodadSet = reader.ReadInt32(),
                    StartColor = new CImVector
                    {
                        Value = reader.ReadUInt32()
                    },
                    Position = new Vector3
                    {
                        X = reader.ReadSingle(),
                        Y = reader.ReadSingle(),
                        Z = reader.ReadSingle()
                    },
                    Rotation = new Vector3
                    {
                        X = reader.ReadSingle(),
                        Y = reader.ReadSingle(),
                        Z = reader.ReadSingle()
                    },
                    AttenuationStart = reader.ReadSingle(),
                    AttenuationEnd = reader.ReadSingle(),
                    Intensity = reader.ReadSingle(),
                    EndColor = new CImVector
                    {
                        Value = reader.ReadUInt32()
                    },
                    ColorBlendStart = reader.ReadSingle(),
                    ColorBlendEnd = reader.ReadSingle(),
                    // Skip 4 byte gap
                    _ = reader.ReadInt32(),
                    FlickerIntensity = reader.ReadSingle(),
                    FlickerSpeed = reader.ReadSingle(),
                    FlickerMode = reader.ReadInt32(),
                    Field54 = new Vector3
                    {
                        X = reader.ReadSingle(),
                        Y = reader.ReadSingle(),
                        Z = reader.ReadSingle()
                    },
                    // Skip 4 byte gap
                    _ = reader.ReadInt32(),
                    LightCookieFileId = reader.ReadUInt32(),
                    // Skip 20 byte gap
                    _ = reader.ReadBytes(20),
                    SpotlightRadius = reader.ReadSingle(),
                    SpotlightDropoffStart = reader.ReadSingle(),
                    SpotlightDropoffEnd = reader.ReadSingle(),
                    Unknown0 = reader.ReadUInt32(),
                    // Skip 41 byte gap
                    _ = reader.ReadBytes(41),
                    Field50 = reader.ReadByte(),
                    Unknown1 = reader.ReadBytes(2)
                };

                NewLightDefinitions.Add(light);
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var light in NewLightDefinitions)
            {
                writer.Write(light.Type);
                writer.Write(light.LightIndex);
                writer.Write(light.EnableColorGradient);
                writer.Write(light.DoodadSet);
                writer.Write(light.StartColor.Value);
                writer.Write(light.Position.X);
                writer.Write(light.Position.Y);
                writer.Write(light.Position.Z);
                writer.Write(light.Rotation.X);
                writer.Write(light.Rotation.Y);
                writer.Write(light.Rotation.Z);
                writer.Write(light.AttenuationStart);
                writer.Write(light.AttenuationEnd);
                writer.Write(light.Intensity);
                writer.Write(light.EndColor.Value);
                writer.Write(light.ColorBlendStart);
                writer.Write(light.ColorBlendEnd);
                writer.Write(0); // 4 byte gap
                writer.Write(light.FlickerIntensity);
                writer.Write(light.FlickerSpeed);
                writer.Write(light.FlickerMode);
                writer.Write(light.Field54.X);
                writer.Write(light.Field54.Y);
                writer.Write(light.Field54.Z);
                writer.Write(0); // 4 byte gap
                writer.Write(light.LightCookieFileId);
                writer.Write(new byte[20]); // 20 byte gap
                writer.Write(light.SpotlightRadius);
                writer.Write(light.SpotlightDropoffStart);
                writer.Write(light.SpotlightDropoffEnd);
                writer.Write(light.Unknown0);
                writer.Write(new byte[41]); // 41 byte gap
                writer.Write(light.Field50);
                writer.Write(light.Unknown1);
            }
        }

        /// <summary>
        /// Gets a validation report for the new light definitions.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MNLD Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total New Lights: {NewLightDefinitions.Count}");
            report.AppendLine();

            // Count light types
            var pointLights = 0;
            var spotLights = 0;
            var unknownTypes = new HashSet<int>();

            // Track light properties
            var minIntensity = float.MaxValue;
            var maxIntensity = float.MinValue;
            var minAttenStart = float.MaxValue;
            var maxAttenStart = float.MinValue;
            var minAttenEnd = float.MaxValue;
            var maxAttenEnd = float.MinValue;
            var uniqueDoodadSets = new HashSet<int>();
            var lightsWithGradient = 0;
            var lightsWithFlicker = 0;
            var uniqueFlickerModes = new HashSet<int>();
            var lightsWithCookies = 0;

            foreach (var light in NewLightDefinitions)
            {
                // Track light types
                switch (light.Type)
                {
                    case 0: pointLights++; break;
                    case 1: spotLights++; break;
                    default: unknownTypes.Add(light.Type); break;
                }

                // Track properties
                minIntensity = Math.Min(minIntensity, light.Intensity);
                maxIntensity = Math.Max(maxIntensity, light.Intensity);
                minAttenStart = Math.Min(minAttenStart, light.AttenuationStart);
                maxAttenStart = Math.Max(maxAttenStart, light.AttenuationStart);
                minAttenEnd = Math.Min(minAttenEnd, light.AttenuationEnd);
                maxAttenEnd = Math.Max(maxAttenEnd, light.AttenuationEnd);

                uniqueDoodadSets.Add(light.DoodadSet);
                if (light.EnableColorGradient != 0) lightsWithGradient++;
                if (light.FlickerMode != 0)
                {
                    lightsWithFlicker++;
                    uniqueFlickerModes.Add(light.FlickerMode);
                }
                if (light.LightCookieFileId != 0) lightsWithCookies++;
            }

            // Report light types
            report.AppendLine("Light Types:");
            report.AppendLine($"  Point Lights: {pointLights}");
            report.AppendLine($"  Spot Lights: {spotLights}");
            if (unknownTypes.Count > 0)
            {
                report.AppendLine($"  Unknown Types: {string.Join(", ", unknownTypes)}");
            }
            report.AppendLine();

            // Report properties
            report.AppendLine("Light Properties:");
            report.AppendLine($"  Unique Doodad Sets: {uniqueDoodadSets.Count}");
            report.AppendLine($"  Lights with Color Gradient: {lightsWithGradient}");
            report.AppendLine($"  Lights with Flicker: {lightsWithFlicker}");
            report.AppendLine($"  Unique Flicker Modes: {string.Join(", ", uniqueFlickerModes)}");
            report.AppendLine($"  Lights with Cookies: {lightsWithCookies}");
            report.AppendLine();

            // Report ranges
            report.AppendLine("Value Ranges:");
            report.AppendLine($"  Intensity: {minIntensity:F2} to {maxIntensity:F2}");
            report.AppendLine($"  Attenuation Start: {minAttenStart:F2} to {maxAttenStart:F2}");
            report.AppendLine($"  Attenuation End: {minAttenEnd:F2} to {maxAttenEnd:F2}");

            // Check for potential issues
            var invalidLights = NewLightDefinitions.Where(l =>
                l.AttenuationStart > l.AttenuationEnd ||
                l.Intensity < 0 ||
                l.AttenuationStart < 0 ||
                l.AttenuationEnd < 0 ||
                (l.Type == 1 && l.SpotlightDropoffEnd > l.SpotlightRadius) || // Spot light specific validation
                (l.Type == 1 && l.SpotlightDropoffStart > l.SpotlightDropoffEnd)).ToList();

            if (invalidLights.Any())
            {
                report.AppendLine();
                report.AppendLine("Invalid Lights:");
                foreach (var light in invalidLights)
                {
                    report.AppendLine($"  Light Index {light.LightIndex}:");
                    if (light.AttenuationStart > light.AttenuationEnd)
                        report.AppendLine($"    - Invalid attenuation range: {light.AttenuationStart:F2} > {light.AttenuationEnd:F2}");
                    if (light.Intensity < 0)
                        report.AppendLine($"    - Negative intensity: {light.Intensity:F2}");
                    if (light.AttenuationStart < 0)
                        report.AppendLine($"    - Negative attenuation start: {light.AttenuationStart:F2}");
                    if (light.AttenuationEnd < 0)
                        report.AppendLine($"    - Negative attenuation end: {light.AttenuationEnd:F2}");
                    if (light.Type == 1)
                    {
                        if (light.SpotlightDropoffEnd > light.SpotlightRadius)
                            report.AppendLine($"    - Invalid spotlight dropoff: end ({light.SpotlightDropoffEnd:F2}) > radius ({light.SpotlightRadius:F2})");
                        if (light.SpotlightDropoffStart > light.SpotlightDropoffEnd)
                            report.AppendLine($"    - Invalid spotlight dropoff range: start ({light.SpotlightDropoffStart:F2}) > end ({light.SpotlightDropoffEnd:F2})");
                    }
                }
            }

            return report.ToString();
        }
    }
} 