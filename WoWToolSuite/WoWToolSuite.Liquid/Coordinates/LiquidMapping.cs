using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using WowToolSuite.Liquid.Models;

namespace WowToolSuite.Liquid.Coordinates
{
    [JsonConverter(typeof(AdtCoordinateConverter))]
    public class AdtCoordinate
    {
        public int X { get; set; }
        public int Y { get; set; }

        public AdtCoordinate(int x, int y)
        {
            X = x;
            Y = y;
        }

        public override bool Equals(object? obj)
        {
            if (obj is AdtCoordinate other)
            {
                return X == other.X && Y == other.Y;
            }
            return false;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(X, Y);
        }

        public static implicit operator AdtCoordinate((int X, int Y) tuple)
        {
            return new AdtCoordinate(tuple.X, tuple.Y);
        }

        public void Deconstruct(out int x, out int y)
        {
            x = X;
            y = Y;
        }
    }

    public class AdtCoordinateConverter : JsonConverter<AdtCoordinate>
    {
        public override AdtCoordinate Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException();

            int x = 0, y = 0;

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                    return new AdtCoordinate(x, y);

                if (reader.TokenType != JsonTokenType.PropertyName)
                    throw new JsonException();

                var propertyName = reader.GetString();
                reader.Read();

                switch (propertyName)
                {
                    case "X":
                        x = reader.GetInt32();
                        break;
                    case "Y":
                        y = reader.GetInt32();
                        break;
                }
            }

            throw new JsonException();
        }

        public override void Write(Utf8JsonWriter writer, AdtCoordinate value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            writer.WriteNumber("X", value.X);
            writer.WriteNumber("Y", value.Y);
            writer.WriteEndObject();
        }
    }

    public class LiquidBlockInfo
    {
        public string? SourceFile { get; set; }
        public float GlobalX { get; set; }
        public float GlobalY { get; set; }
        public float GlobalZ { get; set; }
        public int LiquidType { get; set; }
        public float MinHeight { get; set; }
        public float MaxHeight { get; set; }
    }

    public class AdtMappingInfo
    {
        public required AdtCoordinate AdtCoordinates { get; set; }
        public List<LiquidBlock> LiquidBlocks { get; set; } = new List<LiquidBlock>();
    }

    public class LiquidMapping
    {
        private readonly string _outputPath;
        private readonly bool _verbose;
        public Dictionary<string, AdtMappingInfo> AdtMappings { get; } = new Dictionary<string, AdtMappingInfo>();

        public LiquidMapping(string outputPath, bool verbose = false)
        {
            _outputPath = outputPath;
            _verbose = verbose;
        }

        public void AddLiquidBlock(string wlwFile, LiquidBlock block, WowCoordinates coords)
        {
            var key = $"{coords.AdtCoordinates.X}_{coords.AdtCoordinates.Y}";
            if (!AdtMappings.TryGetValue(key, out var mapping))
            {
                mapping = new AdtMappingInfo { AdtCoordinates = coords.AdtCoordinates };
                AdtMappings[key] = mapping;
            }

            mapping.LiquidBlocks.Add(block);

            if (_verbose)
            {
                Console.WriteLine($"Added liquid block from {wlwFile} at ({block.GlobalX}, {block.GlobalY}) to ADT ({coords.AdtCoordinates.X}, {coords.AdtCoordinates.Y})");
            }
        }

        public void SaveMapping()
        {
            var mappingPath = Path.Combine(_outputPath, "liquid_mapping.json");
            var summaryPath = Path.Combine(_outputPath, "liquid_mapping_summary.txt");

            // Save JSON mapping
            var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(AdtMappings, jsonOptions);
            File.WriteAllText(mappingPath, json);

            // Save summary
            using var writer = new StreamWriter(summaryPath);
            writer.WriteLine("Liquid Mapping Summary");
            writer.WriteLine("=====================");
            writer.WriteLine();

            foreach (var mapping in AdtMappings)
            {
                writer.WriteLine($"ADT ({mapping.Value.AdtCoordinates.X}, {mapping.Value.AdtCoordinates.Y}):");
                writer.WriteLine($"  Total blocks: {mapping.Value.LiquidBlocks.Count}");
                writer.WriteLine();
            }

            if (_verbose)
            {
                Console.WriteLine($"Saved liquid mapping to {mappingPath}");
                Console.WriteLine($"Saved liquid mapping summary to {summaryPath}");
            }
        }
    }

    public class TupleConverter : JsonConverter<(int X, int Y)>
    {
        public override (int X, int Y) Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException();

            int x = 0, y = 0;

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                    return (x, y);

                if (reader.TokenType != JsonTokenType.PropertyName)
                    throw new JsonException();

                var propertyName = reader.GetString();
                reader.Read();

                switch (propertyName)
                {
                    case "X":
                        x = reader.GetInt32();
                        break;
                    case "Y":
                        y = reader.GetInt32();
                        break;
                }
            }

            throw new JsonException();
        }

        public override void Write(Utf8JsonWriter writer, (int X, int Y) value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            writer.WriteNumber("X", value.X);
            writer.WriteNumber("Y", value.Y);
            writer.WriteEndObject();
        }
    }
} 