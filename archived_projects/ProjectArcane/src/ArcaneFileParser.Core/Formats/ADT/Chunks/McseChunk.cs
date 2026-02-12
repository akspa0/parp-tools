using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map Chunk Sound Emitters subchunk containing sound emitter data.
/// </summary>
public class McseChunk : ChunkBase
{
    public override string ChunkId => "MCSE";

    /// <summary>
    /// Sound emitter entry structure.
    /// </summary>
    public struct SoundEmitter
    {
        public uint SoundId;        // Sound entry ID from SoundEntries.dbc
        public Vector3F Position;   // Position in world space
        public float MinDistance;   // Minimum audible distance
        public float MaxDistance;   // Maximum audible distance
        public float Frequency;     // Sound frequency modifier
        public uint Flags;          // Sound flags
        public uint Phase;          // Sound phase
        public uint Unknown;        // Unknown value
    }

    /// <summary>
    /// Gets the list of sound emitters.
    /// </summary>
    public List<SoundEmitter> Emitters { get; } = new();

    public override void Parse(BinaryReader reader, uint size)
    {
        // Clear existing data
        Emitters.Clear();

        // Each emitter entry is 32 bytes
        var emitterCount = size / 32;

        // Read all emitter entries
        for (int i = 0; i < emitterCount; i++)
        {
            var emitter = new SoundEmitter
            {
                SoundId = reader.ReadUInt32(),
                Position = reader.ReadVector3F(),
                MinDistance = reader.ReadSingle(),
                MaxDistance = reader.ReadSingle(),
                Frequency = reader.ReadSingle(),
                Flags = reader.ReadUInt32(),
                Phase = reader.ReadUInt32(),
                Unknown = reader.ReadUInt32()
            };

            Emitters.Add(emitter);
        }
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        // Write each emitter entry
        foreach (var emitter in Emitters)
        {
            writer.Write(emitter.SoundId);
            writer.WriteVector3F(emitter.Position);
            writer.Write(emitter.MinDistance);
            writer.Write(emitter.MaxDistance);
            writer.Write(emitter.Frequency);
            writer.Write(emitter.Flags);
            writer.Write(emitter.Phase);
            writer.Write(emitter.Unknown);
        }
    }

    /// <summary>
    /// Gets a sound emitter by index.
    /// </summary>
    /// <param name="index">Index of the emitter.</param>
    /// <returns>The sound emitter if found, null otherwise.</returns>
    public SoundEmitter? GetEmitter(int index)
    {
        if (index < 0 || index >= Emitters.Count)
            return null;

        return Emitters[index];
    }

    /// <summary>
    /// Adds a new sound emitter.
    /// </summary>
    /// <param name="soundId">Sound entry ID from SoundEntries.dbc.</param>
    /// <param name="position">Position in world space.</param>
    /// <param name="minDistance">Minimum audible distance.</param>
    /// <param name="maxDistance">Maximum audible distance.</param>
    /// <param name="frequency">Sound frequency modifier.</param>
    /// <param name="flags">Sound flags.</param>
    /// <param name="phase">Sound phase.</param>
    public void AddEmitter(uint soundId, Vector3F position, float minDistance, float maxDistance, float frequency = 1.0f, uint flags = 0, uint phase = 0)
    {
        var emitter = new SoundEmitter
        {
            SoundId = soundId,
            Position = position,
            MinDistance = minDistance,
            MaxDistance = maxDistance,
            Frequency = frequency,
            Flags = flags,
            Phase = phase,
            Unknown = 0
        };

        Emitters.Add(emitter);
    }

    /// <summary>
    /// Gets all sound emitters within a specified range of a point.
    /// </summary>
    /// <param name="point">The point to check from.</param>
    /// <param name="range">The range to check within.</param>
    /// <returns>A list of sound emitters within range.</returns>
    public List<SoundEmitter> GetEmittersInRange(Vector3F point, float range)
    {
        var result = new List<SoundEmitter>();
        float rangeSq = range * range;

        foreach (var emitter in Emitters)
        {
            float distSq = Vector3F.DistanceSquared(point, emitter.Position);
            if (distSq <= rangeSq)
            {
                result.Add(emitter);
            }
        }

        return result;
    }

    public override string ToHumanReadable()
    {
        var builder = new StringBuilder();
        builder.AppendLine($"Sound Emitters: {Emitters.Count}");

        for (int i = 0; i < Emitters.Count; i++)
        {
            var emitter = Emitters[i];
            builder.AppendLine($"\nEmitter {i}:");
            builder.AppendLine($"  Sound ID: {emitter.SoundId}");
            builder.AppendLine($"  Position: {emitter.Position}");
            builder.AppendLine($"  Range: {emitter.MinDistance:F2} - {emitter.MaxDistance:F2}");
            builder.AppendLine($"  Frequency: {emitter.Frequency:F2}");
            builder.AppendLine($"  Flags: 0x{emitter.Flags:X8}");
            builder.AppendLine($"  Phase: {emitter.Phase}");
        }

        return builder.ToString();
    }
} 