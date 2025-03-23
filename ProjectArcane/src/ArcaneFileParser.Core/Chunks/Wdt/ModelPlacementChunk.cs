using System;
using System.IO;
using System.Runtime.InteropServices;
using ArcaneFileParser.Core.Common.Types;

namespace ArcaneFileParser.Core.Chunks.Wdt;

/// <summary>
/// Model placement chunk defining WMO model positions in the world.
/// </summary>
public class ModelPlacementChunk : ChunkBase
{
    /// <summary>
    /// The expected signature for MODF chunks.
    /// </summary>
    public const uint ExpectedSignature = 0x46444F4D; // "MODF"

    /// <summary>
    /// Flags indicating model placement properties.
    /// </summary>
    [Flags]
    public enum ModelFlags : ushort
    {
        None = 0x0,
        DoodadSet0 = 0x1,          // Uses doodad set 0
        DoodadSet1 = 0x2,          // Uses doodad set 1
        DoodadSet2 = 0x4,          // Uses doodad set 2
        DoodadSet3 = 0x8,          // Uses doodad set 3
        HasLights = 0x10,          // Model has lights
        HasDoodads = 0x20,         // Model has doodads
        HasWater = 0x40,           // Model has water
        IsIndoor = 0x80,           // Model is indoor
        ShowSkybox = 0x100,        // Show skybox inside model
        IsMountAllowed = 0x200,    // Mounting is allowed inside
    }

    /// <summary>
    /// Structure representing a single model placement entry.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ModelPlacement
    {
        public uint NameId;             // Index into MWID/string table
        public uint UniqueId;           // Unique identifier
        public Vector3F Position;       // Position in world space
        public Vector3F Rotation;       // Rotation (radians)
        public Vector3F LowerBounds;    // Lower bounds of the model
        public Vector3F UpperBounds;    // Upper bounds of the model
        public ModelFlags Flags;        // Model flags
        public ushort DoodadSetIndex;   // Index of the doodad set
        public ushort NameSetIndex;     // Index of the name set
        public ushort Scale;            // Model scale (percentage)

        /// <summary>
        /// Gets the model's bounding box.
        /// </summary>
        public BoundingBox Bounds => new(LowerBounds, UpperBounds);

        /// <summary>
        /// Gets whether the model uses a specific doodad set.
        /// </summary>
        public bool UsesDoodadSet(int index) =>
            index switch
            {
                0 => Flags.HasFlag(ModelFlags.DoodadSet0),
                1 => Flags.HasFlag(ModelFlags.DoodadSet1),
                2 => Flags.HasFlag(ModelFlags.DoodadSet2),
                3 => Flags.HasFlag(ModelFlags.DoodadSet3),
                _ => false
            };

        /// <summary>
        /// Gets the model's scale as a float (1.0 = 100%).
        /// </summary>
        public float ScaleFloat => Scale / 1024.0f;
    }

    /// <summary>
    /// Gets the array of model placements.
    /// </summary>
    public ModelPlacement[] Placements { get; }

    /// <summary>
    /// Creates a new instance of the MODF chunk.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    public ModelPlacementChunk(BinaryReader reader) : base(reader)
    {
        if (!ValidateSignature(ExpectedSignature))
        {
            return;
        }

        // Each placement is 64 bytes
        int placementCount = (int)(Size / 64);
        Placements = new ModelPlacement[placementCount];

        for (int i = 0; i < placementCount; i++)
        {
            Placements[i] = new ModelPlacement
            {
                NameId = reader.ReadUInt32(),
                UniqueId = reader.ReadUInt32(),
                Position = reader.ReadVector3F(),
                Rotation = reader.ReadVector3F(),
                LowerBounds = reader.ReadVector3F(),
                UpperBounds = reader.ReadVector3F(),
                Flags = (ModelFlags)reader.ReadUInt16(),
                DoodadSetIndex = reader.ReadUInt16(),
                NameSetIndex = reader.ReadUInt16(),
                Scale = reader.ReadUInt16()
            };
        }

        EnsureAtEnd(reader);
    }

    /// <summary>
    /// Gets a model placement by its unique ID.
    /// </summary>
    /// <param name="uniqueId">The unique ID to search for.</param>
    /// <returns>The model placement with the specified ID, or null if not found.</returns>
    public ModelPlacement? GetPlacementById(uint uniqueId)
    {
        foreach (var placement in Placements)
        {
            if (placement.UniqueId == uniqueId)
                return placement;
        }
        return null;
    }

    /// <summary>
    /// Creates a string representation of the chunk for debugging.
    /// </summary>
    public override string ToString() =>
        $"MODF [Size: {Size}, Models: {Placements.Length}, Valid: {IsValid}]";
} 