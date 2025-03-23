using System;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map Chunk Bounding Box subchunk containing terrain bounding box data.
/// </summary>
public class McbbChunk : ChunkBase
{
    public override string ChunkId => "MCBB";

    /// <summary>
    /// Gets or sets the minimum corner of the bounding box.
    /// </summary>
    public Vector3F Min { get; set; }

    /// <summary>
    /// Gets or sets the maximum corner of the bounding box.
    /// </summary>
    public Vector3F Max { get; set; }

    public override void Parse(BinaryReader reader, uint size)
    {
        // Read bounding box corners
        Min = reader.ReadVector3F();
        Max = reader.ReadVector3F();
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        // Write bounding box corners
        writer.WriteVector3F(Min);
        writer.WriteVector3F(Max);
    }

    /// <summary>
    /// Updates the bounding box to include a point.
    /// </summary>
    /// <param name="point">The point to include.</param>
    public void ExpandToInclude(Vector3F point)
    {
        Min = new Vector3F(
            Math.Min(Min.X, point.X),
            Math.Min(Min.Y, point.Y),
            Math.Min(Min.Z, point.Z)
        );

        Max = new Vector3F(
            Math.Max(Max.X, point.X),
            Math.Max(Max.Y, point.Y),
            Math.Max(Max.Z, point.Z)
        );
    }

    /// <summary>
    /// Updates the bounding box to include another bounding box.
    /// </summary>
    /// <param name="other">The other bounding box to include.</param>
    public void ExpandToInclude(McbbChunk other)
    {
        ExpandToInclude(other.Min);
        ExpandToInclude(other.Max);
    }

    /// <summary>
    /// Checks if a point is inside the bounding box.
    /// </summary>
    /// <param name="point">The point to check.</param>
    /// <returns>True if the point is inside the bounding box.</returns>
    public bool Contains(Vector3F point)
    {
        return point.X >= Min.X && point.X <= Max.X &&
               point.Y >= Min.Y && point.Y <= Max.Y &&
               point.Z >= Min.Z && point.Z <= Max.Z;
    }

    /// <summary>
    /// Gets the center point of the bounding box.
    /// </summary>
    /// <returns>The center point.</returns>
    public Vector3F GetCenter()
    {
        return new Vector3F(
            (Min.X + Max.X) * 0.5f,
            (Min.Y + Max.Y) * 0.5f,
            (Min.Z + Max.Z) * 0.5f
        );
    }

    /// <summary>
    /// Gets the dimensions of the bounding box.
    /// </summary>
    /// <returns>The dimensions (width, height, depth).</returns>
    public Vector3F GetDimensions()
    {
        return new Vector3F(
            Max.X - Min.X,
            Max.Y - Min.Y,
            Max.Z - Min.Z
        );
    }

    /// <summary>
    /// Gets the volume of the bounding box.
    /// </summary>
    /// <returns>The volume.</returns>
    public float GetVolume()
    {
        var dimensions = GetDimensions();
        return dimensions.X * dimensions.Y * dimensions.Z;
    }

    public override string ToHumanReadable()
    {
        var builder = new StringBuilder();
        builder.AppendLine("Bounding Box:");
        builder.AppendLine($"  Min: {Min}");
        builder.AppendLine($"  Max: {Max}");
        builder.AppendLine($"  Center: {GetCenter()}");
        builder.AppendLine($"  Dimensions: {GetDimensions()}");
        builder.AppendLine($"  Volume: {GetVolume():F2}");

        return builder.ToString();
    }
} 