using WowViewer.Core.IO.Terrain;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Editor.Operations;

/// <summary>
/// Snapshot of chunk state before or after a terrain stamp operation, enabling exact non-destructive undo/redo.
/// </summary>
public sealed class ChunkTerrainSnapshot
{
    public int ChunkIndex { get; init; }
    public float[] Heights { get; init; } = [];
    public byte[] Normals { get; init; } = [];
    public List<string> TexturePaths { get; init; } = [];
    public List<byte[]> AlphaSplats { get; init; } = [];
}

/// <summary>
/// Reversible editor operation that applies a terrain brush paste to one or more ADT terrain chunks
/// with rotation, scaling, SmoothStep boundary feathering, multi-layer alpha allocation, and normal recalculation.
/// </summary>
public sealed class TerrainStampOperation : EditorOperation
{
    public const string PluginId = "terrain.stamp";

    public string AffectedAdtPath { get; }
    public string PasteId { get; }
    public TerrainStampOptions Options { get; }
    public IReadOnlyList<ChunkTerrainSnapshot> BeforeSnapshots { get; }
    public IReadOnlyList<ChunkTerrainSnapshot> AfterSnapshots { get; }

    public override IReadOnlyList<string> AffectedPaths => string.IsNullOrEmpty(AffectedAdtPath) ? [] : [AffectedAdtPath];

    public TerrainStampOperation(
        string operationId,
        string affectedAdtPath,
        string pasteId,
        TerrainStampOptions options,
        IReadOnlyList<ChunkTerrainSnapshot> beforeSnapshots,
        IReadOnlyList<ChunkTerrainSnapshot> afterSnapshots,
        string description = "Apply terrain stamp")
        : base(operationId, PluginId, description, undoable: true)
    {
        AffectedAdtPath = affectedAdtPath;
        PasteId = pasteId;
        Options = options;
        BeforeSnapshots = beforeSnapshots;
        AfterSnapshots = afterSnapshots;
    }

    public override EditorOperation CreateReverse()
    {
        return new TerrainStampOperation(
            Guid.NewGuid().ToString("N"),
            AffectedAdtPath,
            PasteId,
            Options,
            beforeSnapshots: AfterSnapshots,
            afterSnapshots: BeforeSnapshots,
            description: $"Undo stamp {PasteId}");
    }

    /// <summary>
    /// Applies a terrain brush paste directly onto an array of 145 chunk heights with rotation, scaling, and feathering.
    /// </summary>
    public static float[] StampHeightmap(
        float[] existing145,
        float chunkOriginWorldX,
        float chunkOriginWorldY,
        TerrainBrushPaste paste,
        TerrainStampOptions options)
    {
        ArgumentNullException.ThrowIfNull(existing145);
        ArgumentNullException.ThrowIfNull(paste);
        ArgumentNullException.ThrowIfNull(options);

        var result = (float[])existing145.Clone();
        float rad = options.RotationDegrees * (MathF.PI / 180f);
        float cosR = MathF.Cos(rad);
        float sinR = MathF.Sin(rad);

        float halfW = (paste.WidthMeters * options.Scale) * 0.5f;
        float halfL = (paste.LengthMeters * options.Scale) * 0.5f;
        float feather = MathF.Max(0.01f, options.FeatherRadiusMeters);

        const float ChunkSize = 33.33333f;
        const float Step = ChunkSize / 8f; // 4.16666m between outer vertices

        // Iterate across 145 MCVT vertices (9x9 outer + 8x8 inner)
        for (int vIdx = 0; vIdx < 145; vIdx++)
        {
            float localVx, localVy;
            if (vIdx < 81) // 9x9 outer vertices
            {
                int r = vIdx / 9;
                int c = vIdx % 9;
                localVx = c * Step;
                localVy = r * Step;
            }
            else // 8x8 inner vertices (offset by half step)
            {
                int innerIdx = vIdx - 81;
                int r = innerIdx / 8;
                int c = innerIdx % 8;
                localVx = (c + 0.5f) * Step;
                localVy = (r + 0.5f) * Step;
            }

            // World coordinates of vertex
            float worldX = chunkOriginWorldX + localVx;
            float worldY = chunkOriginWorldY + localVy;

            // Offset relative to stamp center
            float dx = worldX - options.CenterWorldX;
            float dy = worldY - options.CenterWorldY;

            // Rotate back into paste local space
            float localX = dx * cosR + dy * sinR;
            float localY = -dx * sinR + dy * cosR;

            // Check if vertex falls within stamp footprint
            if (MathF.Abs(localX) <= halfW && MathF.Abs(localY) <= halfL)
            {
                float u = (localX + halfW) / (halfW * 2f);
                float v = (localY + halfL) / (halfL * 2f);

                float sampledDelta = paste.SampleHeight(u, v) * options.HeightMultiplier;

                // Calculate edge distance for SmoothStep boundary feathering
                float distToEdgeX = halfW - MathF.Abs(localX);
                float distToEdgeY = halfL - MathF.Abs(localY);
                float minEdgeDist = MathF.Min(distToEdgeX, distToEdgeY);

                float weight = SmoothStep(0f, feather, minEdgeDist) * options.StampWeight;

                switch (options.BlendMode)
                {
                    case TerrainStampBlendMode.Additive:
                        result[vIdx] += sampledDelta * weight;
                        break;
                    case TerrainStampBlendMode.Replace:
                        result[vIdx] = (result[vIdx] * (1f - weight)) + (sampledDelta * weight);
                        break;
                    case TerrainStampBlendMode.Maximum:
                        result[vIdx] = MathF.Max(result[vIdx], sampledDelta * weight);
                        break;
                    case TerrainStampBlendMode.Minimum:
                        result[vIdx] = MathF.Min(result[vIdx], sampledDelta * weight);
                        break;
                }
            }
        }

        return result;
    }

    private static float SmoothStep(float edge0, float edge1, float x)
    {
        float t = Math.Clamp((x - edge0) / (edge1 - edge0), 0f, 1f);
        return t * t * (3f - 2f * t);
    }
}
