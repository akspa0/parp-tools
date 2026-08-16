using System.Numerics;

using WowViewer.Core.Audio;
using WoWViewer.Rendering;
using WoWViewer.Terrain;

namespace WoWViewer.Audio;

/// <summary>
/// Measures what coordinate frame decoded MCSE emitter positions are actually in.
/// <para>
/// <c>AlphaTerrainAdapter.ConvertSoundPosition</c> transforms MCSE positions as
/// <c>chunkCorner - local</c>, on the stated assumption that "Alpha MCSE stores a chunk-local
/// C3Vector". That assumption has never been verified: the Ghidra work on
/// <c>CMapChunk::Create</c> proved the 0x34-byte <em>field layout</em>, not the frame the position
/// is expressed in. If the stored vector is not chunk-local, subtracting it from a chunk corner
/// throws every MCSE emitter tens of thousands of units off-map — which reads exactly as
/// "everything is out of range no matter where the camera goes", and explains why MCNK liquid
/// emitters are the only ones that behave: their positions are derived from the renderer's own
/// <c>chunk.WorldPosition</c> and never pass through this transform.
/// </para>
/// <para>
/// This type does not guess or correct. It reports the raw magnitudes so the frame is settled by
/// measurement, on real client data, in one look.
/// </para>
/// </summary>
public readonly record struct McseFrameEvidence(
    int SampleCount,
    float MinX,
    float MaxX,
    float MinY,
    float MaxY,
    float MinZ,
    float MaxZ,
    int WithinChunkBounds,
    int WithinTileBounds,
    int BeyondTileBounds)
{
    public static McseFrameEvidence Empty { get; } = new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    /// <summary>One chunk edge in world units — the bound a genuinely chunk-local vector must satisfy.</summary>
    public static float ChunkEdge => WoWConstants.ChunkSize / WoWConstants.ChunksPerTileEdge;

    /// <summary>One tile edge in world units.</summary>
    public static float TileEdge => WoWConstants.ChunkSize;

    /// <summary>
    /// The frame the measurement supports, stated plainly. Deliberately reports "inconclusive"
    /// rather than picking a winner from a mixed sample.
    /// </summary>
    public string Verdict
    {
        get
        {
            if (SampleCount == 0)
                return "No MCSE emitters resident — nothing measured.";

            if (WithinChunkBounds == SampleCount)
                return $"CHUNK-LOCAL confirmed: all {SampleCount} positions fall inside one chunk edge ({ChunkEdge:0.##}). The current transform is correct.";

            if (BeyondTileBounds == SampleCount)
                return $"NOT chunk-local: all {SampleCount} positions exceed a tile edge ({TileEdge:0.##}), so `chunkCorner - local` places every emitter off-map. This is the out-of-range cause.";

            if (WithinChunkBounds == 0)
                return $"NOT chunk-local: 0 of {SampleCount} positions fit inside a chunk edge ({ChunkEdge:0.##}). `chunkCorner - local` cannot be the right transform.";

            return $"INCONCLUSIVE: {WithinChunkBounds} of {SampleCount} fit a chunk, {WithinTileBounds} fit a tile, {BeyondTileBounds} exceed a tile. Mixed sample — do not switch frames on this evidence.";
        }
    }

    public static McseFrameEvidence Measure(
        IEnumerable<KeyValuePair<(int TileX, int TileY), IReadOnlyList<TerrainSoundEmitter>>> tiles,
        Func<int, int, bool> tileFilter)
    {
        ArgumentNullException.ThrowIfNull(tiles);
        ArgumentNullException.ThrowIfNull(tileFilter);

        int count = 0;
        float minX = float.MaxValue, maxX = float.MinValue;
        float minY = float.MaxValue, maxY = float.MinValue;
        float minZ = float.MaxValue, maxZ = float.MinValue;
        int withinChunk = 0;
        int withinTile = 0;
        int beyondTile = 0;

        float chunkEdge = ChunkEdge;
        float tileEdge = TileEdge;

        foreach (KeyValuePair<(int TileX, int TileY), IReadOnlyList<TerrainSoundEmitter>> pair in tiles)
        {
            if (!tileFilter(pair.Key.TileX, pair.Key.TileY))
                continue;

            foreach (TerrainSoundEmitter emitter in pair.Value)
            {
                // Only MCSE rows carry a file-authored position. Liquid rows are synthesised from
                // the renderer's chunk position, so including them would mask the very thing being
                // measured.
                if (emitter.TriggerKind != AudioTriggerKind.Mcse)
                    continue;

                Vector3 raw = emitter.RawPosition;
                if (!float.IsFinite(raw.X) || !float.IsFinite(raw.Y) || !float.IsFinite(raw.Z))
                    continue;

                count++;
                minX = MathF.Min(minX, raw.X); maxX = MathF.Max(maxX, raw.X);
                minY = MathF.Min(minY, raw.Y); maxY = MathF.Max(maxY, raw.Y);
                minZ = MathF.Min(minZ, raw.Z); maxZ = MathF.Max(maxZ, raw.Z);

                float planarExtent = MathF.Max(MathF.Abs(raw.X), MathF.Abs(raw.Y));
                if (planarExtent <= chunkEdge)
                    withinChunk++;
                else if (planarExtent <= tileEdge)
                    withinTile++;
                else
                    beyondTile++;
            }
        }

        if (count == 0)
            return Empty;

        return new McseFrameEvidence(
            count, minX, maxX, minY, maxY, minZ, maxZ, withinChunk, withinTile, beyondTile);
    }
}
