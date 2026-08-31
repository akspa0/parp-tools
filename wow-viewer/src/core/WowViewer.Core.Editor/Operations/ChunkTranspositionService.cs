using System.Numerics;

namespace WowViewer.Core.Editor.Operations;

/// <summary>
/// Service executing sub-cell chunk extraction, spatial transformation, texture harmonization,
/// and multi-tile destination transposition.
/// </summary>
public static class ChunkTranspositionService
{
    public static ChunkTranspositionPayload ExtractPayload(
        IEnumerable<GlobalChunkCoordinate> coordinates,
        Func<GlobalChunkCoordinate, TransposedChunkRecord?> chunkReader,
        Func<GlobalChunkCoordinate, IEnumerable<TransposedObjectPlacement>?>? placementReader = null)
    {
        ArgumentNullException.ThrowIfNull(coordinates);
        ArgumentNullException.ThrowIfNull(chunkReader);

        var coordList = coordinates.Where(static c => c.IsValid).ToList();
        if (coordList.Count == 0)
            return new ChunkTranspositionPayload();

        int minGx = coordList.Min(static c => c.Gx);
        int minGy = coordList.Min(static c => c.Gy);
        int maxGx = coordList.Max(static c => c.Gx);
        int maxGy = coordList.Max(static c => c.Gy);

        var origin = new GlobalChunkCoordinate(minGx, minGy);
        var payload = new ChunkTranspositionPayload
        {
            Origin = origin,
            WidthInChunks = maxGx - minGx + 1,
            HeightInChunks = maxGy - minGy + 1,
        };

        foreach (var coord in coordList)
        {
            var record = chunkReader(coord);
            if (record != null)
            {
                record.RelativeGx = coord.Gx - minGx;
                record.RelativeGy = coord.Gy - minGy;
                payload.Chunks.Add(record);
            }

            if (placementReader != null)
            {
                var placements = placementReader(coord);
                if (placements != null)
                {
                    foreach (var p in placements)
                        payload.Placements.Add(p);
                }
            }
        }

        return payload;
    }

    public static ChunkTranspositionPayload TransformPayload(
        ChunkTranspositionPayload source,
        ChunkTranspositionOptions options)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(options);

        var result = new ChunkTranspositionPayload
        {
            Origin = source.Origin,
            WidthInChunks = source.WidthInChunks,
            HeightInChunks = source.HeightInChunks,
        };

        int rot = ((options.RotationDegrees % 360) + 360) % 360;

        foreach (var chunk in source.Chunks)
        {
            int rx = chunk.RelativeGx;
            int ry = chunk.RelativeGy;

            if (options.MirrorX) rx = source.WidthInChunks - 1 - rx;
            if (options.MirrorY) ry = source.HeightInChunks - 1 - ry;

            if (rot == 90)
            {
                int temp = rx;
                rx = source.HeightInChunks - 1 - ry;
                ry = temp;
            }
            else if (rot == 180)
            {
                rx = source.WidthInChunks - 1 - rx;
                ry = source.HeightInChunks - 1 - ry;
            }
            else if (rot == 270)
            {
                int temp = rx;
                rx = ry;
                ry = source.WidthInChunks - 1 - temp;
            }

            float[]? newHeights = chunk.Heights != null ? (float[])chunk.Heights.Clone() : null;
            if (newHeights != null && options.HeightOffset != 0f)
            {
                for (int i = 0; i < newHeights.Length; i++)
                    newHeights[i] += options.HeightOffset;
            }

            var transformed = new TransposedChunkRecord
            {
                RelativeGx = rx,
                RelativeGy = ry,
                Heights = newHeights,
                Normals = chunk.Normals != null ? (Vector3[])chunk.Normals.Clone() : null,
                HoleMask = chunk.HoleMask,
                AreaId = chunk.AreaId,
                McnkFlags = chunk.McnkFlags,
                ShadowMap = chunk.ShadowMap != null ? (byte[])chunk.ShadowMap.Clone() : null,
                MccvColors = chunk.MccvColors != null ? (byte[])chunk.MccvColors.Clone() : null,
                Liquid = chunk.Liquid != null ? (byte[])chunk.Liquid.Clone() : null,
            };

            foreach (var l in chunk.Layers)
            {
                transformed.Layers.Add(new TransposedLayerRecord
                {
                    TextureIndex = l.TextureIndex,
                    TexturePath = l.TexturePath,
                    AlphaMap = l.AlphaMap != null ? (byte[])l.AlphaMap.Clone() : null,
                    Flags = l.Flags,
                    EffectId = l.EffectId,
                });
            }

            result.Chunks.Add(transformed);
        }

        if (rot == 90 || rot == 270)
        {
            result.WidthInChunks = source.HeightInChunks;
            result.HeightInChunks = source.WidthInChunks;
        }

        foreach (var p in source.Placements)
        {
            Vector3 relPos = p.RelativePosition;
            if (options.HeightOffset != 0f)
                relPos.Z += options.HeightOffset;

            result.Placements.Add(new TransposedObjectPlacement
            {
                IsWmo = p.IsWmo,
                AssetPath = p.AssetPath,
                RelativePosition = relPos,
                Rotation = p.Rotation,
                Scale = p.Scale,
                UniqueId = p.UniqueId,
            });
        }

        return result;
    }
}
