using System.Numerics;
using WowViewer.Core.Audio;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WoWViewer.Rendering;

namespace WoWViewer.Terrain;

/// <summary>
/// Projects decoded MCNK liquid/environment state into the resident audio
/// trigger list. SoundWaterType supplies the build-scoped SoundEntries ID later
/// in WorldAudioRuntime; this factory never invents one.
/// </summary>
internal static class LegacyLiquidSoundEmitterFactory
{
    private const uint LiquidFlagMask = 0x3Cu;

    public static void Append(List<TerrainSoundEmitter> emitters, TerrainChunkData chunk)
    {
        ArgumentNullException.ThrowIfNull(emitters);
        ArgumentNullException.ThrowIfNull(chunk);

        uint mcnkFlags = unchecked((uint)chunk.McnkFlags);
        if ((mcnkFlags & LiquidFlagMask) == 0 && chunk.Liquid is null)
            return;

        // MH2O is applied after the MCNK pass in the standard adapter. Replace
        // the provisional MCNK-only row when the later liquid source supplies
        // the authoritative family/height instead of leaving two rows for one
        // chunk.
        for (int index = emitters.Count - 1; index >= 0; index--)
        {
            TerrainSoundEmitter existing = emitters[index];
            if (existing.TriggerKind == AudioTriggerKind.McnkLiquid
                && existing.TileX == chunk.TileX
                && existing.TileY == chunk.TileY
                && existing.ChunkX == chunk.ChunkX
                && existing.ChunkY == chunk.ChunkY)
            {
                emitters.RemoveAt(index);
            }
        }

        AdtLiquidBasicType liquidType = chunk.Liquid is { } liquid
            ? (AdtLiquidBasicType)liquid.Type
            : McnkFlagDecoder.Decode(mcnkFlags);
        int liquidFamily = (int)liquidType;
        int soundSubtype = ResolveSoundSubtype(mcnkFlags, liquidType);
        Vector3 position = ResolvePosition(chunk);
        uint soundPointId = 0x80000000u
            | ((uint)(chunk.ChunkY & 0x0F) << 24)
            | ((uint)(chunk.ChunkX & 0x0F) << 16)
            | ((uint)liquidFamily << 8)
            | (uint)(soundSubtype & 0xFF);

        emitters.Add(new TerrainSoundEmitter(
            chunk.TileX,
            chunk.TileY,
            chunk.ChunkX,
            chunk.ChunkY,
            soundPointId,
            0,
            position,
            position,
            0f,
            0f,
            0f,
            0,
            0,
            0,
            Array.Empty<byte>(),
            TriggerKind: AudioTriggerKind.McnkLiquid,
            McnkFlags: mcnkFlags,
            LiquidFamily: liquidFamily,
            SoundWaterSubtype: soundSubtype));
    }

    private static int ResolveSoundSubtype(uint mcnkFlags, AdtLiquidBasicType liquidType)
    {
        // SoundWaterType encodes the legacy fluid-speed/subtype selector as
        // 0, 4, or 8. MCNK's 0x04 river and 0x08 ocean bits are the build
        // evidence available at this boundary; magma/slime use 0.
        if (liquidType is AdtLiquidBasicType.Magma or AdtLiquidBasicType.Slime)
            return 0;
        if ((mcnkFlags & 0x08u) != 0)
            return 8;
        if ((mcnkFlags & 0x04u) != 0)
            return 4;
        return 0;
    }

    private static Vector3 ResolvePosition(TerrainChunkData chunk)
    {
        float surface = chunk.Liquid is { } liquid
            && float.IsFinite(liquid.MinHeight)
            && float.IsFinite(liquid.MaxHeight)
            ? (liquid.MinHeight + liquid.MaxHeight) * 0.5f
            : AverageFiniteHeight(chunk.Heights);

        float chunkSize = WoWConstants.ChunkSize / 16f;
        return new Vector3(
            chunk.WorldPosition.X + chunkSize * 0.5f,
            chunk.WorldPosition.Y + chunkSize * 0.5f,
            surface);
    }

    private static float AverageFiniteHeight(float[] heights)
    {
        if (heights is null || heights.Length == 0)
            return 0f;

        double sum = 0d;
        int count = 0;
        for (int index = 0; index < heights.Length; index++)
        {
            float height = heights[index];
            if (!float.IsFinite(height) || MathF.Abs(height) > 50000f)
                continue;
            sum += height;
            count++;
        }

        return count == 0 ? 0f : (float)(sum / count);
    }
}
