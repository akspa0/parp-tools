using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.AssetReferences;

/// <summary>
/// Pulls the asset paths a world object claims: the doodad models it places and the textures its
/// materials name.
/// </summary>
/// <remarks>
/// Every byte of record decoding here belongs to an existing reader: <see cref="WmoDoodadDetailReader"/>
/// owns MODD/MODN placement decoding and <see cref="WmoMaterialDetailReader"/> owns MOMT/MOTX material
/// decoding — the same readers the viewer already uses. This class does not parse either chunk itself.
/// The only thing it does directly is check which top-level chunks are present, via the same
/// <see cref="WmoRootReaderCommon"/> chunk table those readers are built on, because a world object with
/// no doodads or no materials is an ordinary state and neither reader tolerates a missing required chunk.
/// </remarks>
public static class WmoReferenceExtractor
{
    /// <summary>
    /// Extracts every doodad and texture path named by a world object root file.
    /// Returns paths as authored, without normalisation, so a case or separator difference stays
    /// visible to later comparison rather than being silently repaired here.
    /// </summary>
    public static IReadOnlyList<(AssetReferenceKind Kind, string Path)> Extract(Stream stream, string sourcePath)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        stream.Position = 0;
        (uint? _, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);

        List<(AssetReferenceKind, string)> references = [];

        if (HasChunk(chunks, WmoChunkIds.Modd) && HasChunk(chunks, WmoChunkIds.Modn))
            AddDoodadNames(stream, sourcePath, references);

        if (HasChunk(chunks, WmoChunkIds.Mohd) && HasChunk(chunks, WmoChunkIds.Momt))
            AddTextureNames(stream, sourcePath, references);

        return references;
    }

    private static void AddDoodadNames(Stream stream, string sourcePath, List<(AssetReferenceKind, string)> references)
    {
        stream.Position = 0;
        IReadOnlyList<WmoDoodadPlacementDetail> placements = WmoDoodadDetailReader.ReadPlacements(stream, sourcePath);

        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);
        foreach (WmoDoodadPlacementDetail placement in placements)
        {
            if (placement.ModelPath.Length == 0 || !seen.Add(placement.ModelPath))
                continue;

            references.Add((AssetReferenceKind.PlacedDoodad, placement.ModelPath));
        }
    }

    private static void AddTextureNames(Stream stream, string sourcePath, List<(AssetReferenceKind, string)> references)
    {
        stream.Position = 0;
        IReadOnlyList<WmoMaterialDetail> materials = WmoMaterialDetailReader.Read(stream, sourcePath);

        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);
        foreach (WmoMaterialDetail material in materials)
        {
            AddTextureName(material.Texture1Name, seen, references);
            AddTextureName(material.Texture2Name, seen, references);
            AddTextureName(material.Texture3Name, seen, references);
        }
    }

    private static void AddTextureName(string name, HashSet<string> seen, List<(AssetReferenceKind, string)> references)
    {
        // A material's unused texture slots resolve to an empty string, which is a real authored
        // state (no texture bound), not a reference to a missing empty path.
        if (string.IsNullOrEmpty(name) || !seen.Add(name))
            return;

        references.Add((AssetReferenceKind.WorldObjectTexture, name));
    }

    private static bool HasChunk(IReadOnlyList<ChunkSpan> chunks, FourCC chunkId)
    {
        foreach (ChunkSpan chunk in chunks)
        {
            if (chunk.Header.Id == chunkId)
                return true;
        }

        return false;
    }
}
