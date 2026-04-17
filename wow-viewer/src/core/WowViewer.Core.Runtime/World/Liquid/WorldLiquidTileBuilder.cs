using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Runtime.World.Liquid;

public static class WorldLiquidTileBuilder
{
    public static WorldLiquidTileData Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(path));
        return Read(stream, fileSummary);
    }

    public static WorldLiquidTileData Read(Stream stream, MapFileSummary fileSummary)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(fileSummary);

        if (fileSummary.Kind != MapFileKind.Adt)
            throw new InvalidDataException($"World liquid tile builder requires a root ADT file, but found {fileSummary.Kind}.");

        AdtLiquidFile liquidFile = AdtLiquidReader.Read(stream, fileSummary);
        List<WorldLiquidChunkData> chunks = new(liquidFile.Chunks.Count);
        foreach (AdtLiquidChunk chunk in liquidFile.Chunks)
        {
            if (chunk.Layers.Count == 0)
                continue;

            List<WorldLiquidLayerData> layers = new(chunk.Layers.Count);
            foreach (AdtLiquidLayer layer in chunk.Layers)
            {
                layers.Add(new WorldLiquidLayerData(
                    layer.LiquidTypeId,
                    layer.BasicType,
                    layer.VertexFormat,
                    layer.MinHeight,
                    layer.MaxHeight,
                    layer.XOffset,
                    layer.YOffset,
                    layer.Width,
                    layer.Height,
                    layer.VisibleTileCount,
                    layer.Depths is not null,
                    layer.Heights is not null,
                    layer.Uvs is not null));
            }

            chunks.Add(new WorldLiquidChunkData(
                chunk.ChunkIndex,
                chunk.ChunkIndex % 16,
                chunk.ChunkIndex / 16,
                chunk.FishableMask,
                chunk.DeepMask,
                layers));
        }

        return new WorldLiquidTileData(liquidFile.SourcePath, liquidFile.Kind, chunks);
    }
}