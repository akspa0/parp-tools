using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtMcalSummaryReader
{
    public static AdtMcalSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(path));
        return Read(stream, fileSummary);
    }

    public static AdtMcalSummary Read(Stream stream, MapFileSummary fileSummary)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(fileSummary);

        if (fileSummary.Kind is not (MapFileKind.Adt or MapFileKind.AdtTex))
            throw new InvalidDataException($"ADT MCAL summary requires a root ADT or _tex0.adt file, but found {fileSummary.Kind}.");

        AdtTextureFile textureFile = AdtTextureReader.Read(stream, fileSummary);
        AdtMcalDecodeProfile decodeProfile = textureFile.DecodeProfile;

        int mcnkWithLayerTableCount = 0;
        int overlayLayerCount = 0;
        int decodedLayerCount = 0;
        int missingPayloadLayerCount = 0;
        int decodeFailureCount = 0;
        int compressedLayerCount = 0;
        int bigAlphaLayerCount = 0;
        int bigAlphaFixedLayerCount = 0;
        int packedLayerCount = 0;

        foreach (AdtTextureChunk chunk in textureFile.Chunks)
        {
            if (chunk.Layers.Count == 0)
                continue;

            mcnkWithLayerTableCount++;
            overlayLayerCount += Math.Max(0, chunk.Layers.Count - 1);
            foreach (AdtTextureChunkLayer layer in chunk.Layers.Skip(1))
            {
                if (chunk.AlphaPayloadBytes == 0)
                {
                    missingPayloadLayerCount++;
                    continue;
                }

                if (layer.DecodedAlpha is null)
                {
                    if (unchecked((int)layer.AlphaOffset) >= 0 && unchecked((int)layer.AlphaOffset) < chunk.AlphaPayloadBytes)
                        decodeFailureCount++;
                    else
                        missingPayloadLayerCount++;

                    continue;
                }

                decodedLayerCount++;
                switch (layer.DecodedAlpha.Encoding)
                {
                    case AdtMcalAlphaEncoding.Compressed:
                        compressedLayerCount++;
                        break;
                    case AdtMcalAlphaEncoding.BigAlpha:
                        bigAlphaLayerCount++;
                        break;
                    case AdtMcalAlphaEncoding.BigAlphaFixed:
                        bigAlphaFixedLayerCount++;
                        break;
                    case AdtMcalAlphaEncoding.Packed4Bit:
                        packedLayerCount++;
                        break;
                }
            }
        }

        return new AdtMcalSummary(
            fileSummary.SourcePath,
            fileSummary.Kind,
            decodeProfile,
            mcnkWithLayerTableCount,
            overlayLayerCount,
            decodedLayerCount,
            missingPayloadLayerCount,
            decodeFailureCount,
            compressedLayerCount,
            bigAlphaLayerCount,
            bigAlphaFixedLayerCount,
            packedLayerCount);
    }
}