using System;
using System.Threading.Tasks;

namespace WoWRollback.LkToAlphaModule.Services;

public sealed class AdtConverterService
{
    public Task<LkToAlphaConversionResult> ConvertTileAsync(
        string lkRootAdtPath,
        string lkObjAdtPath,
        string lkTexAdtPath,
        string outFile,
        string mapName,
        LkToAlphaOptions opts)
    {
        if (opts is null) throw new ArgumentNullException(nameof(opts));
        // TODO: Wire Readers -> Converters -> Writers
        return Task.FromResult(new LkToAlphaConversionResult(
            AlphaOutputDirectory: System.IO.Path.GetDirectoryName(outFile) ?? string.Empty,
            TilesProcessed: 0,
            Success: false,
            ErrorMessage: "Not implemented"));
    }

    public Task<LkToAlphaConversionResult> ConvertMapAsync(
        string lkDir,
        string outDir,
        string mapName,
        LkToAlphaOptions opts)
    {
        if (opts is null) throw new ArgumentNullException(nameof(opts));
        // TODO: Enumerate tiles and call ConvertTileAsync
        return Task.FromResult(new LkToAlphaConversionResult(
            AlphaOutputDirectory: outDir,
            TilesProcessed: 0,
            Success: false,
            ErrorMessage: "Not implemented"));
    }
}
