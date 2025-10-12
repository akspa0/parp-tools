using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace WoWRollback.Core.Services.Viewer;

public sealed class MinimapLocator
{
    private readonly IMinimapProvider _provider;
    private readonly bool _ownsProvider;

    private MinimapLocator(IMinimapProvider provider, bool ownsProvider = false)
    {
        _provider = provider;
        _ownsProvider = ownsProvider;
    }

    /// <summary>
    /// Creates a locator using loose BLP files from the file system (legacy behavior).
    /// </summary>
    public static MinimapLocator Build(string rootDirectory, IReadOnlyList<string> versions)
    {
        var provider = LooseFileMinimapProvider.Build(rootDirectory, versions);
        return new MinimapLocator(provider, ownsProvider: true);
    }

    /// <summary>
    /// Creates a locator using MPQ archives.
    /// </summary>
    /// <param name="versionMpqPaths">Mapping of version identifier to MPQ directory path.</param>
    public static MinimapLocator BuildFromMpq(IReadOnlyDictionary<string, string> versionMpqPaths)
    {
        var provider = MpqMinimapProvider.Build(versionMpqPaths);
        return new MinimapLocator(provider, ownsProvider: true);
    }

    /// <summary>
    /// Creates a locator from a custom provider.
    /// </summary>
    public static MinimapLocator FromProvider(IMinimapProvider provider, bool ownsProvider = false)
    {
        return new MinimapLocator(provider, ownsProvider);
    }

    /// <summary>
    /// Attempts to open a minimap tile stream asynchronously.
    /// </summary>
    public async Task<Stream?> TryOpenTileAsync(string version, string map, int tileRow, int tileCol)
    {
        // Note: MinimapLocator uses (Row, Col) which maps to (TileY, TileX)
        return await _provider.OpenTileAsync(version, map, tileCol, tileRow);
    }

    /// <summary>
    /// Legacy method for backward compatibility. Returns a MinimapTile struct.
    /// </summary>
    public bool TryGetTile(string version, string map, int tileRow, int tileCol, out MinimapTile tile)
    {
        tile = default;
        
        // Synchronously check if tile exists by attempting to open it
        var stream = _provider.OpenTileAsync(version, map, tileCol, tileRow).GetAwaiter().GetResult();
        if (stream == null)
            return false;

        // Create a temporary MinimapTile that wraps the stream
        // Note: This is not ideal as it holds the stream, but maintains backward compatibility
        tile = new MinimapTile(stream, tileCol, tileRow, version, false);
        return true;
    }

    // --- Enumeration helpers to expose loaded maps/tiles ---
    public IEnumerable<string> EnumerateMaps(string version)
    {
        return _provider.EnumerateMaps(version);
    }

    public IEnumerable<(int Row, int Col)> EnumerateTiles(string version, string map)
    {
        // Convert from provider's (TileX, TileY) to (Row, Col)
        foreach (var (tileX, tileY) in _provider.EnumerateTiles(version, map))
        {
            yield return (tileY, tileX);
        }
    }

    /// <summary>
    /// Minimap tile information struct for backward compatibility.
    /// </summary>
    public readonly record struct MinimapTile(Stream? SourceStream, int TileX, int TileY, string Version, bool IsAlternate)
    {
        public Stream Open()
        {
            if (SourceStream == null)
                throw new InvalidOperationException("No stream available for this tile.");
            
            // If stream is seekable, reset to beginning
            if (SourceStream.CanSeek)
                SourceStream.Seek(0, SeekOrigin.Begin);
            
            return SourceStream;
        }

        public string BuildFileName(string mapName) => $"{mapName}_{TileX}_{TileY}{(IsAlternate ? "__alt" : string.Empty)}.png";

        // Legacy constructor for file path (for LooseFileMinimapProvider compatibility)
        public MinimapTile(string sourcePath, int tileX, int tileY, string version, bool isAlternate)
            : this(File.Exists(sourcePath) ? File.OpenRead(sourcePath) : null, tileX, tileY, version, isAlternate)
        {
        }
    }

}
