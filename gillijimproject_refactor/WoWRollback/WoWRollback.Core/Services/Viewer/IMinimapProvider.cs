using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Abstraction for accessing minimap tiles from various sources (loose files, MPQ archives, etc.).
/// </summary>
public interface IMinimapProvider
{
    /// <summary>
    /// Attempts to locate and open a minimap tile stream for the specified version, map, and tile coordinates.
    /// </summary>
    /// <param name="version">Version identifier (e.g., "0.5.3").</param>
    /// <param name="mapName">Map name (e.g., "Kalimdor").</param>
    /// <param name="tileX">Tile X coordinate.</param>
    /// <param name="tileY">Tile Y coordinate.</param>
    /// <returns>Stream containing BLP or PNG data if found; null otherwise. Caller must dispose the stream.</returns>
    Task<Stream?> OpenTileAsync(string version, string mapName, int tileX, int tileY);

    /// <summary>
    /// Enumerates all maps available for a specific version.
    /// </summary>
    /// <param name="version">Version identifier.</param>
    /// <returns>Collection of map names.</returns>
    IEnumerable<string> EnumerateMaps(string version);

    /// <summary>
    /// Enumerates all available tile coordinates for a specific version and map.
    /// </summary>
    /// <param name="version">Version identifier.</param>
    /// <param name="mapName">Map name.</param>
    /// <returns>Collection of (TileX, TileY) tuples.</returns>
    IEnumerable<(int TileX, int TileY)> EnumerateTiles(string version, string mapName);
}
