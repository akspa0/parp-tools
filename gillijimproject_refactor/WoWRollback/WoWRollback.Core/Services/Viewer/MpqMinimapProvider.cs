using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using MPQToTACT.MPQ;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Provides minimap tiles from MPQ archives using StormLib.
/// </summary>
public sealed class MpqMinimapProvider : IMinimapProvider, IDisposable
{
    // Version -> MpqArchive
    private readonly Dictionary<string, MpqArchive> _archives = new(StringComparer.OrdinalIgnoreCase);
    
    // Version -> Map -> (TileX, TileY) -> MD5 hash
    private readonly Dictionary<string, Dictionary<string, Dictionary<(int, int), string>>> _tileToMd5 = new(StringComparer.OrdinalIgnoreCase);
    
    private bool _disposed;

    private MpqMinimapProvider()
    {
    }

    /// <summary>
    /// Creates a provider by opening MPQ archives for each version.
    /// </summary>
    /// <param name="versionMpqPaths">Mapping of version identifier to MPQ directory path.</param>
    /// <returns>Configured provider.</returns>
    public static MpqMinimapProvider Build(IReadOnlyDictionary<string, string> versionMpqPaths)
    {
        var provider = new MpqMinimapProvider();

        foreach (var (version, mpqPath) in versionMpqPaths)
        {
            try
            {
                provider.LoadVersion(version, mpqPath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MpqMinimapProvider] Failed to load version {version} from {mpqPath}: {ex.Message}");
                // Continue loading other versions
            }
        }

        return provider;
    }

    public async Task<Stream?> OpenTileAsync(string version, string mapName, int tileX, int tileY)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_archives.TryGetValue(version, out var archive))
            return null;

        if (!_tileToMd5.TryGetValue(version, out var mapLookup))
            return null;

        if (!mapLookup.TryGetValue(mapName, out var tileLookup))
            return null;

        if (!tileLookup.TryGetValue((tileX, tileY), out var md5Hash))
            return null;

        try
        {
            // Try multiple possible paths
            var paths = new[]
            {
                $"textures\\minimap\\{md5Hash}.blp",
                $"Textures\\Minimap\\{md5Hash}.blp",
                $"textures/minimap/{md5Hash}.blp",
                $"Textures/Minimap/{md5Hash}.blp"
            };

            foreach (var path in paths)
            {
                var mpqStream = archive.OpenFile(path);
                if (mpqStream != null)
                {
                    // Copy to MemoryStream for seekability
                    var buffer = new MemoryStream();
                    await mpqStream.CopyToAsync(buffer);
                    mpqStream.Dispose();
                    buffer.Seek(0, SeekOrigin.Begin);
                    return buffer;
                }
            }

            return null;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MpqMinimapProvider] Error opening tile {mapName} ({tileX}, {tileY}) from {version}: {ex.Message}");
            return null;
        }
    }

    public IEnumerable<string> EnumerateMaps(string version)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_tileToMd5.TryGetValue(version, out var mapLookup))
            return Enumerable.Empty<string>();

        return mapLookup.Keys;
    }

    public IEnumerable<(int TileX, int TileY)> EnumerateTiles(string version, string mapName)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_tileToMd5.TryGetValue(version, out var mapLookup))
            yield break;

        if (!mapLookup.TryGetValue(mapName, out var tileLookup))
            yield break;

        foreach (var (tileX, tileY) in tileLookup.Keys)
        {
            yield return (tileX, tileY);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;

        foreach (var archive in _archives.Values)
        {
            try
            {
                archive.Dispose();
            }
            catch
            {
                // Ignore disposal errors
            }
        }

        _archives.Clear();
        _tileToMd5.Clear();
        _disposed = true;
    }

    private void LoadVersion(string version, string mpqDirectory)
    {
        if (!Directory.Exists(mpqDirectory))
        {
            Console.WriteLine($"[MpqMinimapProvider] MPQ directory not found: {mpqDirectory}");
            return;
        }

        // Find all MPQ files in the directory
        var mpqFiles = Directory.GetFiles(mpqDirectory, "*.MPQ", SearchOption.AllDirectories)
            .Concat(Directory.GetFiles(mpqDirectory, "*.mpq", SearchOption.AllDirectories))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (mpqFiles.Count == 0)
        {
            Console.WriteLine($"[MpqMinimapProvider] No MPQ files found in {mpqDirectory}");
            return;
        }

        Console.WriteLine($"[MpqMinimapProvider] Loading {version} from {mpqFiles.Count} MPQ file(s)");

        // Open the base archive (usually the first/largest one)
        MpqArchive? baseArchive = null;
        var patchArchives = new List<string>();

        foreach (var mpqFile in mpqFiles)
        {
            if (baseArchive == null)
            {
                try
                {
                    baseArchive = new MpqArchive(mpqFile, FileAccess.Read);
                    Console.WriteLine($"[MpqMinimapProvider] Opened base archive: {Path.GetFileName(mpqFile)}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[MpqMinimapProvider] Failed to open {Path.GetFileName(mpqFile)}: {ex.Message}");
                }
            }
            else
            {
                patchArchives.Add(mpqFile);
            }
        }

        if (baseArchive == null)
        {
            Console.WriteLine($"[MpqMinimapProvider] Failed to open any MPQ archives for {version}");
            return;
        }

        // Apply patch archives
        if (patchArchives.Count > 0)
        {
            try
            {
                baseArchive.AddPatchArchives(patchArchives);
                Console.WriteLine($"[MpqMinimapProvider] Applied {patchArchives.Count} patch archive(s)");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MpqMinimapProvider] Warning: Failed to apply patches: {ex.Message}");
            }
        }

        _archives[version] = baseArchive;

        // Parse md5translate.trs from the archive
        ParseMd5TranslateFromArchive(version, baseArchive);
    }

    private void ParseMd5TranslateFromArchive(string version, MpqArchive archive)
    {
        // Try multiple possible paths for md5translate.trs/txt
        var paths = new[]
        {
            "textures\\minimap\\md5translate.trs",
            "Textures\\Minimap\\md5translate.trs",
            "textures/minimap/md5translate.trs",
            "Textures/Minimap/md5translate.trs",
            "textures\\minimap\\md5translate.txt",
            "Textures\\Minimap\\md5translate.txt",
            "textures/minimap/md5translate.txt",
            "Textures/Minimap/md5translate.txt"
        };

        Stream? trsStream = null;
        foreach (var path in paths)
        {
            try
            {
                trsStream = archive.OpenFile(path);
                if (trsStream != null)
                {
                    Console.WriteLine($"[MpqMinimapProvider] Found md5translate at: {path}");
                    break;
                }
            }
            catch
            {
                // Try next path
            }
        }

        if (trsStream == null)
        {
            Console.WriteLine($"[MpqMinimapProvider] Warning: md5translate not found in {version} archive");
            return;
        }

        try
        {
            using var reader = new StreamReader(trsStream);
            var mapLookup = new Dictionary<string, Dictionary<(int, int), string>>(StringComparer.OrdinalIgnoreCase);
            string? currentMap = null;

            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine()?.Trim();
                if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#"))
                    continue;

                if (line.StartsWith("dir:", StringComparison.OrdinalIgnoreCase))
                {
                    currentMap = line.Substring(4).Trim();
                    if (!mapLookup.ContainsKey(currentMap))
                    {
                        mapLookup[currentMap] = new Dictionary<(int, int), string>();
                    }
                    continue;
                }

                if (currentMap == null)
                    continue;

                var parts = line.Split('\t');
                if (parts.Length != 2)
                    continue;

                var left = parts[0].Trim();
                var right = parts[1].Trim();

                // Determine which side is the map file (e.g., "map12_34.blp")
                var mapFile = left.Contains("map", StringComparison.OrdinalIgnoreCase) ? left : right;
                var md5Hash = mapFile.Contains("map", StringComparison.OrdinalIgnoreCase) ? right : left;

                // Parse tile coordinates from mapXX_YY.blp
                var stem = Path.GetFileNameWithoutExtension(mapFile);
                if (!stem.StartsWith("map", StringComparison.OrdinalIgnoreCase))
                    continue;

                var coords = stem.Substring(3).Split('_');
                if (coords.Length != 2)
                    continue;

                if (!int.TryParse(coords[0], out var tileX))
                    continue;
                if (!int.TryParse(coords[1], out var tileY))
                    continue;

                // Store the mapping
                mapLookup[currentMap][(tileX, tileY)] = Path.GetFileNameWithoutExtension(md5Hash);
            }

            _tileToMd5[version] = mapLookup;
            var totalTiles = mapLookup.Values.Sum(d => d.Count);
            Console.WriteLine($"[MpqMinimapProvider] Parsed {mapLookup.Count} map(s) with {totalTiles} tile(s) for {version}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MpqMinimapProvider] Error parsing md5translate for {version}: {ex.Message}");
        }
        finally
        {
            trsStream?.Dispose();
        }
    }
}
