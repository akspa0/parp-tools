using WowViewer.Core.Audio;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.IO.Audio;

public sealed class AlphaAreaAudioAssetResolver
{
    public AlphaAreaAudioBindingAssetReport Resolve(
        AlphaAreaAudioBinding binding,
        IEnumerable<string> searchPaths,
        IArchiveReader? archiveReader = null)
    {
        ArgumentNullException.ThrowIfNull(binding);
        ArgumentNullException.ThrowIfNull(searchPaths);

        string[] roots = NormalizeSearchPaths(searchPaths);
        Dictionary<string, AssetResolution> cache = new(StringComparer.OrdinalIgnoreCase);
        return ResolveBinding(binding, roots, archiveReader, cache);
    }

    public IReadOnlyList<AlphaAreaAudioBindingAssetReport> ResolveAll(
        IEnumerable<AlphaAreaAudioBinding> bindings,
        IEnumerable<string> searchPaths,
        IArchiveReader? archiveReader = null)
    {
        ArgumentNullException.ThrowIfNull(bindings);
        ArgumentNullException.ThrowIfNull(searchPaths);

        string[] roots = NormalizeSearchPaths(searchPaths);
        Dictionary<string, AssetResolution> cache = new(StringComparer.OrdinalIgnoreCase);

        return bindings
            .Select(binding => ResolveBinding(binding, roots, archiveReader, cache))
            .ToArray();
    }

    private static AlphaAreaAudioBindingAssetReport ResolveBinding(
        AlphaAreaAudioBinding binding,
        string[] searchPaths,
        IArchiveReader? archiveReader,
        Dictionary<string, AssetResolution> cache)
    {
        return new AlphaAreaAudioBindingAssetReport(
            binding,
            ResolveAsset(AlphaAreaAudioAssetRole.DaySequence, binding.MidiAmbience?.DaySequence, searchPaths, archiveReader, cache),
            ResolveAsset(AlphaAreaAudioAssetRole.NightSequence, binding.MidiAmbience?.NightSequence, searchPaths, archiveReader, cache),
            ResolveAsset(AlphaAreaAudioAssetRole.DlsFile, binding.MidiAmbience?.DlsFile, searchPaths, archiveReader, cache),
            ResolveAsset(AlphaAreaAudioAssetRole.UnderwaterDaySequence, binding.UnderwaterMidiAmbience?.DaySequence, searchPaths, archiveReader, cache),
            ResolveAsset(AlphaAreaAudioAssetRole.UnderwaterNightSequence, binding.UnderwaterMidiAmbience?.NightSequence, searchPaths, archiveReader, cache),
            ResolveAsset(AlphaAreaAudioAssetRole.UnderwaterDlsFile, binding.UnderwaterMidiAmbience?.DlsFile, searchPaths, archiveReader, cache));
    }

    private static AlphaAreaAudioAssetProbe ResolveAsset(
        AlphaAreaAudioAssetRole role,
        string? requestedPath,
        string[] searchPaths,
        IArchiveReader? archiveReader,
        Dictionary<string, AssetResolution> cache)
    {
        string normalizedPath = NormalizeVirtualPath(requestedPath);
        if (string.IsNullOrEmpty(normalizedPath))
        {
            return new AlphaAreaAudioAssetProbe(role, string.Empty, null, AlphaAreaAudioAssetSource.None);
        }

        if (!cache.TryGetValue(normalizedPath, out AssetResolution resolution))
        {
            resolution = ResolveAssetPath(normalizedPath, searchPaths, archiveReader);
            cache[normalizedPath] = resolution;
        }

        return new AlphaAreaAudioAssetProbe(role, normalizedPath, resolution.ResolvedPath, resolution.Source);
    }

    private static AssetResolution ResolveAssetPath(
        string normalizedPath,
        string[] searchPaths,
        IArchiveReader? archiveReader)
    {
        string relativePath = normalizedPath.Replace('\\', Path.DirectorySeparatorChar);

        foreach (string root in searchPaths)
        {
            foreach (string baseDirectory in EnumerateCandidateRoots(root))
            {
                string candidate = Path.Combine(baseDirectory, relativePath);
                if (File.Exists(candidate))
                {
                    return new AssetResolution(Path.GetFullPath(candidate), AlphaAreaAudioAssetSource.Disk);
                }
            }
        }

        if (archiveReader?.FileExists(normalizedPath) == true)
        {
            return new AssetResolution(normalizedPath, AlphaAreaAudioAssetSource.Archive);
        }

        return new AssetResolution(null, AlphaAreaAudioAssetSource.None);
    }

    private static IEnumerable<string> EnumerateCandidateRoots(string searchPath)
    {
        if (!Directory.Exists(searchPath))
        {
            yield break;
        }

        yield return Path.GetFullPath(searchPath);

        string dataDirectory = Path.Combine(searchPath, "Data");
        if (Directory.Exists(dataDirectory))
        {
            yield return Path.GetFullPath(dataDirectory);
        }
    }

    private static string[] NormalizeSearchPaths(IEnumerable<string> searchPaths)
    {
        return searchPaths
            .Where(static path => !string.IsNullOrWhiteSpace(path))
            .Select(static path => path.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static string NormalizeVirtualPath(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return string.Empty;
        }

        return path
            .Trim()
            .TrimStart('\\', '/')
            .Replace('/', '\\');
    }

    private sealed record AssetResolution(string? ResolvedPath, AlphaAreaAudioAssetSource Source);
}
