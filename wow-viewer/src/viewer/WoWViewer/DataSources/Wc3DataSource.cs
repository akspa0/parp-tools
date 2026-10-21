using System.Buffers.Binary;
using WowViewer.Core.IO.Wc3;

namespace WoWViewer.DataSources;

/// <summary>
/// Warcraft III loose-extracted or MPQ-backed folder as a regular <see cref="IDataSource"/>.
/// Wraps the shared <see cref="Wc3FileSource"/> (existing MPQ catalog + listfile support; no new archive reader).
/// <see cref="ReadFile"/> hands the unchanged model renderer a lossless BLP2 payload for BLP0 beta textures;
/// <see cref="ReadRaw"/> returns the original bytes (viewer, export and unpack paths use that).
/// </summary>
public sealed class Wc3DataSource : IDataSource
{
    private readonly Wc3FileSource _source;
    private readonly HashSet<string> _files;
    private readonly Dictionary<string, string> _fileNameIndex = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _stemIndex = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, byte[]> _converted = new(StringComparer.OrdinalIgnoreCase);
    private readonly object _gate = new();
    private IReadOnlyList<string>? _listCache;

    public Wc3DataSource(string root, string? listfilePath)
    {
        _source = Wc3FileSource.Open(root, listfilePath);
        _files = new HashSet<string>(_source.Files.Select(Normalize), StringComparer.OrdinalIgnoreCase);
        foreach (string file in _source.Files)
        {
            string norm = Normalize(file);
            string fileName = Path.GetFileName(norm);
            string stem = Path.GetFileNameWithoutExtension(norm);
            _fileNameIndex.TryAdd(fileName, norm);
            _stemIndex.TryAdd(stem, norm);
        }
    }

    public string Root => _source.Root;
    public string Summary => _source.Summary;
    public string Name => "WC3 Beta: " + _source.Root;
    public bool IsLoaded => true;
    public List<string> Diagnostics { get; } = [];

    private static string Normalize(string path) => path.Replace('/', '\\').TrimStart('\\');

    public bool TryResolvePath(string virtualPath, out string resolved)
    {
        string norm = Normalize(virtualPath);
        if (_files.Contains(norm))
        {
            resolved = norm;
            return true;
        }

        // 1. Extension flip (.tga <-> .blp)
        string ext = Path.GetExtension(norm);
        if (ext.Equals(".tga", StringComparison.OrdinalIgnoreCase))
        {
            string blpCandidate = Path.ChangeExtension(norm, ".blp");
            if (_files.Contains(blpCandidate)) { resolved = blpCandidate; return true; }
        }
        else if (ext.Equals(".blp", StringComparison.OrdinalIgnoreCase))
        {
            string tgaCandidate = Path.ChangeExtension(norm, ".tga");
            if (_files.Contains(tgaCandidate)) { resolved = tgaCandidate; return true; }
        }

        // 2. Prepend "Textures\" if path had no directory
        if (!norm.Contains('\\'))
        {
            string inTextures = "Textures\\" + norm;
            if (_files.Contains(inTextures)) { resolved = inTextures; return true; }
            if (ext.Equals(".tga", StringComparison.OrdinalIgnoreCase) && _files.Contains("Textures\\" + Path.ChangeExtension(norm, ".blp")))
            {
                resolved = "Textures\\" + Path.ChangeExtension(norm, ".blp");
                return true;
            }
            if (ext.Equals(".blp", StringComparison.OrdinalIgnoreCase) && _files.Contains("Textures\\" + Path.ChangeExtension(norm, ".tga")))
            {
                resolved = "Textures\\" + Path.ChangeExtension(norm, ".tga");
                return true;
            }
        }

        // 3. Fallback: Lookup by filename across the entire source
        string fileName = Path.GetFileName(norm);
        if (_fileNameIndex.TryGetValue(fileName, out string? match))
        {
            resolved = match;
            return true;
        }

        // 4. Fallback: Lookup by filename with extension flip
        if (ext.Equals(".tga", StringComparison.OrdinalIgnoreCase) && _fileNameIndex.TryGetValue(Path.ChangeExtension(fileName, ".blp"), out string? blpMatch))
        {
            resolved = blpMatch;
            return true;
        }
        if (ext.Equals(".blp", StringComparison.OrdinalIgnoreCase) && _fileNameIndex.TryGetValue(Path.ChangeExtension(fileName, ".tga"), out string? tgaMatch))
        {
            resolved = tgaMatch;
            return true;
        }

        // 5. Fallback: Lookup by stem
        string stem = Path.GetFileNameWithoutExtension(norm);
        if (_stemIndex.TryGetValue(stem, out string? stemMatch))
        {
            resolved = stemMatch;
            return true;
        }

        // 6. WC3 Synthetic Replaceable Textures fallback (TeamColor, TeamGlow, Cliff, Lumber)
        if (norm.StartsWith(@"ReplaceableTextures\TeamColor", StringComparison.OrdinalIgnoreCase) ||
            norm.StartsWith(@"ReplaceableTextures\TeamGlow", StringComparison.OrdinalIgnoreCase) ||
            norm.StartsWith(@"ReplaceableTextures\Cliff", StringComparison.OrdinalIgnoreCase) ||
            norm.StartsWith(@"ReplaceableTextures\Lumber", StringComparison.OrdinalIgnoreCase))
        {
            resolved = norm;
            return true;
        }

        resolved = norm;
        return false;
    }

    public bool FileExists(string virtualPath) => TryResolvePath(virtualPath, out _);

    public IReadOnlyList<string> GetFileList(string? extensionFilter = null)
    {
        _listCache ??= _source.Files.Select(Normalize).ToArray();
        if (string.IsNullOrEmpty(extensionFilter)) return _listCache;
        return _listCache.Where(p => p.EndsWith(extensionFilter, StringComparison.OrdinalIgnoreCase)).ToArray();
    }

    public bool TryResolveWritablePath(string virtualPath, out string? fullPath)
    {
        fullPath = null;
        return false;
    }

    /// <summary>Original stored bytes, or null when the entry is absent/unreadable.</summary>
    public byte[]? ReadRaw(string virtualPath)
    {
        try
        {
            if (!TryResolvePath(virtualPath, out string resolved))
                return null;
            return _source.ReadFile(resolved);
        }
        catch (Exception ex)
        {
            AddDiagnostic($"{virtualPath}: {ex.Message}");
            return null;
        }
    }

    public string ExtractFile(string virtualPath, string outputRoot, CancellationToken token = default)
    {
        string path = TryResolvePath(virtualPath, out string resolved) ? resolved : Normalize(virtualPath);
        return _source.ExtractFile(path, outputRoot, token);
    }

    public byte[]? ReadFile(string virtualPath)
    {
        if (!TryResolvePath(virtualPath, out string path))
            return null;

        lock (_gate)
            if (_converted.TryGetValue(path, out byte[]? cached)) return cached;

        byte[]? bytes = ReadRaw(path);
        if (bytes == null)
        {
            byte[]? synthetic = TryGenerateSyntheticReplaceable(path);
            if (synthetic != null)
            {
                lock (_gate)
                {
                    _converted[path] = synthetic;
                    _converted[virtualPath] = synthetic;
                }
                return synthetic;
            }
            return null;
        }

        // If it's already a standard DirectX BLP2, let the renderer read it directly.
        if (bytes.Length >= 148 && bytes.AsSpan().StartsWith("BLP2"u8) && bytes[8] == 2)
            return bytes;

        // Any other image/texture format (BLP0, BLP1, TGA, PNG, etc.): decode to raw RGBA and wrap in synthetic uncompressed BLP2.
        try
        {
            Wc3AssetPreview preview = Wc3AssetPreviewReader.Read(bytes, path, p => ReadRaw(p));
            if (preview.Rgba == null) return bytes;

            byte[] blp = new byte[148 + preview.Rgba.Length];
            "BLP2"u8.CopyTo(blp);
            BinaryPrimitives.WriteUInt32LittleEndian(blp.AsSpan(4), 1);
            blp[8] = 3; blp[9] = 8;
            BinaryPrimitives.WriteInt32LittleEndian(blp.AsSpan(12), preview.Width);
            BinaryPrimitives.WriteInt32LittleEndian(blp.AsSpan(16), preview.Height);
            BinaryPrimitives.WriteUInt32LittleEndian(blp.AsSpan(20), 148);
            BinaryPrimitives.WriteInt32LittleEndian(blp.AsSpan(84), preview.Rgba.Length);
            for (int i = 0; i < preview.Rgba.Length; i += 4)
            {
                blp[148 + i] = preview.Rgba[i + 2];
                blp[148 + i + 1] = preview.Rgba[i + 1];
                blp[148 + i + 2] = preview.Rgba[i];
                blp[148 + i + 3] = preview.Rgba[i + 3];
            }
            lock (_gate)
            {
                _converted[path] = blp;
                _converted[virtualPath] = blp;
            }
            return blp;
        }
        catch (Exception ex)
        {
            AddDiagnostic($"{path}: {ex.Message}");
            return bytes;
        }
    }

    private void AddDiagnostic(string message)
    {
        lock (_gate)
            if (Diagnostics.Count < 32) Diagnostics.Add(message);
    }

    private static byte[]? TryGenerateSyntheticReplaceable(string path)
    {
        string norm = Normalize(path);
        if (norm.StartsWith(@"ReplaceableTextures\TeamColor", StringComparison.OrdinalIgnoreCase))
        {
            // Player 1 Red: 255, 3, 3, 255
            return CreateSolidBlp2(16, 16, 255, 3, 3, 255);
        }
        if (norm.StartsWith(@"ReplaceableTextures\TeamGlow", StringComparison.OrdinalIgnoreCase))
        {
            // Radial red glow
            return CreateGlowBlp2(16, 16, 255, 3, 3);
        }
        if (norm.StartsWith(@"ReplaceableTextures\Cliff", StringComparison.OrdinalIgnoreCase))
        {
            // Neutral stone cliff gray/brown
            return CreateSolidBlp2(16, 16, 130, 120, 110, 255);
        }
        if (norm.StartsWith(@"ReplaceableTextures\Lumber", StringComparison.OrdinalIgnoreCase))
        {
            // Neutral wood brown
            return CreateSolidBlp2(16, 16, 140, 95, 50, 255);
        }
        return null;
    }

    private static byte[] CreateSolidBlp2(int width, int height, byte r, byte g, byte b, byte a)
    {
        int pixelCount = width * height;
        byte[] blp = new byte[148 + pixelCount * 4];
        "BLP2"u8.CopyTo(blp);
        BinaryPrimitives.WriteUInt32LittleEndian(blp.AsSpan(4), 1); // Direct
        blp[8] = 3; blp[9] = 8; // Direct, Argb8888
        BinaryPrimitives.WriteInt32LittleEndian(blp.AsSpan(12), width);
        BinaryPrimitives.WriteInt32LittleEndian(blp.AsSpan(16), height);
        BinaryPrimitives.WriteUInt32LittleEndian(blp.AsSpan(20), 148); // Mip 0 offset
        BinaryPrimitives.WriteInt32LittleEndian(blp.AsSpan(84), pixelCount * 4); // Mip 0 size
        for (int i = 0; i < pixelCount * 4; i += 4)
        {
            blp[148 + i] = b;
            blp[148 + i + 1] = g;
            blp[148 + i + 2] = r;
            blp[148 + i + 3] = a;
        }
        return blp;
    }

    private static byte[] CreateGlowBlp2(int width, int height, byte r, byte g, byte b)
    {
        int pixelCount = width * height;
        byte[] blp = new byte[148 + pixelCount * 4];
        "BLP2"u8.CopyTo(blp);
        BinaryPrimitives.WriteUInt32LittleEndian(blp.AsSpan(4), 1);
        blp[8] = 3; blp[9] = 8;
        BinaryPrimitives.WriteInt32LittleEndian(blp.AsSpan(12), width);
        BinaryPrimitives.WriteInt32LittleEndian(blp.AsSpan(16), height);
        BinaryPrimitives.WriteUInt32LittleEndian(blp.AsSpan(20), 148);
        BinaryPrimitives.WriteInt32LittleEndian(blp.AsSpan(84), pixelCount * 4);
        float cx = (width - 1) / 2f;
        float cy = (height - 1) / 2f;
        float maxRadius = Math.Max(cx, cy);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = (y * width + x) * 4;
                float dx = (x - cx) / maxRadius;
                float dy = (y - cy) / maxRadius;
                float dist = MathF.Sqrt(dx * dx + dy * dy);
                float alphaNorm = Math.Clamp(1f - dist, 0f, 1f);
                byte a = (byte)(alphaNorm * alphaNorm * 255);
                blp[148 + idx] = b;
                blp[148 + idx + 1] = g;
                blp[148 + idx + 2] = r;
                blp[148 + idx + 3] = a;
            }
        }
        return blp;
    }

    public void Dispose()
    {
        lock (_gate) _converted.Clear();
        _source.Dispose();
    }
}
