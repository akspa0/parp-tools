using System;
using System.IO;
using System.Linq;
using AlphaWdtAnalyzer.Core.Assets;

namespace AlphaWdtAnalyzer.Core.Export;

public sealed class AssetFixupPolicy
{
    private readonly MultiListfileResolver _resolver;
    private readonly string _fallbackTileset;
    private readonly string _fallbackNonTilesetBlp;
    private readonly string _fallbackWmo;
    private readonly string _fallbackM2;
    private readonly bool _enableFuzzy;
    private readonly bool _useFallbacks;
    private readonly bool _enableFixups;
    private readonly bool _logExact;
    private readonly FixupLogger? _logger;
    private readonly AssetInventory _inventory;

    public AssetFixupPolicy(
        MultiListfileResolver resolver,
        string fallbackTileset,
        string fallbackNonTilesetBlp,
        string fallbackWmo,
        string fallbackM2,
        bool enableFuzzy,
        bool useFallbacks,
        bool enableFixups,
        FixupLogger? logger,
        AssetInventory inventory,
        bool logExact)
    {
        _resolver = resolver;
        _fallbackTileset = WowPath.Normalize(fallbackTileset);
        _fallbackNonTilesetBlp = WowPath.Normalize(fallbackNonTilesetBlp);
        _fallbackWmo = WowPath.Normalize(fallbackWmo);
        _fallbackM2 = WowPath.Normalize(fallbackM2);
        _enableFuzzy = enableFuzzy;
        _useFallbacks = useFallbacks;
        _enableFixups = enableFixups;
        _logger = logger;
        _inventory = inventory;
        _logExact = logExact;
    }

    public string Resolve(AssetType type, string path)
    {
        return ResolveWithMethod(type, path, out _);
    }

    public string ResolveWithMethod(AssetType type, string path, out string method)
    {
        var norm = WowPath.Normalize(path);
        var inList = _resolver.Exists(norm);
        var onDisk = _inventory.Exists(norm);

        if (inList || onDisk)
        {
            if (_logger is not null)
            {
                if (!inList && onDisk)
                {
                    // no-op for fixup CSV (we only record fuzzy)
                }
                else if (_logExact)
                {
                    Log(type, norm, norm, method: "exact");
                }
            }
            method = onDisk && !inList ? "ondisk_only" : "exact";
            return norm;
        }

        switch (type)
        {
            case AssetType.Wmo:
            {
                if (_enableFuzzy)
                {
                    var fuzzy = _resolver.FindSimilar(norm, new[] { ".wmo" });
                    if (fuzzy is not null) { var m = SourceOf(fuzzy, "fuzzy"); Log(AssetType.Wmo, norm, fuzzy, method: m); method = m; return fuzzy; }
                }
                if (_useFallbacks)
                {
                    Log(AssetType.Wmo, norm, _fallbackWmo, method: "fallback");
                    method = "fallback";
                    return _fallbackWmo;
                }
                method = "preserve_missing";
                return norm;
            }
            case AssetType.MdxOrM2:
            {
                var originalExt = Path.GetExtension(norm).ToLowerInvariant();
                var allowed = (originalExt == ".mdx" || originalExt == ".m2") ? new[] { originalExt } : new[] { ".m2", ".mdx" };
                if (_enableFuzzy)
                {
                    var fuzzy = _resolver.FindSimilar(norm, allowed);
                    if (fuzzy is not null) { var m = SourceOf(fuzzy, "fuzzy"); Log(AssetType.MdxOrM2, norm, fuzzy, method: m); method = m; return fuzzy; }
                }
                if (_useFallbacks)
                {
                    // Prefer a fallback with the same extension as the original if possible
                    var fb = _fallbackM2;
                    if (!string.IsNullOrWhiteSpace(originalExt))
                    {
                        var fbExt = Path.GetExtension(fb).ToLowerInvariant();
                        if (!fbExt.Equals(originalExt, StringComparison.OrdinalIgnoreCase))
                        {
                            var alt = Path.ChangeExtension(fb, originalExt);
                            if (ExistsPath(alt)) fb = alt;
                        }
                    }
                    Log(AssetType.MdxOrM2, norm, fb, method: "fallback");
                    method = "fallback";
                    return fb;
                }
                method = "preserve_missing";
                return norm;
            }
            default:
            {
                method = "preserve_missing";
                return norm;
            }
        }
    }

    public string ResolveTexture(string texturePath)
    {
        return ResolveTextureWithMethod(texturePath, out _);
    }

    public string ResolveTextureWithMethod(string texturePath, out string method)
    {
        var norm = WowPath.Normalize(texturePath);
        var inList = _resolver.Exists(norm);
        var onDisk = _inventory.Exists(norm);
        if (inList || onDisk)
        {
            if (_logger is not null)
            {
                if (!inList && onDisk)
                {
                    // ignore in fixup CSV
                }
                else if (_logExact)
                {
                    Log(AssetType.Blp, norm, norm, method: "exact");
                }
            }
            method = onDisk && !inList ? "ondisk_only" : "exact";
            return norm;
        }

        var isTileset = norm.Contains("/tileset/", StringComparison.OrdinalIgnoreCase);
        var isBlp = norm.EndsWith(".blp", StringComparison.OrdinalIgnoreCase);
        var origBaseNoExt = Path.ChangeExtension(norm, null);
        var originalIsSpecular = origBaseNoExt.EndsWith("_s", StringComparison.OrdinalIgnoreCase);

        if (_enableFixups && isTileset && isBlp)
        {
            // Handle _s variant substitutions (only when original is missing everywhere)
            if (originalIsSpecular)
            {
                var withoutS = origBaseNoExt.Substring(0, origBaseNoExt.Length - 2) + ".blp";
                if (_resolver.Exists(withoutS) || _inventory.Exists(withoutS)) { Log(AssetType.Blp, norm, withoutS, method: "tileset_variant"); method = "tileset_variant"; return withoutS; }
            }
        }

        if (_enableFuzzy && isBlp)
        {
            var fuzzy = _resolver.FindSimilar(norm, new[] { ".blp" });
            if (fuzzy is not null)
            {
                var fuzzyBaseNoExt = Path.ChangeExtension(fuzzy, null);
                // SAFETY: do not map non-_s original to _s candidate
                if (!originalIsSpecular && fuzzyBaseNoExt.EndsWith("_s", StringComparison.OrdinalIgnoreCase))
                {
                    // Try the non-_s flavor of the fuzzy candidate
                    var withoutS = fuzzyBaseNoExt.Substring(0, fuzzyBaseNoExt.Length - 2) + ".blp";
                    if (_resolver.Exists(withoutS) || _inventory.Exists(withoutS))
                    {
                        var m2 = SourceOf(withoutS, "fuzzy");
                        Log(AssetType.Blp, norm, withoutS, method: m2);
                        method = m2;
                        return withoutS;
                    }
                    // Otherwise reject this fuzzy result and continue to fallback stage
                }
                else
                {
                    var m = SourceOf(fuzzy, "fuzzy");
                    Log(AssetType.Blp, norm, fuzzy, method: m);
                    method = m;
                    return fuzzy;
                }
            }
        }

        if (_useFallbacks && isBlp && _resolver.Exists(_fallbackTileset))
        {
            Log(AssetType.Blp, norm, _fallbackTileset, method: "fallback");
            method = "fallback";
            return _fallbackTileset;
        }

        if (_useFallbacks && _resolver.Exists(_fallbackNonTilesetBlp))
        {
            Log(AssetType.Blp, norm, _fallbackNonTilesetBlp, method: "fallback");
            method = "fallback";
            return _fallbackNonTilesetBlp;
        }

        // preserve missing
        method = "preserve_missing";
        return norm;
    }

    private string SourceOf(string resolvedPath, string defaultMethod)
    {
        if (_resolver.ContainsPrimary(resolvedPath)) return defaultMethod + ":primary";
        if (_resolver.ContainsSecondary(resolvedPath)) return defaultMethod + ":secondary";
        return defaultMethod;
    }

    private void Log(AssetType type, string original, string resolved, string method)
    {
        _logger?.Write(new FixupRecord
        {
            Type = type.ToString(),
            Original = original,
            Resolved = resolved,
            Method = method
        });
    }

    // Exposed helpers for in-place MTEX patching
    public string TilesetFallbackPath => _fallbackTileset;
    public string NonTilesetFallbackPath => _fallbackNonTilesetBlp;

    public bool ExistsPath(string path)
    {
        var norm = WowPath.Normalize(path);
        return _resolver.Exists(norm) || _inventory.Exists(norm);
    }
}
