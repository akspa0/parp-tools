using System;
using System.IO;
using System.Linq;

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
    private readonly FixupLogger? _logger;

    public AssetFixupPolicy(
        MultiListfileResolver resolver,
        string fallbackTileset,
        string fallbackNonTilesetBlp,
        string fallbackWmo,
        string fallbackM2,
        bool enableFuzzy,
        bool useFallbacks,
        bool enableFixups,
        FixupLogger? logger = null)
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
    }

    public string Resolve(AssetType type, string path)
    {
        var norm = WowPath.Normalize(path);
        if (_resolver.Exists(norm))
        {
            Log(type, norm, norm, method: "exact");
            return norm;
        }

        switch (type)
        {
            case AssetType.Wmo:
                return ResolveWmo(norm);
            case AssetType.MdxOrM2:
                return ResolveM2(norm);
            default:
                Log(type, norm, norm, method: "preserve");
                return norm;
        }
    }

    public string ResolveTexture(string texturePath)
    {
        var norm = WowPath.Normalize(texturePath);
        if (_resolver.Exists(norm))
        {
            Log(AssetType.Blp, norm, norm, method: "exact");
            return norm;
        }

        var isTileset = norm.Contains("/tileset/", StringComparison.OrdinalIgnoreCase);
        var isBlp = norm.EndsWith(".blp", StringComparison.OrdinalIgnoreCase);

        if (_enableFixups && isTileset && isBlp)
        {
            // Handle _s variant substitutions
            var baseNoExt = Path.ChangeExtension(norm, null);
            if (baseNoExt.EndsWith("_s", StringComparison.OrdinalIgnoreCase))
            {
                var withoutS = baseNoExt.Substring(0, baseNoExt.Length - 2) + ".blp";
                if (_resolver.Exists(withoutS)) { Log(AssetType.Blp, norm, withoutS, method: "tileset_variant"); return withoutS; }
            }
            else
            {
                var withS = baseNoExt + "_s.blp";
                if (_resolver.Exists(withS)) { Log(AssetType.Blp, norm, withS, method: "tileset_variant"); return withS; }
            }
        }

        if (isTileset && isBlp && _useFallbacks && _resolver.Exists(_fallbackTileset))
        {
            Log(AssetType.Blp, norm, _fallbackTileset, method: "fallback");
            return _fallbackTileset;
        }

        if (_useFallbacks && _resolver.Exists(_fallbackNonTilesetBlp))
        {
            Log(AssetType.Blp, norm, _fallbackNonTilesetBlp, method: "fallback");
            return _fallbackNonTilesetBlp;
        }

        Log(AssetType.Blp, norm, norm, method: "preserve");
        return norm;
    }

    private string ResolveWmo(string norm)
    {
        if (_enableFuzzy)
        {
            var fuzzy = _resolver.FindSimilar(norm, new[] { ".wmo" });
            if (fuzzy is not null) { Log(AssetType.Wmo, norm, fuzzy, method: SourceOf(fuzzy, "fuzzy")); return fuzzy; }
        }
        if (_useFallbacks)
        {
            Log(AssetType.Wmo, norm, _fallbackWmo, method: "fallback");
            return _fallbackWmo;
        }
        Log(AssetType.Wmo, norm, norm, method: "preserve");
        return norm;
    }

    private string ResolveM2(string norm)
    {
        if (_enableFuzzy)
        {
            var fuzzy = _resolver.FindSimilar(norm, new[] { ".m2", ".mdx" });
            if (fuzzy is not null) { Log(AssetType.MdxOrM2, norm, fuzzy, method: SourceOf(fuzzy, "fuzzy")); return fuzzy; }
        }
        if (_useFallbacks)
        {
            Log(AssetType.MdxOrM2, norm, _fallbackM2, method: "fallback");
            return _fallbackM2;
        }
        Log(AssetType.MdxOrM2, norm, norm, method: "preserve");
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
}
