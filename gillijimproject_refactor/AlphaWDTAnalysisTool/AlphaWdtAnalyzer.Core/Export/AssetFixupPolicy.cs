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

    public AssetFixupPolicy(MultiListfileResolver resolver,
        string fallbackTileset,
        string fallbackNonTilesetBlp,
        string fallbackWmo,
        string fallbackM2,
        bool enableFuzzy)
    {
        _resolver = resolver;
        _fallbackTileset = WowPath.Normalize(fallbackTileset);
        _fallbackNonTilesetBlp = WowPath.Normalize(fallbackNonTilesetBlp);
        _fallbackWmo = WowPath.Normalize(fallbackWmo);
        _fallbackM2 = WowPath.Normalize(fallbackM2);
        _enableFuzzy = enableFuzzy;
    }

    public string Resolve(AssetType type, string path)
    {
        var norm = WowPath.Normalize(path);
        if (_resolver.Exists(norm)) return norm;

        switch (type)
        {
            case AssetType.Wmo:
                return ResolveWmo(norm);
            case AssetType.MdxOrM2:
                return ResolveM2(norm);
            default:
                return norm;
        }
    }

    public string ResolveTexture(string texturePath)
    {
        var norm = WowPath.Normalize(texturePath);
        if (_resolver.Exists(norm)) return norm;

        var isTileset = norm.Contains("/tileset/", StringComparison.OrdinalIgnoreCase);
        var isBlp = norm.EndsWith(".blp", StringComparison.OrdinalIgnoreCase);

        if (isTileset && isBlp)
        {
            // Handle _s variant substitutions
            var baseNoExt = Path.ChangeExtension(norm, null);
            if (baseNoExt.EndsWith("_s", StringComparison.OrdinalIgnoreCase))
            {
                var withoutS = baseNoExt.Substring(0, baseNoExt.Length - 2) + ".blp";
                if (_resolver.Exists(withoutS)) return withoutS;
            }
            else
            {
                var withS = baseNoExt + "_s.blp";
                if (_resolver.Exists(withS)) return withS;
            }
            // Fallback tileset texture
            return _resolver.Exists(_fallbackTileset) ? _fallbackTileset : norm;
        }

        // Non-tileset missing texture -> generic temp 64
        return _resolver.Exists(_fallbackNonTilesetBlp) ? _fallbackNonTilesetBlp : norm;
    }

    private string ResolveWmo(string norm)
    {
        if (_enableFuzzy)
        {
            var fuzzy = _resolver.FindSimilar(norm, new[] { ".wmo" });
            if (fuzzy is not null) return fuzzy;
        }
        return _fallbackWmo;
    }

    private string ResolveM2(string norm)
    {
        if (_enableFuzzy)
        {
            var fuzzy = _resolver.FindSimilar(norm, new[] { ".m2", ".mdx" });
            if (fuzzy is not null) return fuzzy;
        }
        return _fallbackM2;
    }
}
