using System;
using System.IO;
using System.Numerics;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using SharpGLTF.Geometry;
using SharpGLTF.Geometry.VertexTypes;
using SharpGLTF.Materials;
using SharpGLTF.Scenes;
using SharpGLTF.Schema2;
using BLPSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.Processing;

namespace WoWRollback.WDLtoGLB;

/// <summary>
/// Ported WDL → GLB exporter (merged and per-tile). Adds optional texture support.
/// Coordinates: Z is up; triangles wound for front faces with Z-up mapping.
/// </summary>
public static class WdlGltfExporter
{
    private const double CellWorldSize = 533.3333333333 / 16.0; // yards per height cell

    public sealed record ExportOptions(
        double Scale = 1.0,
        bool SkipHoles = true,
        bool NormalizeWorld = true,
        double HeightScale = 1.0,
        string? TexturePath = null,
        string? MapName = null,
        string? MinimapFolder = null,
        string? MinimapRoot = null,
        string? TrsPath = null
    );

    public sealed record ExportStats(
        int TilesExported,
        int VerticesApprox,
        int FacesWritten
    );

    /// <summary>
    /// Export a merged GLB of all available 64x64 WDL tiles.
    /// If TexturePath is provided, a single texture is applied with UVs spanning the full map.
    /// </summary>
    public static ExportStats ExportMerged(Wdl wdl, string mergedGlbPath, ExportOptions? options = null)
    {
        options ??= new ExportOptions();
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(mergedGlbPath)) ?? ".");

        int tiles = 0, faces = 0, vApprox = 0;
        var scene = new SceneBuilder();
        double xyScale = (options.NormalizeWorld ? CellWorldSize : 1.0) * options.Scale;

        // Full-map UV span across XZ plane (Y is up)
        float totalCellsX = 64 * 16;
        float totalCellsZ = 64 * 16;
        float widthWorld = (float)(totalCellsX * xyScale);
        float heightWorld = (float)(totalCellsZ * xyScale);

        // Build a shared material (with or without texture)
        // If no texture provided but TRS/minimap is available, auto-build a mosaic next to the GLB.
        string? finalTexture = options.TexturePath;
        if (string.IsNullOrWhiteSpace(finalTexture))
        {
            var outDir = Path.GetDirectoryName(Path.GetFullPath(mergedGlbPath)) ?? ".";
            var stem = Path.GetFileNameWithoutExtension(mergedGlbPath);
            var mosaic = BuildMinimapMosaicFromTrs(outDir, stem, options.MinimapRoot, options.TrsPath, options.MapName, 256);
            if (!string.IsNullOrWhiteSpace(mosaic) && File.Exists(mosaic))
            {
                finalTexture = mosaic;
                System.Console.WriteLine($"[info] Built mosaic for merged texture: {mosaic}");
            }
        }
        var mat = CreateTerrainMaterial(finalTexture);

        for (int ty = 0; ty < 64; ty++)
        {
            for (int tx = 0; tx < 64; tx++)
            {
                var t = wdl.Tiles[ty, tx];
                if (t is null) continue;

                // Base translation for this tile in world X/Z (Y is up). North-up ⇒ Z decreases as ty increases.
                float baseX = (float)(tx * 16 * xyScale);
                float baseZ = (float)(-ty * 16 * xyScale);

                var mesh = BuildTileMesh(t, xyScale, options.HeightScale, options.SkipHoles,
                                         mat,
                                         baseX, baseZ, widthWorld, heightWorld,
                                         out int vCount, out int fCount,
                                         perTileUvs: false);
                var xform = Matrix4x4.CreateTranslation(baseX, 0f, baseZ);
                scene.AddRigidMesh(mesh, xform);

                tiles++;
                faces += fCount;
                vApprox += vCount;
            }
        }

        var model = scene.ToGltf2();
        model.SaveGLB(mergedGlbPath);
        return new ExportStats(tiles, vApprox, faces);
    }

    /// <summary>
    /// Export per-tile GLBs; if TexturePath is provided, UVs are 0..1 within each tile.
    /// </summary>
    public static ExportStats ExportPerTile(Wdl wdl, string tilesDir, ExportOptions? options = null)
    {
        options ??= new ExportOptions();
        Directory.CreateDirectory(tilesDir);

        int tiles = 0, faces = 0, vApprox = 0;
        double xyScale = (options.NormalizeWorld ? CellWorldSize : 1.0) * options.Scale;

        // Create a default material if no per-tile texture is found
        var defaultMat = CreateTerrainMaterial(options.TexturePath);
        // Build index of available images under MinimapFolder, if provided
        if (!string.IsNullOrWhiteSpace(options.MinimapFolder))
        {
            EnsureMinimapIndex(options.MinimapFolder!);
        }

        // Try to build TRS index if either an explicit TRS path or a minimap root is provided
        var trsIndex = TryBuildTrsIndex(options.MinimapRoot, options.TrsPath, out var trsInfo);
        if (trsInfo.Used && trsInfo.Entries > 0)
        {
            System.Console.WriteLine($"[info] TRS loaded: entries={trsInfo.Entries} (root={options.MinimapRoot ?? ""})");
        }

        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                var t = wdl.Tiles[y, x];
                if (t is null) continue;

                var scene = new SceneBuilder();
                // World placement for this tile (north-up):
                float baseX = (float)(x * 16 * xyScale);
                float baseZ = (float)(-y * 16 * xyScale);
                // Resolve per-tile image preferring TRS mapping when available, else fall back to name index
                var mat = defaultMat;
                string? resolvedPath = null;
                if (trsIndex != null && !string.IsNullOrWhiteSpace(options.MapName))
                {
                    var key = (options.MapName!.ToLowerInvariant(), x, y);
                    if (trsIndex.TryGetValue(key, out var blp)) resolvedPath = blp;
                }
                if (resolvedPath == null && !string.IsNullOrWhiteSpace(options.MinimapFolder))
                {
                    if (TryResolveTileImage(options.MapName, options.MinimapFolder!, x, y, out var imagePath))
                        resolvedPath = imagePath;
                }

                if (!string.IsNullOrWhiteSpace(resolvedPath))
                {
                    if (resolvedPath.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
                    {
                        var pngBytes = TryDecodeBlpToPng(resolvedPath);
                        if (pngBytes != null) mat = CreateTerrainMaterialFromBytes(pngBytes, $"tile_{x}_{y}.png");
                    }
                    else
                    {
                        mat = CreateTerrainMaterial(resolvedPath);
                    }
                }
                // For per-tile, use per-tile UV span (0..1)
                var mesh = BuildTileMesh(t, xyScale, options.HeightScale, options.SkipHoles,
                                         mat,
                                         0f, 0f, (float)(16 * xyScale), (float)(16 * xyScale),
                                         out int vCount, out int fCount,
                                         perTileUvs: true);
                scene.AddRigidMesh(mesh, Matrix4x4.CreateTranslation(baseX, 0f, baseZ));

                var model = scene.ToGltf2();
                var outPath = Path.Combine(tilesDir, $"tile_{x}_{y}.glb");
                model.SaveGLB(outPath);

                tiles++;
                faces += fCount;
                vApprox += vCount;
            }
        }
        return new ExportStats(tiles, vApprox, faces);
    }

    private static MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty> BuildTileMesh(
        WdlTile tile,
        double xyScale,
        double heightScale,
        bool skipHoles,
        MaterialBuilder material,
        float tileBaseX,
        float tileBaseY,
        float uvWidthWorld,
        float uvHeightWorld,
        out int verticesApprox,
        out int faces,
        bool perTileUvs = false)
    {
        var mesh = new MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>("tile");
        var prim = mesh.UsePrimitive(material);
        faces = 0;

        // Precompute normals per 17x17 vertex (Y is up = height). North-up: Z decreases with j.
        var normals = new Vector3[17, 17];
        for (int j = 0; j <= 16; j++)
        {
            for (int i = 0; i <= 16; i++)
            {
                float hL = tile.Height17[j, Math.Max(0, i - 1)];
                float hR = tile.Height17[j, Math.Min(16, i + 1)];
                float hU = tile.Height17[Math.Max(0, j - 1), i];
                float hD = tile.Height17[Math.Min(16, j + 1), i];
                // For Y-up, north-up: along +Z direction, j decreases ⇒ slope sign flips
                var dx = new Vector3((float)(2 * xyScale), +(hR - hL) * (float)heightScale, 0f);
                var dz = new Vector3(0f, -(hD - hU) * (float)heightScale, (float)(2 * xyScale));
                var n = Vector3.Cross(dz, dx);
                if (n.LengthSquared() < 1e-6f) n = new Vector3(0, 1, 0);
                else n = Vector3.Normalize(n);
                normals[j, i] = n;
            }
        }

        VertexBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty> V(int i, int j)
        {
            // Y-up height on XZ plane, north-up ⇒ z = -(j*scale)
            var pos = new Vector3((float)(i * xyScale), tile.Height17[j, i] * (float)heightScale, (float)(-j * xyScale));
            var nrm = normals[j, i];
            var uv = ComputeUv(i, j, xyScale, tileBaseX, tileBaseY, uvWidthWorld, uvHeightWorld, perTileUvs);
            var geo = new VertexPositionNormal(pos, nrm);
            VertexTexture1 mat = uv; // implicit conversion from Vector2
            return new VertexBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>(geo, mat);
        }

        for (int j = 0; j < 16; j++)
        {
            for (int i = 0; i < 16; i++)
            {
                if (skipHoles && tile.IsHole(j, i)) continue;
                var v00 = V(i, j);
                var v10 = V(i + 1, j);
                var v01 = V(i, j + 1);
                var v11 = V(i + 1, j + 1);
                // CCW from +Y view with Z flipped: reorder to keep top faces front-facing
                prim.AddTriangle(v00, v11, v10);
                prim.AddTriangle(v00, v01, v11);
                faces += 2;
            }
        }

        verticesApprox = 17 * 17;
        return mesh;
    }

    private static System.Numerics.Vector2 ComputeUv(int i, int j, double xyScale, float baseX, float baseY, float widthWorld, float heightWorld, bool perTile)
    {
        // baseX: world X origin of tile; baseY: world Z origin of tile (north-up means baseY is negative)
        float worldX = baseX + (float)(i * xyScale);
        float worldZ = baseY + (float)(j * xyScale); // baseY carries Z
        if (perTile)
        {
            // Normalize within a single tile footprint; flip V for north-up
            float uT = (widthWorld <= 0.0f) ? 0f : ((float)(i * xyScale) / widthWorld);
            float vT = (heightWorld <= 0.0f) ? 0f : (1f - ((float)(j * xyScale) / heightWorld));
            return new System.Numerics.Vector2(uT, vT);
        }
        // Normalize within full map footprint; flip V for north-up (Z increases to north)
        float u = (widthWorld <= 0.0f) ? 0f : (worldX / widthWorld);
        float v = (heightWorld <= 0.0f) ? 0f : (1f - ((-baseY + (float)(j * xyScale)) / heightWorld));
        return new System.Numerics.Vector2(u, v);
    }

    private static MaterialBuilder CreateTerrainMaterial(string? texturePath)
    {
        var mat = new MaterialBuilder("terrain").WithMetallicRoughnessShader();
        if (!string.IsNullOrWhiteSpace(texturePath) && File.Exists(texturePath))
        {
            // BaseColor texture with clamp-to-edge and mipmap settings to reduce seams and preserve detail
            var img = ImageBuilder.From(texturePath);
            var ch = mat.UseChannel(KnownChannel.BaseColor).UseTexture();
            ch.WithPrimaryImage(img);
            ch.WithSampler(TextureWrapMode.CLAMP_TO_EDGE, TextureWrapMode.CLAMP_TO_EDGE,
                           TextureMipMapFilter.LINEAR_MIPMAP_LINEAR,
                           TextureInterpolationFilter.LINEAR);
        }
        return mat;
    }

    private static MaterialBuilder CreateTerrainMaterialFromBytes(byte[] pngBytes, string? logicalName = null)
    {
        var mat = new MaterialBuilder("terrain").WithMetallicRoughnessShader();
        if (pngBytes != null && pngBytes.Length > 0)
        {
            var img = ImageBuilder.From(pngBytes);
            if (!string.IsNullOrWhiteSpace(logicalName)) img.AlternateWriteFileName = logicalName;
            var ch = mat.UseChannel(KnownChannel.BaseColor).UseTexture();
            ch.WithPrimaryImage(img);
            ch.WithSampler(TextureWrapMode.CLAMP_TO_EDGE, TextureWrapMode.CLAMP_TO_EDGE,
                           TextureMipMapFilter.LINEAR_MIPMAP_LINEAR,
                           TextureInterpolationFilter.LINEAR);
        }
        return mat;
    }

    // --- Minimap index & resolution ---
    private static readonly object _indexLock = new();
    private static string? _indexedRoot;
    private static Dictionary<(string map,int x,int y), string>? _mapIndexedPaths; // by map
    private static Dictionary<(int x,int y), string>? _tileIndexedPaths; // by tile_{x}_{y}

    private static void EnsureMinimapIndex(string root)
    {
        lock (_indexLock)
        {
            if (string.Equals(_indexedRoot, Path.GetFullPath(root), StringComparison.OrdinalIgnoreCase) && _mapIndexedPaths != null)
                return;

            _indexedRoot = Path.GetFullPath(root);
            _mapIndexedPaths = new();
            _tileIndexedPaths = new();

            if (!Directory.Exists(_indexedRoot)) return;

            var allowedExts = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".blp", ".png", ".jpg", ".jpeg", ".webp" };
            foreach (var file in Directory.EnumerateFiles(_indexedRoot, "*.*", SearchOption.AllDirectories))
            {
                var ext = Path.GetExtension(file);
                if (!allowedExts.Contains(ext)) continue;
                var name = Path.GetFileNameWithoutExtension(file);

                // Patterns: Map_X_Y or tile_X_Y
                if (TryParseMapXY(name, out var map, out var x, out var y))
                {
                    // normalize map to lower for consistent lookups
                    _mapIndexedPaths[(map!.ToLowerInvariant(), x, y)] = file;
                    continue;
                }
                if (TryParseTileXY(name, out x, out y))
                {
                    _tileIndexedPaths[(x, y)] = file;
                }
            }
        }
    }

    private static bool TryParseMapXY(string name, out string? map, out int x, out int y)
    {
        map = null; x = 0; y = 0;
        var m = Regex.Match(name, @"^(?<map>[A-Za-z0-9_]+)_(?<x>\d+)_(?<y>\d+)$");
        if (!m.Success) return false;
        map = m.Groups["map"].Value;
        return int.TryParse(m.Groups["x"].Value, out x) && int.TryParse(m.Groups["y"].Value, out y);
    }

    private static bool TryParseTileXY(string name, out int x, out int y)
    {
        x = 0; y = 0;
        var m = Regex.Match(name, @"^tile_(?<x>\d+)_(?<y>\d+)$", RegexOptions.IgnoreCase);
        if (!m.Success) return false;
        return int.TryParse(m.Groups["x"].Value, out x) && int.TryParse(m.Groups["y"].Value, out y);
    }

    private static bool TryResolveTileImage(string? mapName, string root, int x, int y, out string imagePath)
    {
        imagePath = string.Empty;
        if (_mapIndexedPaths is null || _tileIndexedPaths is null) return false;
        if (!string.IsNullOrWhiteSpace(mapName))
        {
            var key = (mapName!.ToLowerInvariant(), x, y);
            if (_mapIndexedPaths.TryGetValue(key, out var p)) { imagePath = p; return true; }
        }
        if (_tileIndexedPaths.TryGetValue((x, y), out var p2)) { imagePath = p2; return true; }
        return false;
    }

    private static byte[]? TryDecodeBlpToPng(string blpPath)
    {
        try
        {
            using var fs = File.OpenRead(blpPath);
            using var blp = new BLPFile(fs);
            var pixels = blp.GetPixels(0, out var w, out var h);
            using var img = SixLabors.ImageSharp.Image.LoadPixelData<Bgra32>(pixels, w, h);
            using var ms = new MemoryStream();
            img.Save(ms, new PngEncoder());
            return ms.ToArray();
        }
        catch { return null; }
    }

    /// <summary>
    /// Build a full-map mosaic PNG from TRS minimap tiles. Returns the PNG file path, or null on failure.
    /// North is up: row y maps to (63 - y) in the output image.
    /// </summary>
    public static string? BuildMinimapMosaicFromTrs(string outputDirectory, string outputStem, string? minimapRoot, string? trsPath, string? mapName, int tileSize = 256)
    {
        if (string.IsNullOrWhiteSpace(mapName)) return null;
        var index = TryBuildTrsIndex(minimapRoot, trsPath, out var info);
        if (index == null || info.Entries == 0) return null;

        var lowerMap = mapName!.ToLowerInvariant();
        int W = tileSize * 64;
        int H = tileSize * 64;
        try
        {
            using var canvas = new Image<Bgra32>(W, H);
            for (int y = 0; y < 64; y++)
            {
                for (int x = 0; x < 64; x++)
                {
                    if (!index.TryGetValue((lowerMap, x, y), out var blp)) continue;
                    if (!File.Exists(blp)) continue;
                    using var tile = DecodeBlpImage(blp);
                    if (tile == null) continue;
                    // Resize if needed
                    if (tile.Width != tileSize || tile.Height != tileSize)
                    {
                        tile.Mutate(ctx => ctx.Resize(tileSize, tileSize));
                    }
                    int dx = x * tileSize;
                    int dy = (63 - y) * tileSize; // north-up
                    canvas.Mutate(ctx => ctx.DrawImage(tile, new Point(dx, dy), 1.0f));
                }
            }
            Directory.CreateDirectory(outputDirectory);
            var outPng = Path.Combine(outputDirectory, outputStem + "_mosaic.png");
            canvas.Save(outPng, new PngEncoder());
            return outPng;
        }
        catch { return null; }
    }

    private static Image<Bgra32>? DecodeBlpImage(string blpPath)
    {
        try
        {
            using var fs = File.OpenRead(blpPath);
            using var blp = new BLPFile(fs);
            var pixels = blp.GetPixels(0, out var w, out var h);
            return SixLabors.ImageSharp.Image.LoadPixelData<Bgra32>(pixels, w, h);
        }
        catch { return null; }
    }

    // --- TRS (md5translate) parsing ---
    private static Dictionary<(string map,int x,int y), string>? TryBuildTrsIndex(string? minimapRoot, string? trsPath, out (bool Used,int Entries) info)
    {
        info = (false, 0);
        try
        {
            string? root = string.IsNullOrWhiteSpace(minimapRoot) ? null : Path.GetFullPath(minimapRoot!);
            string? trs = trsPath;
            if (string.IsNullOrWhiteSpace(trs) && !string.IsNullOrWhiteSpace(root))
            {
                var p1 = Path.Combine(root!, "md5translate.trs");
                var p2 = Path.Combine(root!, "md5translate.txt");
                trs = File.Exists(p1) ? p1 : (File.Exists(p2) ? p2 : null);
            }
            if (string.IsNullOrWhiteSpace(trs) || !File.Exists(trs)) return null;

            var dict = new Dictionary<(string,int,int), string>();
            string baseDir = Path.GetDirectoryName(trs)!;
            string? currentMap = null;
            int entries = 0;
            foreach (var raw in File.ReadAllLines(trs))
            {
                var line = raw.Trim();
                if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#")) continue;
                if (line.StartsWith("dir:", StringComparison.OrdinalIgnoreCase))
                {
                    currentMap = line.Substring(4).Trim();
                    continue;
                }
                if (currentMap == null) continue;
                var parts = line.Split('\t');
                if (parts.Length != 2) continue;
                string a = parts[0].Trim();
                string b = parts[1].Trim();
                string mapSide;
                string actualSide;
                if (a.Contains("map") && a.Contains(".blp", StringComparison.OrdinalIgnoreCase)) { mapSide = a; actualSide = b; }
                else { mapSide = b; actualSide = a; }

                var stem = Path.GetFileNameWithoutExtension(mapSide);
                if (!stem.StartsWith("map", StringComparison.OrdinalIgnoreCase)) continue;
                var xy = stem.Substring(3).Split('_');
                if (xy.Length != 2 || !int.TryParse(xy[0], out var tx) || !int.TryParse(xy[1], out var ty)) continue;

                // actualSide is relative to textures/minimap from TRS location
                string fullPath = Path.Combine(baseDir, actualSide.Replace('/', Path.DirectorySeparatorChar));
                dict[(currentMap.ToLowerInvariant(), tx, ty)] = fullPath;
                entries++;
            }
            info = (true, entries);
            return dict;
        }
        catch
        {
            info = (true, 0);
            return null;
        }
    }
}
