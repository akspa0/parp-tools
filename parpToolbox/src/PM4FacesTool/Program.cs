using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.PM4;

namespace PM4FacesTool;

internal static class Program
{
    private sealed record Options(
        string Input,
        string OutDir,
        bool LegacyParity,
        bool ProjectLocal,
        bool SnapToPlane,
        float HeightScale,
        bool HeightFitReport,
        string GroupBy,
        bool Batch,
        bool CkUseMslk,
        double CkAllowUnlinkedRatio,
        int CkMinTris,
        bool CkMergeComponents,
        bool CkMonolithic,
        bool ExportGltf,
        bool ExportGlb,
        bool NoMscnRemap
    );

    public static int Main(string[] args)
    {
        try
        {
            var opts = ParseArgs(args);
            if (opts == null)
            {
                PrintHelp();
                return 2;
            }

            Directory.CreateDirectory(opts.OutDir);

            if (opts.Batch && Directory.Exists(opts.Input))
            {
                var pm4s = Directory.EnumerateFiles(opts.Input, "*.pm4", SearchOption.TopDirectoryOnly)
                    .OrderBy(f => f)
                    .ToList();

                if (pm4s.Count == 0)
                {
                    Console.WriteLine("No .pm4 files found for --batch input directory.");
                    return 1;
                }

                foreach (var firstTile in pm4s)
                {
                    Console.WriteLine($"[pm4-faces] Processing batch start tile: {firstTile}");
                    ProcessOne(firstTile, opts);
                }
            }
            else
            {
                ProcessOne(opts.Input, opts);
            }

            Console.WriteLine("[pm4-faces] Done.");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("[pm4-faces] ERROR: " + ex);
            return 1;
        }
    }

    private static void ExportTypeAttrInstances(
        Pm4Scene scene,
        string objectsDir,
        Options opts,
        string outDir,
        List<string> coverageCsv,
        List<ObjectIndexEntry> objectIndex)
    {
        var groups = scene.Surfaces
            .Select((e, i) => (e, i))
            .GroupBy(x => new { x.e.GroupKey, x.e.AttributeMask, Ck24 = CK24(x.e.CompositeKey) })
            .OrderBy(g => g.Key.GroupKey)
            .ThenBy(g => g.Key.AttributeMask)
            .ThenBy(g => g.Key.Ck24);

        foreach (var g in groups)
        {
            var surfaceSceneIndices = g.Select(x => x.i).ToList();
            int n = surfaceSceneIndices.Count;
            if (n == 0) continue;

            var groupDir = objectsDir; // flatten: no TYPE/ATTR subfolders
            Directory.CreateDirectory(groupDir);

            // Build vertex -> list(local surface indices) map within this type+attr bucket
            var vertToSurfs = new Dictionary<int, List<int>>();
            for (int i = 0; i < n; i++)
            {
                int sIdx = surfaceSceneIndices[i];
                var surf = scene.Surfaces[sIdx];
                if (surf.MsviFirstIndex > int.MaxValue) continue;
                int first = unchecked((int)surf.MsviFirstIndex);
                int count = surf.IndexCount;
                if (first < 0 || count < 3 || first + count > scene.Indices.Count) continue;

                var used = new HashSet<int>();
                for (int k = 0; k < count; k++)
                {
                    int v = scene.Indices[first + k];
                    if (v < 0 || v >= scene.Vertices.Count) continue;
                    if (!used.Add(v)) continue;
                    if (!vertToSurfs.TryGetValue(v, out var list)) { list = new List<int>(); vertToSurfs[v] = list; }
                    list.Add(i); // local index inside this TYPE+ATTR group
                }
            }

            // DSU on local indices for this type+attr
            var dsu = new DSU(n);
            foreach (var kv in vertToSurfs)
            {
                var list = kv.Value;
                if (list.Count <= 1) continue;
                int root = list[0];
                for (int i = 1; i < list.Count; i++) dsu.Union(root, list[i]);
            }

            // Gather components as lists of scene-surface-indices
            var compToSurfaces = new Dictionary<int, List<int>>();
            for (int i = 0; i < n; i++)
            {
                int r = dsu.Find(i);
                if (!compToSurfaces.TryGetValue(r, out var lst)) { lst = new List<int>(); compToSurfaces[r] = lst; }
                lst.Add(surfaceSceneIndices[i]);
            }

            // Export each component
            int compIndex = 0;
            foreach (var comp in compToSurfaces.Values)
            {
                var items = comp.Select(si => (scene.Surfaces[si], si)).ToList();
                // Determine dominant tile for this component (optional)
                int? domTile = DominantTileIdFor(scene, items.Select(it => it.Item1));
                string tileSuffix = domTile.HasValue ? $"_t{domTile.Value % 64:D2}_{domTile.Value / 64:D2}" : string.Empty;
                string name = $"attr_{g.Key.AttributeMask:D3}_type_{g.Key.GroupKey:D3}_ck{g.Key.Ck24:X6}_inst_{compIndex:D5}{tileSuffix}";
                string safe = SanitizeFileName(name);
                // Bucket by tile subfolder
                string folder = domTile.HasValue ? $"t{domTile.Value % 64:D2}_{domTile.Value / 64:D2}" : "t_unknown";
                // Then bucket by dominant object id
                int domObj = DominantObjectIdFor(items.Select(it => it.Item1));
                string objFolder = domObj > 0 ? $"obj_{domObj:D5}" : "obj_unknown";
                string objDir = Path.Combine(objectsDir, folder, objFolder);
                Directory.CreateDirectory(objDir);
                string objPath = Path.Combine(objDir, safe + ".obj");

                // Optional filtering to reduce tiny instance outputs
                int triEstimate = items.Sum(it => (int)it.Item1.IndexCount) / 3;
                bool skipWrite = opts.CkMinTris > 0 && triEstimate < opts.CkMinTris;

                int writtenFaces = 0;
                int skippedFaces = 0;
                if (!skipWrite)
                {
                    AssembleAndWrite(scene, items, objPath, opts with { SnapToPlane = false }, out writtenFaces, out skippedFaces);
                }

                // Coverage row for quick visibility
                coverageCsv.Add(string.Join(',',
                    $"group:{name}",
                    items.First().Item1.GroupKey,
                    items.First().Item1.CompositeKey,
                    items.Min(it => (int)it.Item1.MsviFirstIndex) + "-" + items.Max(it => (int)it.Item1.MsviFirstIndex),
                    items.Sum(it => (int)it.Item1.IndexCount),
                    writtenFaces,
                    skippedFaces,
                    (skipWrite ? string.Empty : EscapeCsv(Path.GetRelativePath(Path.Combine(objectsDir, ".."), objPath)))));

                // Add to object index if not skipped
                if (!skipWrite)
                {
                    objectIndex.Add(new ObjectIndexEntry
                    {
                        Id = $"typeattr:{g.Key.GroupKey:D3}:{g.Key.AttributeMask:D3}:ck24:{g.Key.Ck24:X6}:inst:{compIndex:D5}",
                        Name = name,
                        Group = "type-attr-instance",
                        ObjPath = Path.GetRelativePath(outDir, objPath).Replace("\\", "/"),
                        GltfPath = opts.ExportGltf ? Path.GetRelativePath(outDir, Path.ChangeExtension(objPath, ".gltf")).Replace("\\", "/") : null,
                        GlbPath = opts.ExportGlb ? Path.GetRelativePath(outDir, Path.ChangeExtension(objPath, ".glb")).Replace("\\", "/") : null,
                        FacesWritten = writtenFaces,
                        FacesSkipped = skippedFaces,
                        SurfaceIndices = items.Select(t => t.si).ToList(),
                        IndexFirst = items.Min(it => (int)it.Item1.MsviFirstIndex),
                        IndexCount = items.Sum(it => (int)it.Item1.IndexCount),
                        FlipX = true,
                        IsWalkable = (g.Key.GroupKey == 16 && g.Key.AttributeMask == 2),
                        IsM2 = (g.Key.GroupKey == 3 && g.Key.AttributeMask == 1 && g.Key.Ck24 == 0),
                        Ck24 = g.Key.Ck24
                    });
                }

                compIndex++;
            }
        }
    }

    private static void ProcessOne(string firstTilePath, Options opts)
    {
        if (string.IsNullOrWhiteSpace(firstTilePath) || !File.Exists(firstTilePath))
            throw new FileNotFoundException("Input PM4 file not found", firstTilePath);

        // Load using Global Tile Loader. In single-file mode, load ONLY that tile.
        var dir = Path.GetDirectoryName(firstTilePath) ?? Environment.CurrentDirectory;
        var baseName = Path.GetFileNameWithoutExtension(firstTilePath);
        var parts = baseName.Split('_');
        var prefix = parts.Length > 2 ? string.Join('_', parts.Take(parts.Length - 2)) : baseName;
        string pattern = opts.Batch
            ? (prefix + "_*.pm4")
            : Path.GetFileName(firstTilePath);

        if (!opts.Batch)
        {
            Console.WriteLine($"[pm4-faces] Single-file mode: loading only tile '{pattern}'");
        }

        var global = Pm4GlobalTileLoader.LoadRegion(dir, pattern, !opts.NoMscnRemap);
        var scene = Pm4GlobalTileLoader.ToStandardScene(global);

        var outDir = Path.Combine(opts.OutDir, SessionNameFrom(firstTilePath));
        Directory.CreateDirectory(outDir);
        var objectsDir = Path.Combine(outDir, "objects");
        Directory.CreateDirectory(objectsDir);
        var tilesDir = Path.Combine(outDir, "tiles");
        Directory.CreateDirectory(tilesDir);

        // Emit lightweight diagnostics for MSUR.Height variability
        EmitMsurHeightDiagnostics(scene, outDir);
        if (opts.HeightFitReport)
        {
            EmitHeightFitReport(scene, outDir);
        }

        Console.WriteLine($"[pm4-faces] Export strategy: objects={opts.GroupBy}, tiles=on");
        Console.WriteLine($"[pm4-faces] Scene: Vertices={scene.Vertices.Count}, Indices={scene.Indices.Count}, Surfaces={scene.Surfaces.Count}, Tiles={scene.TileIndexOffsetByTileId.Count}");

        var coverageCsv = new List<string>
        {
            "surface_index,group_key,composite_key,first_index,index_count,faces_written,faces_skipped,obj_path"
        };
        var tileCsv = new List<string>
        {
            "tile_id,start_index,index_count,faces_written,faces_skipped,obj_path"
        };

        // JSON indexes for objects and tiles
        var objectIndex = new List<ObjectIndexEntry>();
        var tileIndex = new List<TileIndexEntry>();

        // Group selection
        IEnumerable<(MsurChunk.Entry entry, int index)> surfaces = scene.Surfaces
            .Select((e, i) => (e, i));

        if (opts.GroupBy.Equals("composite-instance", StringComparison.OrdinalIgnoreCase))
        {
            ExportCompositeInstances(scene, objectsDir, opts, outDir, coverageCsv, objectIndex);
        }
        else if (opts.GroupBy.Equals("type-instance", StringComparison.OrdinalIgnoreCase))
        {
            ExportTypeInstances(scene, objectsDir, opts, outDir, coverageCsv, objectIndex);
        }
        else if (opts.GroupBy.Equals("type-attr-instance", StringComparison.OrdinalIgnoreCase))
        {
            ExportTypeAttrInstances(scene, objectsDir, opts, outDir, coverageCsv, objectIndex);
        }
        else if (opts.GroupBy.Equals("groupkey", StringComparison.OrdinalIgnoreCase) ||
                 opts.GroupBy.Equals("type", StringComparison.OrdinalIgnoreCase))
        {
            var grouped = scene.Surfaces
                .Select((e, i) => (e, i))
                .GroupBy(x => x.e.GroupKey)
                .OrderBy(g => g.Key);

            foreach (var g in grouped)
            {
                var name = $"groupkey_{g.Key:D3}";
                ExportGroup(scene, g.ToList(), name, objectsDir, opts, coverageCsv, objectIndex);
            }
        }
        else if (opts.GroupBy.Equals("composite", StringComparison.OrdinalIgnoreCase))
        {
            var grouped = scene.Surfaces
                .Select((e, i) => (e, i))
                .GroupBy(x => x.e.CompositeKey)
                .OrderBy(g => g.Key);

            foreach (var g in grouped)
            {
                var name = $"ck_{g.Key:X8}";
                ExportGroup(scene, g.ToList(), name, objectsDir, opts, coverageCsv, objectIndex);
            }
        }
        else if (opts.GroupBy.Equals("surface", StringComparison.OrdinalIgnoreCase))
        {
            // Per-surface
            foreach (var (entry, idx) in surfaces)
            {
                var safe = SanitizeFileName($"surface_{idx:D6}_g{entry.GroupKey:D3}_ck{entry.CompositeKey:X8}");
                var objPath = Path.Combine(objectsDir, safe + ".obj");
                AssembleAndWrite(scene, new[] {(entry, idx)}, objPath, opts, out var written, out var skipped);
                coverageCsv.Add(string.Join(',',
                    idx,
                    entry.GroupKey,
                    entry.CompositeKey,
                    entry.MsviFirstIndex,
                    entry.IndexCount,
                    written,
                    skipped,
                    EscapeCsv(Path.GetRelativePath(outDir, objPath))));

                // Add to object index
                objectIndex.Add(new ObjectIndexEntry
                {
                    Id = $"surface:{idx:D6}",
                    Name = safe,
                    Group = "surface",
                    ObjPath = Path.GetRelativePath(outDir, objPath).Replace("\\", "/"),
                    GltfPath = opts.ExportGltf ? Path.GetRelativePath(outDir, Path.ChangeExtension(objPath, ".gltf")).Replace("\\", "/") : null,
                    GlbPath = opts.ExportGlb ? Path.GetRelativePath(outDir, Path.ChangeExtension(objPath, ".glb")).Replace("\\", "/") : null,
                    FacesWritten = written,
                    FacesSkipped = skipped,
                    SurfaceIndices = new List<int> { idx },
                    IndexFirst = (int)entry.MsviFirstIndex,
                    IndexCount = entry.IndexCount,
                    FlipX = true
                });
            }
        }
        else if (opts.GroupBy.Equals("render-mesh", StringComparison.OrdinalIgnoreCase) ||
                 opts.GroupBy.Equals("surfaces-all", StringComparison.OrdinalIgnoreCase))
        {
            // Skip object exports; rely solely on tile exports below.
        }
        else
        {
            // Default to composite-instance if an unknown group was provided
            Console.WriteLine($"[pm4-faces] Unknown --group-by '{opts.GroupBy}', defaulting to composite-instance");
            ExportCompositeInstances(scene, objectsDir, opts, outDir, coverageCsv, objectIndex);
        }

        File.WriteAllLines(Path.Combine(outDir, "surface_coverage.csv"), coverageCsv);

        // Always export tiles as a separate set; preserve original prefix and XY suffix
        ExportTiles(scene, tilesDir, opts, outDir, tileCsv, tileIndex, prefix);
        File.WriteAllLines(Path.Combine(outDir, "tile_coverage.csv"), tileCsv);

        // Walkable summary (type=016 & attr=002) by tile folder and CK24
        var walkable = objectIndex.Where(o => o.Group == "type-attr-instance" && o.IsWalkable);
        var walkSummary = new Dictionary<(string Tile, string Ck24), (int Count, int Faces)>();
        foreach (var o in walkable)
        {
            string pathNorm = (o.ObjPath ?? string.Empty).Replace('\\', '/');
            string tile = "t_unknown";
            var pathParts = pathNorm.Split('/', StringSplitOptions.RemoveEmptyEntries);
            foreach (var p in pathParts)
            {
                if (p.Length == 6 && p[0] == 't' && char.IsDigit(p[1]) && char.IsDigit(p[2]) && p[3] == '_' && char.IsDigit(p[4]) && char.IsDigit(p[5]))
                {
                    tile = p; break;
                }
            }
            string ck = $"{o.Ck24:X6}";
            var key = (tile, ck);
            if (!walkSummary.TryGetValue(key, out var agg)) agg = (0, 0);
            walkSummary[key] = (agg.Count + 1, agg.Faces + o.FacesWritten);
        }
        var walkCsv = new List<string> { "tile,ck24,objects,faces" };
        foreach (var kv in walkSummary.OrderBy(k => k.Key.Tile).ThenBy(k => k.Key.Ck24))
        {
            walkCsv.Add(string.Join(',', kv.Key.Tile, $"0x{kv.Key.Ck24}", kv.Value.Count, kv.Value.Faces));
        }
        File.WriteAllLines(Path.Combine(outDir, "walkable_coverage.csv"), walkCsv);

        // M2 summary (type=003 & attr=001 & ck24=0) by tile folder and CK24
        var m2 = objectIndex.Where(o => o.Group == "type-attr-instance" && o.IsM2);
        var m2Summary = new Dictionary<(string Tile, string Ck24), (int Count, int Faces)>();
        foreach (var o in m2)
        {
            string pathNorm = (o.ObjPath ?? string.Empty).Replace('\\', '/');
            string tile = "t_unknown";
            var pathParts = pathNorm.Split('/', StringSplitOptions.RemoveEmptyEntries);
            foreach (var p in pathParts)
            {
                if (p.Length == 6 && p[0] == 't' && char.IsDigit(p[1]) && char.IsDigit(p[2]) && p[3] == '_' && char.IsDigit(p[4]) && char.IsDigit(p[5]))
                {
                    tile = p; break;
                }
            }
            string ck = $"{o.Ck24:X6}";
            var key = (tile, ck);
            if (!m2Summary.TryGetValue(key, out var agg)) agg = (0, 0);
            m2Summary[key] = (agg.Count + 1, agg.Faces + o.FacesWritten);
        }
        var m2Csv = new List<string> { "tile,ck24,objects,faces" };
        foreach (var kv in m2Summary.OrderBy(k => k.Key.Tile).ThenBy(k => k.Key.Ck24))
        {
            m2Csv.Add(string.Join(',', kv.Key.Tile, $"0x{kv.Key.Ck24}", kv.Value.Count, kv.Value.Faces));
        }
        File.WriteAllLines(Path.Combine(outDir, "m2_coverage.csv"), m2Csv);

        // Write JSON indexes
        var jsonOpts = new JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(Path.Combine(outDir, "objects_index.json"), JsonSerializer.Serialize(objectIndex, jsonOpts));
        File.WriteAllText(Path.Combine(outDir, "tiles_index.json"), JsonSerializer.Serialize(tileIndex, jsonOpts));
    }

    private static void ExportCompositeInstances(
        Pm4Scene scene,
        string objectsDir,
        Options opts,
        string outDir,
        List<string> coverageCsv,
        List<ObjectIndexEntry> objectIndex)
    {
        // Instance diagnostics
        var instCsv = new List<string> { "ck24,instance_id,surface_count,faces_written,faces_skipped,obj_path" };
        var membersCsv = new List<string> { "ck24,instance_id,surface_scene_index,msvi_first,index_count" };

        static uint CK24(uint key) => (key & 0xFFFFFF00u) >> 8;
        var groups = scene.Surfaces
            .Select((e, i) => (e, i))
            .GroupBy(x => CK24(x.e.CompositeKey))
            .OrderBy(g => g.Key);

        foreach (var g in groups)
        {
            // map scene-surface-index -> local index inside this CK group
            var surfaceSceneIndices = g.Select(x => x.i).ToList();
            int n = surfaceSceneIndices.Count;
            if (n == 0) continue;

            // Flatten: do not create per-CK subfolders; we will bucket by dominant tile folder later

            // If requested, export a single merged (monolithic) OBJ per CK24 and skip per-component outputs
            if (opts.CkMonolithic)
            {
                var allItems = g.Select(x => (scene.Surfaces[x.i], x.i)).ToList();
                // Place merged object in dominant tile folder; no per-CK subfolder
                int? domTile = DominantTileIdFor(scene, allItems.Select(it => it.Item1));
                string folder = domTile.HasValue ? $"t{domTile.Value % 64:D2}_{domTile.Value / 64:D2}" : "t_unknown";
                Directory.CreateDirectory(Path.Combine(objectsDir, folder));
                string name = $"ck{g.Key:X6}_merged";
                string safe = SanitizeFileName(name);
                string objPath = Path.Combine(objectsDir, folder, safe + ".obj");

                // Optional filtering (estimate triangles from MSUR.IndexCount)
                int triEstimate = allItems.Sum(it => (int)it.Item1.IndexCount) / 3;
                bool skipWrite = opts.CkMinTris > 0 && triEstimate < opts.CkMinTris;

                int writtenFaces = 0;
                int skippedFaces = 0;
                if (!skipWrite)
                {
                    AssembleAndWrite(scene, allItems, objPath, opts with { SnapToPlane = false }, out writtenFaces, out skippedFaces);
                }

                // Log one instance row and all membership rows
                instCsv.Add(string.Join(',',
                    $"0x{g.Key:X6}",
                    0,
                    allItems.Count,
                    writtenFaces,
                    skippedFaces,
                    (skipWrite ? string.Empty : EscapeCsv(Path.GetRelativePath(outDir, objPath)))));

                foreach (var (entry, idx) in allItems)
                {
                    membersCsv.Add(string.Join(',',
                        $"0x{g.Key:X6}",
                        0,
                        idx,
                        entry.MsviFirstIndex,
                        entry.IndexCount));
                }

                // Coverage row
                coverageCsv.Add(string.Join(',',
                    $"group:{name}",
                    allItems.First().Item1.GroupKey,
                    allItems.First().Item1.CompositeKey,
                    allItems.Min(it => (int)it.Item1.MsviFirstIndex) + "-" + allItems.Max(it => (int)it.Item1.MsviFirstIndex),
                    allItems.Sum(it => (int)it.Item1.IndexCount),
                    writtenFaces,
                    skippedFaces,
                    (skipWrite ? string.Empty : EscapeCsv(Path.GetRelativePath(Path.Combine(objectsDir, ".."), objPath)))));

                // Add to object index if not skipped
                if (!skipWrite)
                {
                    objectIndex.Add(new ObjectIndexEntry
                    {
                        Id = $"ck24:{g.Key:X6}:merged",
                        Name = name,
                        Group = "composite-monolithic",
                        ObjPath = Path.GetRelativePath(outDir, objPath).Replace("\\", "/"),
                        GltfPath = opts.ExportGltf ? Path.GetRelativePath(outDir, Path.ChangeExtension(objPath, ".gltf")).Replace("\\", "/") : null,
                        GlbPath = opts.ExportGlb ? Path.GetRelativePath(outDir, Path.ChangeExtension(objPath, ".glb")).Replace("\\", "/") : null,
                        FacesWritten = writtenFaces,
                        FacesSkipped = skippedFaces,
                        SurfaceIndices = allItems.Select(t => t.i).ToList(),
                        IndexFirst = allItems.Min(it => (int)it.Item1.MsviFirstIndex),
                        IndexCount = allItems.Sum(it => (int)it.Item1.IndexCount),
                        FlipX = true
                    });
                }

                continue; // skip per-component path for this CK24
            }

            // Build vertex -> list(local surface indices) map
            var vertToSurfs = new Dictionary<int, List<int>>();
            for (int i = 0; i < n; i++)
            {
                int sIdx = surfaceSceneIndices[i];
                var surf = scene.Surfaces[sIdx];
                if (surf.MsviFirstIndex > int.MaxValue) continue;
                int first = unchecked((int)surf.MsviFirstIndex);
                int count = surf.IndexCount;
                if (first < 0 || count < 3 || first + count > scene.Indices.Count) continue;

                var used = new HashSet<int>();
                for (int k = 0; k < count; k++)
                {
                    int v = scene.Indices[first + k];
                    if (v < 0 || v >= scene.Vertices.Count) continue;
                    if (!used.Add(v)) continue;
                    if (!vertToSurfs.TryGetValue(v, out var list)) { list = new List<int>(); vertToSurfs[v] = list; }
                    list.Add(i); // local index inside this CK group
                }
            }

            // DSU on local indices
            var dsu = new DSU(n);
            foreach (var kv in vertToSurfs)
            {
                var list = kv.Value;
                if (list.Count <= 1) continue;
                int root = list[0];
                for (int i = 1; i < list.Count; i++) dsu.Union(root, list[i]);
            }

            // Gather components as lists of scene-surface-indices
            var compToSurfaces = new Dictionary<int, List<int>>();
            for (int i = 0; i < n; i++)
            {
                int r = dsu.Find(i);
                if (!compToSurfaces.TryGetValue(r, out var lst)) { lst = new List<int>(); compToSurfaces[r] = lst; }
                lst.Add(surfaceSceneIndices[i]);
            }

            // Export each component
            int compIndex = 0;
            foreach (var comp in compToSurfaces.Values)
            {
                var items = comp.Select(si => (scene.Surfaces[si], si)).ToList();
                // Place instance object in dominant tile folder; no per-CK subfolder
                int? domTile = DominantTileIdFor(scene, items.Select(it => it.Item1));
                string folder = domTile.HasValue ? $"t{domTile.Value % 64:D2}_{domTile.Value / 64:D2}" : "t_unknown";
                Directory.CreateDirectory(Path.Combine(objectsDir, folder));
                string name = $"ck{g.Key:X6}_inst_{compIndex:D5}";
                string safe = SanitizeFileName(name);
                string objPath = Path.Combine(objectsDir, folder, safe + ".obj");

                // Optional filtering to reduce tiny instance outputs
                int triEstimate = items.Sum(it => (int)it.Item1.IndexCount) / 3;
                bool skipWrite = opts.CkMinTris > 0 && triEstimate < opts.CkMinTris;

                int writtenFaces = 0;
                int skippedFaces = 0;
                if (!skipWrite)
                {
                    AssembleAndWrite(scene, items, objPath, opts with { SnapToPlane = false }, out writtenFaces, out skippedFaces);
                }

                // Log instance row and membership rows
                instCsv.Add(string.Join(',',
                    $"0x{g.Key:X6}",
                    compIndex,
                    items.Count,
                    writtenFaces,
                    skippedFaces,
                    (skipWrite ? string.Empty : EscapeCsv(Path.GetRelativePath(outDir, objPath)))));

                foreach (var (entry, idx) in items)
                {
                    membersCsv.Add(string.Join(',',
                        $"0x{g.Key:X6}",
                        compIndex,
                        idx,
                        entry.MsviFirstIndex,
                        entry.IndexCount));
                }

                // Add a coverage row too for quick visibility
                coverageCsv.Add(string.Join(',',
                    $"group:{name}",
                    items.First().Item1.GroupKey,
                    items.First().Item1.CompositeKey,
                    items.Min(it => (int)it.Item1.MsviFirstIndex) + "-" + items.Max(it => (int)it.Item1.MsviFirstIndex),
                    items.Sum(it => (int)it.Item1.IndexCount),
                    writtenFaces,
                    skippedFaces,
                    (skipWrite ? string.Empty : EscapeCsv(Path.GetRelativePath(Path.Combine(objectsDir, ".."), objPath)))));

                compIndex++;

                // Add to object index if not skipped
                if (!skipWrite)
                {
                    objectIndex.Add(new ObjectIndexEntry
                    {
                        Id = $"ck24:{g.Key:X6}:inst:{compIndex - 1:D5}",
                        Name = name,
                        Group = "composite-instance",
                        ObjPath = Path.GetRelativePath(outDir, objPath).Replace("\\", "/"),
                        GltfPath = opts.ExportGltf ? Path.GetRelativePath(outDir, Path.ChangeExtension(objPath, ".gltf")).Replace("\\", "/") : null,
                        GlbPath = opts.ExportGlb ? Path.GetRelativePath(outDir, Path.ChangeExtension(objPath, ".glb")).Replace("\\", "/") : null,
                        FacesWritten = writtenFaces,
                        FacesSkipped = skippedFaces,
                        SurfaceIndices = items.Select(t => t.si).ToList(),
                        IndexFirst = items.Min(it => (int)it.Item1.MsviFirstIndex),
                        IndexCount = items.Sum(it => (int)it.Item1.IndexCount),
                        FlipX = true
                    });
                }
            }
        }

        File.WriteAllLines(Path.Combine(outDir, "ck_instances.csv"), instCsv);
        File.WriteAllLines(Path.Combine(outDir, "instance_members.csv"), membersCsv);
    }

    private static void ExportTypeInstances(
        Pm4Scene scene,
        string objectsDir,
        Options opts,
        string outDir,
        List<string> coverageCsv,
        List<ObjectIndexEntry> objectIndex)
    {
        var groups = scene.Surfaces
            .Select((e, i) => (e, i))
            .GroupBy(x => new { x.e.GroupKey, Ck24 = CK24(x.e.CompositeKey) })
            .OrderBy(g => g.Key.GroupKey)
            .ThenBy(g => g.Key.Ck24);

        foreach (var g in groups)
        {
            var surfaceSceneIndices = g.Select(x => x.i).ToList();
            int n = surfaceSceneIndices.Count;
            if (n == 0) continue;

            var groupDir = objectsDir; // flatten: no TYPE subfolders
            Directory.CreateDirectory(groupDir);

            // Build vertex -> list(local surface indices) map within this type
            var vertToSurfs = new Dictionary<int, List<int>>();
            for (int i = 0; i < n; i++)
            {
                int sIdx = surfaceSceneIndices[i];
                var surf = scene.Surfaces[sIdx];
                if (surf.MsviFirstIndex > int.MaxValue) continue;
                int first = unchecked((int)surf.MsviFirstIndex);
                int count = surf.IndexCount;
                if (first < 0 || count < 3 || first + count > scene.Indices.Count) continue;

                var used = new HashSet<int>();
                for (int k = 0; k < count; k++)
                {
                    int v = scene.Indices[first + k];
                    if (v < 0 || v >= scene.Vertices.Count) continue;
                    if (!used.Add(v)) continue;
                    if (!vertToSurfs.TryGetValue(v, out var list)) { list = new List<int>(); vertToSurfs[v] = list; }
                    list.Add(i); // local index inside this TYPE group
                }
            }

            // DSU on local indices for this type
            var dsu = new DSU(n);
            foreach (var kv in vertToSurfs)
            {
                var list = kv.Value;
                if (list.Count <= 1) continue;
                int root = list[0];
                for (int i = 1; i < list.Count; i++) dsu.Union(root, list[i]);
            }

            // Gather components as lists of scene-surface-indices
            var compToSurfaces = new Dictionary<int, List<int>>();
            for (int i = 0; i < n; i++)
            {
                int r = dsu.Find(i);
                if (!compToSurfaces.TryGetValue(r, out var lst)) { lst = new List<int>(); compToSurfaces[r] = lst; }
                lst.Add(surfaceSceneIndices[i]);
            }

            // Export each component
            int compIndex = 0;
            foreach (var comp in compToSurfaces.Values)
            {
                var items = comp.Select(si => (scene.Surfaces[si], si)).ToList();
                // Determine dominant tile for this component (optional)
                int? domTile = DominantTileIdFor(scene, items.Select(it => it.Item1));
                string tileSuffix = domTile.HasValue ? $"_t{domTile.Value % 64:D2}_{domTile.Value / 64:D2}" : string.Empty;
                string name = $"type_{g.Key.GroupKey:D3}_ck{g.Key.Ck24:X6}_inst_{compIndex:D5}{tileSuffix}";
                string safe = SanitizeFileName(name);
                // Bucket by tile subfolder
                string folder = domTile.HasValue ? $"t{domTile.Value % 64:D2}_{domTile.Value / 64:D2}" : "t_unknown";
                // Then bucket by dominant object id
                int domObj = DominantObjectIdFor(items.Select(it => it.Item1));
                string objFolder = domObj > 0 ? $"obj_{domObj:D5}" : "obj_unknown";
                string objDir = Path.Combine(objectsDir, folder, objFolder);
                Directory.CreateDirectory(objDir);
                string objPath = Path.Combine(objDir, safe + ".obj");

                // Optional filtering to reduce tiny instance outputs
                int triEstimate = items.Sum(it => (int)it.Item1.IndexCount) / 3;
                bool skipWrite = opts.CkMinTris > 0 && triEstimate < opts.CkMinTris;

                int writtenFaces = 0;
                int skippedFaces = 0;
                if (!skipWrite)
                {
                    AssembleAndWrite(scene, items, objPath, opts with { SnapToPlane = false }, out writtenFaces, out skippedFaces);
                }

                // Coverage row for quick visibility
                coverageCsv.Add(string.Join(',',
                    $"group:{name}",
                    items.First().Item1.GroupKey,
                    items.First().Item1.CompositeKey,
                    items.Min(it => (int)it.Item1.MsviFirstIndex) + "-" + items.Max(it => (int)it.Item1.MsviFirstIndex),
                    items.Sum(it => (int)it.Item1.IndexCount),
                    writtenFaces,
                    skippedFaces,
                    (skipWrite ? string.Empty : EscapeCsv(Path.GetRelativePath(Path.Combine(objectsDir, ".."), objPath)))));

                // Add to object index if not skipped
                if (!skipWrite)
                {
                    objectIndex.Add(new ObjectIndexEntry
                    {
                        Id = $"type:{g.Key.GroupKey:D3}:ck24:{g.Key.Ck24:X6}:inst:{compIndex:D5}",
                        Name = name,
                        Group = "type-instance",
                        ObjPath = Path.GetRelativePath(outDir, objPath).Replace("\\", "/"),
                        GltfPath = opts.ExportGltf ? Path.GetRelativePath(outDir, Path.ChangeExtension(objPath, ".gltf")).Replace("\\", "/") : null,
                        GlbPath = opts.ExportGlb ? Path.GetRelativePath(outDir, Path.ChangeExtension(objPath, ".glb")).Replace("\\", "/") : null,
                        FacesWritten = writtenFaces,
                        FacesSkipped = skippedFaces,
                        SurfaceIndices = items.Select(t => t.si).ToList(),
                        IndexFirst = items.Min(it => (int)it.Item1.MsviFirstIndex),
                        IndexCount = items.Sum(it => (int)it.Item1.IndexCount),
                        FlipX = true
                    });
                }

                compIndex++;
            }
        }
    }

    private sealed class DSU
    {
        private readonly int[] parent;
        private readonly byte[] rank;
        public DSU(int n)
        {
            parent = new int[n];
            rank = new byte[n];
            for (int i = 0; i < n; i++) parent[i] = i;
        }
        public int Find(int x)
        {
            if (parent[x] != x) parent[x] = Find(parent[x]);
            return parent[x];
        }
        public void Union(int a, int b)
        {
            a = Find(a); b = Find(b);
            if (a == b) return;
            if (rank[a] < rank[b]) { parent[a] = b; }
            else if (rank[a] > rank[b]) { parent[b] = a; }
            else { parent[b] = a; rank[a]++; }
        }
    }

    private static void ExportGroup(
        Pm4Scene scene,
        List<(MsurChunk.Entry entry, int index)> items,
        string groupName,
        string objectsDir,
        Options opts,
        List<string> coverageCsv,
        List<ObjectIndexEntry> objectIndex)
    {
        var safe = SanitizeFileName(groupName);
        var objPath = Path.Combine(objectsDir, safe + ".obj");
        AssembleAndWrite(scene, items, objPath, opts with { SnapToPlane = false }, out var written, out var skipped);

        // Aggregate a simple coverage row
        int triCount = items.Sum(it => (int)it.entry.IndexCount);
        int firstMin = items.Min(it => (int)it.entry.MsviFirstIndex);
        int firstMax = items.Max(it => (int)it.entry.MsviFirstIndex);
        coverageCsv.Add(string.Join(',',
            $"group:{groupName}",
            items.First().entry.GroupKey,
            items.First().entry.CompositeKey,
            firstMin + "-" + firstMax,
            triCount,
            written,
            skipped,
            EscapeCsv(Path.GetRelativePath(Path.Combine(objectsDir, ".."), objPath))));

        // Add to object index
        objectIndex.Add(new ObjectIndexEntry
        {
            Id = $"group:{groupName}",
            Name = groupName,
            Group = "group",
            ObjPath = Path.GetRelativePath(Path.Combine(objectsDir, ".."), objPath).Replace("\\", "/"),
            GltfPath = opts.ExportGltf ? Path.GetRelativePath(Path.Combine(objectsDir, ".."), Path.ChangeExtension(objPath, ".gltf")).Replace("\\", "/") : null,
            GlbPath = opts.ExportGlb ? Path.GetRelativePath(Path.Combine(objectsDir, ".."), Path.ChangeExtension(objPath, ".glb")).Replace("\\", "/") : null,
            FacesWritten = written,
            FacesSkipped = skipped,
            SurfaceIndices = items.Select(t => t.index).ToList(),
            IndexFirst = firstMin,
            IndexCount = triCount,
            FlipX = true
        });
    }

    private static void AssembleAndWrite(
        Pm4Scene scene,
        IEnumerable<(MsurChunk.Entry entry, int index)> items,
        string objPath,
        Options opts,
        out int facesWritten,
        out int facesSkipped)
    {
        // Build local vertex/triangle lists from global scene for these surfaces
        var localVerts = new List<Vector3>(4096);
        var localTris = new List<(int A, int B, int C)>(4096);

        int skipped = 0, written = 0;

        if (!opts.SnapToPlane)
        {
            // Original behavior: single global-vertex dedup map
            var g2l = new Dictionary<int, int>(4096);
            foreach (var (entry, _) in items)
            {
                int first = unchecked((int)entry.MsviFirstIndex);
                int count = entry.IndexCount;
                if (first < 0 || count <= 0) continue;

                int end = Math.Min(scene.Indices.Count, first + count);
                for (int i = first; i + 2 < end; i += 3)
                {
                    int ga = scene.Indices[i];
                    int gb = scene.Indices[i + 1];
                    int gc = scene.Indices[i + 2];

                    if (!TryMap(scene, ga, g2l, localVerts, out var la) ||
                        !TryMap(scene, gb, g2l, localVerts, out var lb) ||
                        !TryMap(scene, gc, g2l, localVerts, out var lc))
                    {
                        skipped++;
                        continue;
                    }

                    // Skip degenerate faces (duplicate local vertex indices)
                    if (la == lb || lb == lc || lc == la)
                    {
                        skipped++;
                        continue;
                    }

                    localTris.Add((la, lb, lc));
                    written++;
                }
            }
        }
        else
        {
            // Plane snapping: per-surface+vertex key to avoid cross-surface dedup conflicts
            var g2l = new Dictionary<long, int>(4096);
            foreach (var (entry, idx) in items)
            {
                int first = unchecked((int)entry.MsviFirstIndex);
                int count = entry.IndexCount;
                if (first < 0 || count <= 0) continue;

                int end = Math.Min(scene.Indices.Count, first + count);
                for (int i = first; i + 2 < end; i += 3)
                {
                    int ga = scene.Indices[i];
                    int gb = scene.Indices[i + 1];
                    int gc = scene.Indices[i + 2];

                    long sKey = (long)(uint)idx; // stable per-surface key
                    if (!TryMapProjected(scene, entry, sKey, ga, opts.HeightScale, g2l, localVerts, out var la) ||
                        !TryMapProjected(scene, entry, sKey, gb, opts.HeightScale, g2l, localVerts, out var lb) ||
                        !TryMapProjected(scene, entry, sKey, gc, opts.HeightScale, g2l, localVerts, out var lc))
                    {
                        skipped++;
                        continue;
                    }

                    // Skip degenerate faces (duplicate local vertex indices)
                    if (la == lb || lb == lc || lc == la)
                    {
                        skipped++;
                        continue;
                    }

                    localTris.Add((la, lb, lc));
                    written++;
                }
            }
        }
        // Always flip X for objects to match tiles and legacy orientation
        ObjWriter.Write(objPath, localVerts, localTris, opts.LegacyParity, opts.ProjectLocal, true);
        if (opts.ExportGltf)
        {
            GltfWriter.WriteGltf(Path.ChangeExtension(objPath, ".gltf"), localVerts, localTris, opts.LegacyParity, opts.ProjectLocal, true);
        }
        if (opts.ExportGlb)
        {
            GltfWriter.WriteGlb(Path.ChangeExtension(objPath, ".glb"), localVerts, localTris, opts.LegacyParity, opts.ProjectLocal, true);
        }
        facesWritten = written;
        facesSkipped = skipped;
    }

    private static void AssembleAndWriteFromIndexRange(
        Pm4Scene scene,
        int startIndex,
        int indexCount,
        string objPath,
        Options opts,
        bool forceFlipX,
        out int facesWritten,
        out int facesSkipped)
    {
        var g2l = new Dictionary<int, int>(4096);
        var localVerts = new List<Vector3>(4096);
        var localTris = new List<(int A, int B, int C)>(Math.Max(0, indexCount / 3));

        int skipped = 0, written = 0;
        int start = Math.Max(0, startIndex);
        int end = Math.Min(scene.Indices.Count, start + Math.Max(0, indexCount));
        for (int i = start; i + 2 < end; i += 3)
        {
            int ga = scene.Indices[i];
            int gb = scene.Indices[i + 1];
            int gc = scene.Indices[i + 2];

            if (!TryMap(scene, ga, g2l, localVerts, out var la) ||
                !TryMap(scene, gb, g2l, localVerts, out var lb) ||
                !TryMap(scene, gc, g2l, localVerts, out var lc))
            {
                skipped++;
                continue;
            }

            // Skip degenerate faces (duplicate local vertex indices)
            if (la == lb || lb == lc || lc == la)
            {
                skipped++;
                continue;
            }

            localTris.Add((la, lb, lc));
            written++;
        }
        ObjWriter.Write(objPath, localVerts, localTris, opts.LegacyParity, opts.ProjectLocal, forceFlipX);
        if (opts.ExportGltf)
        {
            GltfWriter.WriteGltf(Path.ChangeExtension(objPath, ".gltf"), localVerts, localTris, opts.LegacyParity, opts.ProjectLocal, forceFlipX);
        }
        if (opts.ExportGlb)
        {
            GltfWriter.WriteGlb(Path.ChangeExtension(objPath, ".glb"), localVerts, localTris, opts.LegacyParity, opts.ProjectLocal, forceFlipX);
        }
        facesWritten = written;
        facesSkipped = skipped;
    }

    private static bool TryMap(
        Pm4Scene scene,
        int g,
        Dictionary<int, int> g2l,
        List<Vector3> localVerts,
        out int local)
    {
        local = -1;
        if (g < 0 || g >= scene.Vertices.Count) return false;
        if (!g2l.TryGetValue(g, out local))
        {
            local = localVerts.Count;
            g2l[g] = local;
            localVerts.Add(scene.Vertices[g]);
        }
        return true;
    }

    private static bool TryMapProjected(
        Pm4Scene scene,
        MsurChunk.Entry entry,
        long surfaceKey,
        int g,
        float heightScale,
        Dictionary<long, int> g2l,
        List<Vector3> localVerts,
        out int local)
    {
        local = -1;
        if (g < 0 || g >= scene.Vertices.Count) return false;
        long key = (surfaceKey << 32) | (uint)g;
        if (!g2l.TryGetValue(key, out local))
        {
            Vector3 v = scene.Vertices[g];
            Vector3 n = new Vector3(entry.Nx, entry.Ny, entry.Nz);
            float n2 = n.LengthSquared();
            if (n2 > 1e-12f)
            {
                // Correct projection using unnormalized plane: dot(N, v) = H'
                float H = entry.Height * (heightScale == 0f ? 1f : heightScale);
                float d = (Vector3.Dot(n, v) - H) / n2;
                v -= n * d;
            }
            local = localVerts.Count;
            g2l[key] = local;
            localVerts.Add(v);
        }
        return true;
    }

    // Evaluate candidate scales for MSUR.Height by measuring plane residuals across all surfaces
    private static void EmitHeightFitReport(Pm4Scene scene, string outDir)
    {
        try
        {
            // Candidate scales to test
            var candidates = new float[] { 1.0f, 0.02777778f, 0.0625f, 36.0f, -1.0f, -0.02777778f, -36.0f };
            const float epsilon = 0.01f; // world units

            // Pre-collect per-surface unique vertex indices to speed residual computation
            int surfCount = scene.Surfaces.Count;
            var surfVerts = new List<int>[surfCount];
            for (int si = 0; si < surfCount; si++)
            {
                var s = scene.Surfaces[si];
                int first = unchecked((int)s.MsviFirstIndex);
                int count = s.IndexCount;
                if (first < 0 || count <= 0 || first + count > scene.Indices.Count)
                {
                    surfVerts[si] = new List<int>();
                    continue;
                }
                var set = new HashSet<int>();
                for (int k = 0; k < count; k++)
                {
                    int gi = scene.Indices[first + k];
                    if (gi >= 0 && gi < scene.Vertices.Count) set.Add(gi);
                }
                surfVerts[si] = set.Count == 0 ? new List<int>() : set.ToList();
            }

            var lines = new List<string>();
            lines.Add("scale,total_surfaces,with_data,mean_abs_residual,median_abs_residual,p95_abs_residual,max_abs_residual,ok_ratio(<0.01)");

            foreach (var scale in candidates)
            {
                var perSurfaceAbsMean = new List<float>(surfCount);
                int withData = 0;
                int okCount = 0;
                float maxAbs = 0f;
                double sumAbs = 0.0;

                for (int si = 0; si < surfCount; si++)
                {
                    var entry = scene.Surfaces[si];
                    var verts = surfVerts[si];
                    if (verts == null || verts.Count == 0) continue;

                    Vector3 n = new(entry.Nx, entry.Ny, entry.Nz);
                    float n2 = n.LengthSquared();
                    if (n2 <= 1e-12f) continue;

                    float H = entry.Height * (scale == 0f ? 1f : scale);
                    double acc = 0.0;
                    foreach (var gi in verts)
                    {
                        var v = scene.Vertices[gi];
                        float r = Vector3.Dot(n, v) - H; // residual
                        float ar = MathF.Abs(r);
                        if (ar > maxAbs) maxAbs = ar;
                        acc += ar;
                    }
                    float meanAbs = (float)(acc / verts.Count);
                    perSurfaceAbsMean.Add(meanAbs);
                    withData++;
                    if (meanAbs < epsilon) okCount++;
                    sumAbs += meanAbs;
                }

                float median = 0f, p95 = 0f;
                if (perSurfaceAbsMean.Count > 0)
                {
                    perSurfaceAbsMean.Sort();
                    int mIdx = perSurfaceAbsMean.Count / 2;
                    median = perSurfaceAbsMean[perSurfaceAbsMean.Count % 2 == 1
                        ? mIdx
                        : (int)Math.Clamp(mIdx - 1, 0, perSurfaceAbsMean.Count - 1)];
                    int p95Idx = (int)Math.Clamp(Math.Ceiling(0.95 * perSurfaceAbsMean.Count) - 1, 0, perSurfaceAbsMean.Count - 1);
                    p95 = perSurfaceAbsMean[p95Idx];
                }

                float mean = withData > 0 ? (float)(sumAbs / withData) : 0f;
                float okRatio = withData > 0 ? (float)okCount / withData : 0f;

                lines.Add(string.Join(',',
                    scale.ToString("G9", CultureInfo.InvariantCulture),
                    surfCount.ToString(CultureInfo.InvariantCulture),
                    withData.ToString(CultureInfo.InvariantCulture),
                    mean.ToString("G9", CultureInfo.InvariantCulture),
                    median.ToString("G9", CultureInfo.InvariantCulture),
                    p95.ToString("G9", CultureInfo.InvariantCulture),
                    maxAbs.ToString("G9", CultureInfo.InvariantCulture),
                    okRatio.ToString("G9", CultureInfo.InvariantCulture)));
            }

            File.WriteAllLines(Path.Combine(outDir, "msur_plane_fit.csv"), lines);
            Console.WriteLine("[pm4-faces] Wrote msur_plane_fit.csv (height-fit report)");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[pm4-faces] Failed to emit height fit report: {ex.Message}");
        }
    }

    private static void ExportTiles(
        Pm4Scene scene,
        string tilesDir,
        Options opts,
        string outDir,
        List<string> tileCsv,
        List<TileIndexEntry> tileIndex,
        string tilePrefix)
    {
        if (scene.TileIndexOffsetByTileId.Count == 0)
        {
            Console.WriteLine("[pm4-faces] No tile index metadata present; skipping tile export.");
            return;
        }

        foreach (var kv in scene.TileIndexOffsetByTileId.OrderBy(k => k.Key))
        {
            int tileId = kv.Key;
            int start = kv.Value;
            if (!scene.TileIndexCountByTileId.TryGetValue(tileId, out int count))
            {
                // If count missing, try to infer a safe bound (skip if not available)
                Console.WriteLine($"[pm4-faces] Missing index count for tile {tileId}; skipping.");
                continue;
            }

            // Derive name from tileId -> (x,y) using original prefix
            int x = tileId % 64;
            int y = tileId / 64;
            string name = $"{tilePrefix}_{x:D2}_{y:D2}";
            string objPath = Path.Combine(tilesDir, SanitizeFileName(name) + ".obj");

            // Force X-axis flip for tile exports and log mapping
            Console.WriteLine($"[pm4-faces][tile] id={tileId} name={name} start={start} count={count}");
            AssembleAndWriteFromIndexRange(scene, start, count, objPath, opts, true, out var written, out var skipped);

            tileCsv.Add(string.Join(',',
                tileId,
                start,
                count,
                written,
                skipped,
                EscapeCsv(Path.GetRelativePath(outDir, objPath))));

            // Add to tile index
            tileIndex.Add(new TileIndexEntry
            {
                TileId = tileId,
                Name = name,
                ObjPath = Path.GetRelativePath(outDir, objPath).Replace("\\", "/"),
                GltfPath = opts.ExportGltf ? Path.GetRelativePath(outDir, Path.ChangeExtension(objPath, ".gltf")).Replace("\\", "/") : null,
                GlbPath = opts.ExportGlb ? Path.GetRelativePath(outDir, Path.ChangeExtension(objPath, ".glb")).Replace("\\", "/") : null,
                StartIndex = start,
                IndexCount = count,
                FacesWritten = written,
                FacesSkipped = skipped,
                FlipX = true
            });
        }
    }

    private static Options? ParseArgs(string[] args)
    {
        string input = string.Empty;
        string outDir = Path.Combine(Environment.CurrentDirectory, "project_output", "pm4faces_" + DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture));
        bool legacy = false;
        bool projectLocal = false;
        bool snapToPlane = false;
        float heightScale = 1.0f;
        bool heightFitReport = false;
        string groupBy = "composite-instance"; // composite-instance | surface | groupkey | composite
        bool batch = false;
        bool ckUseMslk = false;
        double ckAllowUnlinkedRatio = 0.5;
        int ckMinTris = 0;
        bool ckMergeComponents = false;
        bool ckMonolithic = false;
        bool exportGltf = false;
        bool exportGlb = false;
        bool noMscnRemap = false;

        for (int i = 0; i < args.Length; i++)
        {
            var a = args[i];
            switch (a)
            {
                case "--input":
                case "-i":
                    input = i + 1 < args.Length ? args[++i] : input;
                    break;
                case "--out":
                case "-o":
                    outDir = i + 1 < args.Length ? args[++i] : outDir;
                    break;
                case "--legacy-parity":
                    legacy = true;
                    break;
                case "--project-local":
                    projectLocal = true;
                    break;
                case "--snap-to-plane":
                    snapToPlane = true;
                    break;
                case "--height-scale":
                    if (i + 1 < args.Length && float.TryParse(args[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out var hs))
                    {
                        heightScale = hs;
                        i++;
                    }
                    break;
                case "--height-fit-report":
                    heightFitReport = true;
                    break;
                case "--group-by":
                    groupBy = i + 1 < args.Length ? args[++i] : groupBy;
                    break;
                case "--batch":
                    batch = true;
                    break;
                case "--ck-use-mslk":
                    ckUseMslk = true;
                    break;
                case "--ck-allow-unlinked-ratio":
                    if (i + 1 < args.Length && double.TryParse(args[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out var ratio))
                    {
                        ckAllowUnlinkedRatio = Math.Clamp(ratio, 0.0, 1.0);
                        i++;
                    }
                    break;
                case "--ck-min-tris":
                    if (i + 1 < args.Length && int.TryParse(args[i + 1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var minT))
                    {
                        ckMinTris = Math.Max(0, minT);
                        i++;
                    }
                    break;
                case "--ck-merge-components":
                    ckMergeComponents = true;
                    break;
                case "--ck-monolithic":
                    ckMonolithic = true;
                    break;
                case "--gltf":
                    exportGltf = true;
                    break;
                case "--glb":
                    exportGlb = true;
                    break;
                case "--no-mscn-remap":
                    noMscnRemap = true;
                    break;
                case "--help":
                case "-h":
                    return null;
            }
        }

        if (string.IsNullOrWhiteSpace(input)) return null;
        return new Options(input, outDir, legacy, projectLocal, snapToPlane, heightScale, heightFitReport, groupBy, batch, ckUseMslk, ckAllowUnlinkedRatio, ckMinTris, ckMergeComponents, ckMonolithic, exportGltf, exportGlb, noMscnRemap);
    }

    private static void PrintHelp()
    {
        Console.WriteLine("pm4-faces export --input <tile.pm4|dir> [--out <dir>] [--batch] [--group-by composite-instance|type-instance|type-attr-instance|surface|groupkey|composite|render-mesh] [--legacy-parity] [--project-local] [--snap-to-plane] [--height-scale <float>] [--height-fit-report] [--ck-use-mslk] [--ck-allow-unlinked-ratio <0..1>] [--ck-min-tris <int>] [--ck-merge-components] [--ck-monolithic] [--gltf] [--glb] [--no-mscn-remap]");
        Console.WriteLine("  Single-file input: loads ONLY that tile. Use --batch to process all tiles with the same prefix.");
        Console.WriteLine("  --snap-to-plane: project vertices to each surface's MSUR plane (experimental; off by default).");
        Console.WriteLine("  --height-scale: multiply MSUR Height by this factor during snapping (e.g., 0.02777778 for 1/36, 0.0625 for 1/16).");
        Console.WriteLine("  --height-fit-report: emit msur_plane_fit.csv with residuals for candidate height scales (no behavior change).");
        Console.WriteLine("  --ck-merge-components: retain per-object DSU components under each CK24 (no monolithic merged OBJ).");
        Console.WriteLine("  --ck-monolithic: export a single merged OBJ per CK24 (skip per-component DSU objects).");
        Console.WriteLine("  --gltf / --glb: also export glTF 2.0 (.gltf+.bin) and/or GLB alongside OBJ outputs.");
        Console.WriteLine("  --no-mscn-remap: disable global MSCN vertex remapping during region load (advanced; default is enabled).");
        Console.WriteLine("  Group-by 'render-mesh' (alias 'surfaces-all'): skip object exports; rely solely on tile exports.");
        Console.WriteLine("  Note: X-axis flipping is always applied by default across tiles and objects; --legacy-parity is not required for flipping.");
    }

    private static string SanitizeFileName(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var s = string.Join("_", name.Split(invalid, StringSplitOptions.RemoveEmptyEntries)).TrimEnd('.');
        return string.IsNullOrWhiteSpace(s) ? "object" : s;
    }

    private static string EscapeCsv(string s)
    {
        if (s.Contains(',') || s.Contains('"') || s.Contains('\n'))
        {
            return '"' + s.Replace("\"", "\"\"") + '"';
        }
        return s;
    }

    private static void EmitMsurHeightDiagnostics(Pm4Scene scene, string outDir)
    {
        try
        {
            var heights = scene.Surfaces.Select(s => s.Height).ToList();
            int total = heights.Count;
            int zeros = heights.Count(h => h == 0f);
            int nonzeros = total - zeros;
            float min = total > 0 ? heights.Min() : 0f;
            float max = total > 0 ? heights.Max() : 0f;

            // Unique value counts
            var counts = new Dictionary<float, int>();
            foreach (var h in heights)
            {
                if (!counts.TryGetValue(h, out var c)) c = 0;
                counts[h] = c + 1;
            }

            // Overview CSV
            var overview = new List<string>
            {
                "total,unique,zeros,nonzeros,min,max",
                string.Join(',',
                    total.ToString(CultureInfo.InvariantCulture),
                    counts.Count.ToString(CultureInfo.InvariantCulture),
                    zeros.ToString(CultureInfo.InvariantCulture),
                    nonzeros.ToString(CultureInfo.InvariantCulture),
                    min.ToString("G9", CultureInfo.InvariantCulture),
                    max.ToString("G9", CultureInfo.InvariantCulture))
            };
            File.WriteAllLines(Path.Combine(outDir, "msur_height_overview.csv"), overview);

            // Values CSV (sorted by height)
            var lines = new List<string> { "height,count" };
            foreach (var kv in counts.OrderBy(k => k.Key))
            {
                lines.Add(string.Join(',',
                    kv.Key.ToString("G9", CultureInfo.InvariantCulture),
                    kv.Value.ToString(CultureInfo.InvariantCulture)));
            }
            File.WriteAllLines(Path.Combine(outDir, "msur_height_values.csv"), lines);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[pm4-faces] Failed to emit MSUR height diagnostics: {ex.Message}");
        }
    }

    private static uint CK24(uint key)
    {
        return (key & 0xFFFFFF00u) >> 8;
    }

    private static int? DominantTileIdFor(Pm4Scene scene, IEnumerable<MsurChunk.Entry> entries)
    {
        if (scene.TileIndexOffsetByTileId.Count == 0 || scene.TileIndexCountByTileId.Count == 0)
            return null;

        // Build quick lookup for tile index ranges
        var ranges = new List<(int TileId, int Start, int End)>();
        foreach (var kv in scene.TileIndexOffsetByTileId)
        {
            int tileId = kv.Key;
            int start = kv.Value;
            if (!scene.TileIndexCountByTileId.TryGetValue(tileId, out int count)) continue;
            ranges.Add((tileId, start, start + count));
        }
        if (ranges.Count == 0) return null;

        var weights = new Dictionary<int, int>();
        foreach (var e in entries)
        {
            int first = unchecked((int)e.MsviFirstIndex);
            if (first < 0) continue;
            foreach (var r in ranges)
            {
                if (first >= r.Start && first < r.End)
                {
                    if (!weights.TryGetValue(r.TileId, out var w)) w = 0;
                    weights[r.TileId] = w + Math.Max(1, (int)e.IndexCount);
                    break;
                }
            }
        }
        if (weights.Count == 0) return null;

        int bestTile = -1;
        int bestW = -1;
        foreach (var kv in weights)
        {
            if (kv.Value > bestW) { bestW = kv.Value; bestTile = kv.Key; }
        }
        return bestTile >= 0 ? bestTile : null;
    }

    private static int DominantObjectIdFor(IEnumerable<MsurChunk.Entry> entries)
    {
        if (entries == null) return -1;

        var counts = new Dictionary<int, int>();
        foreach (var e in entries)
        {
            int id = e.IndexCount; // MSUR IndexCount acts as object identifier
            if (id <= 0) continue; // ignore zero to avoid obj_000
            if (!counts.TryGetValue(id, out var c)) c = 0;
            counts[id] = c + 1; // frequency-based dominance
        }

        if (counts.Count == 0) return -1;

        int bestId = -1;
        int bestCount = -1;
        foreach (var kv in counts)
        {
            // prefer higher frequency; tie-breaker: larger id for determinism
            if (kv.Value > bestCount || (kv.Value == bestCount && kv.Key > bestId))
            {
                bestId = kv.Key;
                bestCount = kv.Value;
            }
        }

        return bestId;
    }

    private static string SessionNameFrom(string firstTilePath)
    {
        var parent = Path.GetFileName(Path.GetDirectoryName(firstTilePath) ?? "");
        var tile = Path.GetFileNameWithoutExtension(firstTilePath);
        return $"{tile}";
    }

    // Index DTOs
    private sealed class ObjectIndexEntry
    {
        public string Id { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string Group { get; set; } = string.Empty;
        public string ObjPath { get; set; } = string.Empty;
        public string? GltfPath { get; set; }
        public string? GlbPath { get; set; }
        public int FacesWritten { get; set; }
        public int FacesSkipped { get; set; }
        public List<int> SurfaceIndices { get; set; } = new();
        public int IndexFirst { get; set; }
        public int IndexCount { get; set; }
        public bool FlipX { get; set; }
        public bool IsWalkable { get; set; }
        public bool IsM2 { get; set; }
        public uint Ck24 { get; set; }
    }

    private sealed class TileIndexEntry
    {
        public int TileId { get; set; }
        public string Name { get; set; } = string.Empty;
        public string ObjPath { get; set; } = string.Empty;
        public string? GltfPath { get; set; }
        public string? GlbPath { get; set; }
        public int StartIndex { get; set; }
        public int IndexCount { get; set; }
        public int FacesWritten { get; set; }
        public int FacesSkipped { get; set; }
        public bool FlipX { get; set; }
    }
}
