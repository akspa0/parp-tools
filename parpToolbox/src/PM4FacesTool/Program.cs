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
using ParpToolbox.Services.Geometry;

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
        bool RenderMeshMerged,
        bool TilesApplyTransforms,
        bool MscnFollowTiles,
        bool NoMscnRemap,
        bool FlipXEnabled,
        bool FlipYEnabled,
        float RotXDeg,
        float RotYDeg,
        float RotZDeg,
        float TranslateX,
        float TranslateY,
        float TranslateZ,
        bool MscnSidecar,
        int MscnPreRotZ,
        string MscnPreFlip,
        string MscnBasis
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

    // Simple CLI parser to keep the tool functional
    private static Options? ParseArgs(string[] args)
    {
        string input = string.Empty;
        string outDir = Path.Combine(Directory.GetCurrentDirectory(), "pm4faces_out");
        bool legacyParity = false;
        bool projectLocal = false;
        bool snapToPlane = false;
        float heightScale = 1.0f;
        bool heightFitReport = false;
        string groupBy = "composite-instance";
        bool batch = false;
        bool ckUseMslk = true;
        double ckAllowUnlinkedRatio = 0.05;
        int ckMinTris = 1;
        bool ckMergeComponents = true;
        bool ckMonolithic = false;
        bool exportGltf = false;
        bool exportGlb = false;
        bool renderMeshMerged = false;
        bool noMscnRemap = false;
        bool flipXEnabled = false;
        bool flipYEnabled = false;
        bool tilesApplyTransforms = false;
        bool mscnFollowTiles = true;
        float rotX = 0, rotY = 0, rotZ = 0;
        float tx = 0, ty = 0, tz = 0;
        bool mscnSidecar = false;
        int mscnPreRotZ = 0;
        string mscnPreFlip = "none";
        string mscnBasis = "legacy";

        // Positional path support (first non-flag)
        int i = 0;
        while (i < args.Length)
        {
            var a = args[i];
            if (!a.StartsWith("--"))
            {
                if (string.IsNullOrEmpty(input)) input = a;
                else outDir = a;
                i++; continue;
            }

            string next(int ofs = 1) => (i + ofs < args.Length) ? args[i + ofs] : string.Empty;

            switch (a.ToLowerInvariant())
            {
                case "--input": input = next(); i += 2; break;
                case "--out": case "--out-dir": outDir = next(); i += 2; break;
                case "--legacy-parity": legacyParity = true; i++; break;
                case "--project-local": projectLocal = true; i++; break;
                case "--snap-to-plane": snapToPlane = true; i++; break;
                case "--height-scale": float.TryParse(next(), NumberStyles.Float, CultureInfo.InvariantCulture, out heightScale); i += 2; break;
                case "--height-fit-report": heightFitReport = true; i++; break;
                case "--group-by": groupBy = next(); i += 2; break;
                case "--batch": batch = true; i++; break;
                case "--ck-use-mslk": ckUseMslk = true; i++; break;
                case "--ck-allow-unlinked-ratio": double.TryParse(next(), NumberStyles.Float, CultureInfo.InvariantCulture, out ckAllowUnlinkedRatio); i += 2; break;
                case "--ck-min-tris": int.TryParse(next(), NumberStyles.Integer, CultureInfo.InvariantCulture, out ckMinTris); i += 2; break;
                case "--ck-merge-components": ckMergeComponents = true; i++; break;
                case "--ck-monolithic": ckMonolithic = true; i++; break;
                case "--export-gltf": exportGltf = true; i++; break;
                case "--export-glb": exportGlb = true; i++; break;
                case "--render-mesh-merged": renderMeshMerged = true; i++; break;
                case "--no-mscn-remap": noMscnRemap = true; i++; break;
                case "--flip-x": flipXEnabled = true; i++; break;
                case "--flip-y": flipYEnabled = true; i++; break;
                case "--tiles-apply-transforms": tilesApplyTransforms = true; i++; break;
                case "--mscn-follow-tiles": mscnFollowTiles = true; i++; break;
                case "--no-mscn-follow-tiles": mscnFollowTiles = false; i++; break;
                case "--rot-x": float.TryParse(next(), NumberStyles.Float, CultureInfo.InvariantCulture, out rotX); i += 2; break;
                case "--rot-y": float.TryParse(next(), NumberStyles.Float, CultureInfo.InvariantCulture, out rotY); i += 2; break;
                case "--rot-z": float.TryParse(next(), NumberStyles.Float, CultureInfo.InvariantCulture, out rotZ); i += 2; break;
                case "--translate-x": float.TryParse(next(), NumberStyles.Float, CultureInfo.InvariantCulture, out tx); i += 2; break;
                case "--translate-y": float.TryParse(next(), NumberStyles.Float, CultureInfo.InvariantCulture, out ty); i += 2; break;
                case "--translate-z": float.TryParse(next(), NumberStyles.Float, CultureInfo.InvariantCulture, out tz); i += 2; break;
                case "--mscn-sidecar": mscnSidecar = true; i++; break;
                case "--mscn-pre-rotz": int.TryParse(next(), NumberStyles.Integer, CultureInfo.InvariantCulture, out mscnPreRotZ); i += 2; break;
                case "--mscn-pre-flip": mscnPreFlip = next(); i += 2; break;
                case "--mscn-basis": mscnBasis = next(); i += 2; break;
                case "--help": PrintHelp(); return null;
                default:
                    // Ignore unknown flags for forward-compat
                    i++; break;
            }
        }

        if (string.IsNullOrWhiteSpace(input))
            return null;

        // Derive default outDir next to input if not explicitly provided
        if (!Path.IsPathRooted(outDir) && !outDir.Contains(Path.DirectorySeparatorChar) && !outDir.Contains(Path.AltDirectorySeparatorChar))
        {
            var inpDir = Directory.Exists(input) ? input : Path.GetDirectoryName(input) ?? Directory.GetCurrentDirectory();
            outDir = Path.Combine(inpDir, outDir);
        }

        return new Options(
            input,
            outDir,
            legacyParity,
            projectLocal,
            snapToPlane,
            heightScale,
            heightFitReport,
            groupBy,
            batch,
            ckUseMslk,
            ckAllowUnlinkedRatio,
            ckMinTris,
            ckMergeComponents,
            ckMonolithic,
            exportGltf,
            exportGlb,
            renderMeshMerged,
            tilesApplyTransforms,
            mscnFollowTiles,
            noMscnRemap,
            flipXEnabled,
            flipYEnabled,
            rotX,
            rotY,
            rotZ,
            tx,
            ty,
            tz,
            mscnSidecar,
            mscnPreRotZ,
            mscnPreFlip,
            mscnBasis
        );
    }

    private static void PrintHelp()
    {
        Console.WriteLine("pm4-faces usage:\n  pm4-faces <input.pm4|dir> [outDir] [--group-by <mode>] [--export-gltf] [--export-glb] [--render-mesh-merged]\n  Transforms: --project-local --flip-x --flip-y --rot-x <deg> --rot-y <deg> --rot-z <deg> --translate-x <m> --translate-y <m> --translate-z <m>\n  Tiles: --tiles-apply-transforms (apply global transforms to tile OBJs; default is raw, legacy-parity flipped)\n  MSCN: --mscn-sidecar --mscn-follow-tiles|--no-mscn-follow-tiles (default: follow) --mscn-pre-rotz <deg> --mscn-pre-flip <axes> --mscn-basis <legacy|remap>\n  Misc: --snap-to-plane --height-scale <f> --height-fit-report --batch");
    }

    // Transform helpers bridging CLI options to core services
    private static GeometryTransformService.TransformOptions ToTransformOptions(Options opts)
    {
        return new GeometryTransformService.TransformOptions
        {
            ProjectLocal = opts.ProjectLocal,
            FlipXEnabled = opts.FlipXEnabled,
            FlipYEnabled = opts.FlipYEnabled,
            RotXDeg = opts.RotXDeg,
            RotYDeg = opts.RotYDeg,
            RotZDeg = opts.RotZDeg,
            TranslateX = opts.TranslateX,
            TranslateY = opts.TranslateY,
            TranslateZ = opts.TranslateZ
        };
    }

    private static MscnTransformService.MscnBasis ParseMscnBasis(string basis)
    {
        var b = (basis ?? "legacy").Trim().ToLowerInvariant();
        return b == "remap" ? MscnTransformService.MscnBasis.Remap : MscnTransformService.MscnBasis.Legacy;
    }

    private static MscnTransformService.FlipAxes ParseFlipAxes(string flip)
    {
        var f = (flip ?? "none").Trim().ToLowerInvariant();
        return f switch
        {
            "x" => MscnTransformService.FlipAxes.X,
            "y" => MscnTransformService.FlipAxes.Y,
            "z" => MscnTransformService.FlipAxes.Z,
            "xy" => MscnTransformService.FlipAxes.XY,
            "xz" => MscnTransformService.FlipAxes.XZ,
            "yz" => MscnTransformService.FlipAxes.YZ,
            _ => MscnTransformService.FlipAxes.None
        };
    }

    private static string SanitizeFileName(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var clean = new char[name.Length];
        int j = 0;
        foreach (var ch in name)
        {
            clean[j++] = invalid.Contains(ch) ? '_' : ch;
        }
        return new string(clean, 0, j);
    }

    private static string EscapeCsv(string s)
    {
        if (string.IsNullOrEmpty(s)) return s;
        bool needs = s.Contains(',') || s.Contains('"') || s.Contains('\n');
        if (!needs) return s;
        return '"' + s.Replace("\"", "\"\"") + '"';
    }

    private static void EmitMsurHeightDiagnostics(Pm4Scene scene, string outDir)
    {
        try
        {
            // Lightweight placeholder to keep compatibility; detailed report is in EmitHeightFitReport
            var path = Path.Combine(outDir, "msur_height_diag.csv");
            var lines = new List<string> { "surfaces,total_indices" };
            lines.Add(string.Join(',', scene.Surfaces.Count.ToString(CultureInfo.InvariantCulture), scene.Indices.Count.ToString(CultureInfo.InvariantCulture)));
            File.WriteAllLines(path, lines);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[pm4-faces] Failed to emit msur_height_diag.csv: {ex.Message}");
        }
    }

    private static void ExportTiles(
        Pm4Scene scene,
        string tilesDir,
        Options opts,
        string outDir,
        List<string> tileCsv,
        List<TileIndexEntry> tileIndex,
        string prefix)
    {
        // Generate one OBJ per tile using object-first assembly of surfaces whose first index falls within the tile range
        Directory.CreateDirectory(tilesDir);

        // Build quick list of surfaces with their scene index for selection per tile
        var surfaces = scene.Surfaces.Select((e, i) => (entry: e, idx: i)).ToList();

        foreach (var kv in scene.TileIndexOffsetByTileId)
        {
            int tileId = kv.Key;
            int start = kv.Value;
            if (!scene.TileIndexCountByTileId.TryGetValue(tileId, out int count)) count = 0;
            int end = start + Math.Max(0, count);

            int tileX = tileId % 64;
            int tileY = tileId / 64;

            // Select surfaces whose first index falls into this tile's index range
            var items = new List<(MsurChunk.Entry entry, int idx)>();
            foreach (var s in surfaces)
            {
                int first = unchecked((int)s.entry.MsviFirstIndex);
                if (first >= start && first < end)
                {
                    items.Add((s.entry, s.idx));
                }
            }

            string nameCore = $"{prefix}_t{tileX:D2}_{tileY:D2}";
            string safe = SanitizeFileName(nameCore);
            string objPath = Path.Combine(tilesDir, safe + ".obj");

            int writtenFaces = 0;
            int skippedFaces = 0;
            bool wroteFile = false;

            if (items.Count > 0)
            {
                if (opts.TilesApplyTransforms)
                {
                    // Apply the same global transforms as objects; no legacy writer parity
                    AssembleAndWrite(scene, items, objPath, opts with { SnapToPlane = false }, writerLegacyParity: false, out writtenFaces, out skippedFaces);
                    wroteFile = (writtenFaces > 0);
                }
                else
                {
                    // Default: raw scene space, disable global transforms, fix legacy parity at writer level
                    var tileOpts = opts with
                    {
                        FlipXEnabled = false,
                        FlipYEnabled = false,
                        RotXDeg = 0,
                        RotYDeg = 0,
                        RotZDeg = 0,
                        TranslateX = 0,
                        TranslateY = 0,
                        TranslateZ = 0,
                        SnapToPlane = false,
                        ProjectLocal = false
                    };
                    AssembleAndWrite(scene, items, objPath, tileOpts, writerLegacyParity: true, out writtenFaces, out skippedFaces);
                    wroteFile = (writtenFaces > 0);
                }
            }

            string relObj = wroteFile ? Path.GetRelativePath(outDir, objPath).Replace("\\", "/") : string.Empty;

            // CSV row
            tileCsv.Add(string.Join(',',
                tileId.ToString(CultureInfo.InvariantCulture),
                start.ToString(CultureInfo.InvariantCulture),
                Math.Max(0, count).ToString(CultureInfo.InvariantCulture),
                writtenFaces.ToString(CultureInfo.InvariantCulture),
                skippedFaces.ToString(CultureInfo.InvariantCulture),
                relObj));

            // JSON index entry
            tileIndex.Add(new TileIndexEntry
            {
                TileId = tileId,
                Name = $"t{tileX:D2}_{tileY:D2}",
                ObjPath = relObj,
                GltfPath = (wroteFile && opts.ExportGltf) ? Path.ChangeExtension(relObj, ".gltf") : null,
                GlbPath = (wroteFile && opts.ExportGlb) ? Path.ChangeExtension(relObj, ".glb") : null,
                StartIndex = start,
                IndexCount = Math.Max(0, count),
                FacesWritten = writtenFaces,
                FacesSkipped = skippedFaces,
                FlipX = opts.TilesApplyTransforms ? opts.FlipXEnabled : true
            });
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
                    AssembleAndWrite(scene, items, objPath, opts with { SnapToPlane = false }, writerLegacyParity: false, out writtenFaces, out skippedFaces);
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
                    (skipWrite ? string.Empty : EscapeCsv(Path.GetRelativePath(outDir, objPath)))));

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
                        FlipX = opts.FlipXEnabled,
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

        // Log resolved output paths for traceability
        Console.WriteLine($"[pm4-faces] Output paths:\n  outDir={outDir}\n  objectsDir={objectsDir}\n  tilesDir={tilesDir}");

        // Emit lightweight diagnostics for MSUR.Height variability
        EmitMsurHeightDiagnostics(scene, outDir);
        if (opts.HeightFitReport)
        {
            EmitHeightFitReport(scene, outDir);
        }

        Console.WriteLine($"[pm4-faces] Export strategy: objects={opts.GroupBy}, tiles=on");
        Console.WriteLine($"[pm4-faces] Scene: Vertices={scene.Vertices.Count}, Indices={scene.Indices.Count}, Surfaces={scene.Surfaces.Count}, Tiles={scene.TileIndexOffsetByTileId.Count}");
        // Log tiles pipeline decision for traceability
        if (opts.TilesApplyTransforms)
        {
            Console.WriteLine("[pm4-faces] Tiles: applying global transforms (flip/rot/translate); legacy parity OFF");
        }
        else
        {
            Console.WriteLine("[pm4-faces] Tiles: raw geometry; global transforms IGNORED for tiles; legacy X parity at writer ON");
        }

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
                AssembleAndWrite(scene, new[] {(entry, idx)}, objPath, opts, writerLegacyParity: false, out var written, out var skipped);
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
                    FlipX = opts.FlipXEnabled
                });
            }
        }
        else if (opts.GroupBy.Equals("render-mesh", StringComparison.OrdinalIgnoreCase) ||
                 opts.GroupBy.Equals("surfaces-all", StringComparison.OrdinalIgnoreCase))
        {
            // Skip object exports; also emit a single merged render mesh and rely on tile exports below.
            try
            {
                ExportRenderMeshMerged(scene, tilesDir, opts);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[pm4-faces] Failed to emit merged render mesh: {ex.Message}");
            }
        }
        else
        {
            // Default to composite-instance if an unknown group was provided
            Console.WriteLine($"[pm4-faces] Unknown --group-by '{opts.GroupBy}', defaulting to composite-instance");
            ExportCompositeInstances(scene, objectsDir, opts, outDir, coverageCsv, objectIndex);
        }

        File.WriteAllLines(Path.Combine(outDir, "surface_coverage.csv"), coverageCsv);

        // If requested, emit a merged render mesh alongside any group-by mode
        if (opts.RenderMeshMerged &&
            !opts.GroupBy.Equals("render-mesh", StringComparison.OrdinalIgnoreCase) &&
            !opts.GroupBy.Equals("surfaces-all", StringComparison.OrdinalIgnoreCase))
        {
            try
            {
                ExportRenderMeshMerged(scene, tilesDir, opts);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[pm4-faces] Failed to emit merged render mesh (flag): {ex.Message}");
            }
        }

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

        // Optional: MSCN sidecar export
        if (opts.MscnSidecar)
        {
            try
            {
                ExportMscnSidecar(scene, outDir, opts);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[pm4-faces] MSCN sidecar failed: {ex.Message}");
            }
        }
    }

    private static void ExportMscnSidecar(Pm4Scene scene, string outDir, Options opts)
    {
        var mdir = Path.Combine(outDir, "mscn");
        Directory.CreateDirectory(mdir);

        var header = "tile_id,tile_x,tile_y,count,obj_path";
        var rows = new List<string> { header };

        var verts = scene.MscnVertices ?? new List<Vector3>();
        var tileIds = scene.MscnVertexTileIds ?? new List<int>();
        if (verts.Count == 0 || tileIds.Count != verts.Count)
        {
            // Emit empty counts file to signal no MSCN data
            File.WriteAllLines(Path.Combine(mdir, "mscn_counts.csv"), rows);
            Console.WriteLine("[pm4-faces] No MSCN vertices available for sidecar export.");
            return;
        }

        // Log MSCN pre-transform selection (MSCN-only step before mesh pipeline decisions)
        Console.WriteLine($"[pm4-faces] MSCN pre-transform: basis='{opts.MscnBasis}', rotZ={opts.MscnPreRotZ} deg, flip='{opts.MscnPreFlip}'");

        // Group points by tileId
        var groups = new Dictionary<int, List<Vector3>>();
        for (int i = 0; i < verts.Count; i++)
        {
            int tid = tileIds[i];
            if (!groups.TryGetValue(tid, out var list)) { list = new List<Vector3>(); groups[tid] = list; }
            list.Add(verts[i]);
        }

        foreach (var kv in groups)
        {
            int tileId = kv.Key;
            int tileX = tileId % 64;
            int tileY = tileId / 64;
            var local = new List<Vector3>(kv.Value);

            // Optional MSCN-only pre-transform (basis canonicalization, then rotate around Z and/or mirror)
            MscnTransformService.PreTransform(local, opts.MscnPreRotZ, ParseFlipAxes(opts.MscnPreFlip), ParseMscnBasis(opts.MscnBasis));

            // Decide MSCN effective pipeline based on tiles export mode and follow-tiles flag
            bool followTilesRaw = opts.MscnFollowTiles && !opts.TilesApplyTransforms;
            string name = $"mscn_t{tileX:D2}_{tileY:D2}";
            string objPath = Path.Combine(mdir, name + ".obj");

            if (followTilesRaw)
            {
                // Follow tiles in raw mode: skip global transforms and enforce legacy X parity at writer level
                // Do NOT apply project-local recentring either, to remain in the same space as raw tiles
                Console.WriteLine("[pm4-faces] MSCN follow-tiles: raw tile space; skip global transforms; writer forced X parity ON");
                ObjWriter.Write(objPath, local, new List<(int A, int B, int C)>(), legacyParity: false, projectLocal: false, forceFlipX: true);
            }
            else
            {
                // Standard: apply same mesh pipeline (project-local + global) so MSCN matches transformed meshes/tiles
                GeometryTransformService.ApplyProjectLocal(local, opts.ProjectLocal);
                GeometryTransformService.ApplyGlobal(local, ToTransformOptions(opts));
                Console.WriteLine("[pm4-faces] MSCN standard pipeline: project-local=" + opts.ProjectLocal + ", global transforms applied");
                ObjWriter.Write(objPath, local, new List<(int A, int B, int C)>(), legacyParity: false, projectLocal: false, forceFlipX: false);
            }

            rows.Add(string.Join(',',
                tileId.ToString(CultureInfo.InvariantCulture),
                tileX.ToString(CultureInfo.InvariantCulture),
                tileY.ToString(CultureInfo.InvariantCulture),
                local.Count.ToString(CultureInfo.InvariantCulture),
                Path.GetRelativePath(outDir, objPath).Replace("\\", "/")));
        }

        File.WriteAllLines(Path.Combine(mdir, "mscn_counts.csv"), rows);
        Console.WriteLine($"[pm4-faces] MSCN sidecar: tiles={groups.Count}, points={verts.Count}");
    }

    private static void ExportRenderMeshMerged(
        Pm4Scene scene,
        string tilesDir,
        Options opts)
    {
        string mergedPath = Path.Combine(tilesDir, "render_mesh.obj");
        Console.WriteLine($"[pm4-faces] Emitting merged render mesh (object-first): {mergedPath}");

        var surfacesAll = scene.Surfaces.Select((e, i) => (e, i)).ToList();
        var groups = surfacesAll
            .GroupBy(t => t.e.IndexCount)
            .OrderBy(g => g.Key);

        var verts = new List<Vector3>(Math.Max(8192, scene.Surfaces.Count * 2));
        var tris = new List<(int A, int B, int C)>(Math.Max(8192, scene.Indices.Count / 3));
        int written = 0, skipped = 0;

        foreach (var g in groups)
        {
            if (opts.SnapToPlane)
            {
                var g2lPlane = new Dictionary<long, int>(4096);
                foreach (var (entry, index) in g)
                {
                    int first = unchecked((int)entry.MsviFirstIndex);
                    int count = entry.IndexCount;
                    if (first < 0 || count < 3) continue;
                    int end = Math.Min(scene.Indices.Count, first + count);
                    int polyCount = end - first;
                    if (polyCount < 3) continue;

                    int[] poly = new int[polyCount];
                    for (int k = 0; k < polyCount; k++) poly[k] = scene.Indices[first + k];

                    long sKey = (long)(uint)index;

                    if (polyCount == 3)
                    {
                        EmitTriProjected(scene, entry, sKey, poly[0], poly[1], poly[2], opts.HeightScale, g2lPlane, verts, tris, ref written, ref skipped);
                    }
                    else if (polyCount == 4)
                    {
                        EmitTriProjected(scene, entry, sKey, poly[0], poly[1], poly[2], opts.HeightScale, g2lPlane, verts, tris, ref written, ref skipped);
                        EmitTriProjected(scene, entry, sKey, poly[0], poly[2], poly[3], opts.HeightScale, g2lPlane, verts, tris, ref written, ref skipped);
                    }
                    else
                    {
                        for (int i = 1; i + 1 < polyCount; i++)
                            EmitTriProjected(scene, entry, sKey, poly[0], poly[i], poly[i + 1], opts.HeightScale, g2lPlane, verts, tris, ref written, ref skipped);
                    }
                }
            }
            else
            {
                var g2l = new Dictionary<int, int>(1024);
                foreach (var (entry, _) in g)
                {
                    int first = unchecked((int)entry.MsviFirstIndex);
                    int count = entry.IndexCount;
                    if (first < 0 || count < 3) continue;
                    int end = Math.Min(scene.Indices.Count, first + count);
                    int polyCount = end - first;
                    if (polyCount < 3) continue;

                    int[] poly = new int[polyCount];
                    for (int k = 0; k < polyCount; k++) poly[k] = scene.Indices[first + k];

                    if (polyCount == 3)
                    {
                        EmitTriMapped(scene, poly[0], poly[1], poly[2], g2l, verts, tris, ref written, ref skipped);
                    }
                    else if (polyCount == 4)
                    {
                        EmitTriMapped(scene, poly[0], poly[1], poly[2], g2l, verts, tris, ref written, ref skipped);
                        EmitTriMapped(scene, poly[0], poly[2], poly[3], g2l, verts, tris, ref written, ref skipped);
                    }
                    else
                    {
                        for (int i = 1; i + 1 < polyCount; i++)
                            EmitTriMapped(scene, poly[0], poly[i], poly[i + 1], g2l, verts, tris, ref written, ref skipped);
                    }
                }
            }
        }

        // Apply project-local recentering then configurable global transform for merged render mesh
        GeometryTransformService.ApplyProjectLocal(verts, opts.ProjectLocal);
        GeometryTransformService.ApplyGlobal(verts, ToTransformOptions(opts));
        if (opts.FlipXEnabled ^ opts.FlipYEnabled)
        {
            for (int i = 0; i < tris.Count; i++)
            {
                var t = tris[i];
                tris[i] = (t.A, t.C, t.B);
            }
        }

        // Writers receive geometry already transformed
        ObjWriter.Write(mergedPath, verts, tris, legacyParity: false, projectLocal: false, forceFlipX: false);
        if (opts.ExportGltf)
            GltfWriter.WriteGltf(Path.ChangeExtension(mergedPath, ".gltf"), verts, tris, legacyParity: false, projectLocal: false, forceFlipX: false);
        if (opts.ExportGlb)
            GltfWriter.WriteGlb(Path.ChangeExtension(mergedPath, ".glb"), verts, tris, legacyParity: false, projectLocal: false, forceFlipX: false);

        Console.WriteLine($"[pm4-faces] Merged render mesh faces: written={written} skipped={skipped}");
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
            { skipped++; continue; }
            if (la == lb || lb == lc || lc == la)
            { skipped++; continue; }
            localTris.Add((la, lb, lc));
            written++;
        }
        // Apply project-local recentering then configurable global transform for index-range exports
        GeometryTransformService.ApplyProjectLocal(localVerts, opts.ProjectLocal);
        GeometryTransformService.ApplyGlobal(localVerts, ToTransformOptions(opts));
        if (opts.FlipXEnabled ^ opts.FlipYEnabled)
        {
            for (int i = 0; i < localTris.Count; i++)
            {
                var t = localTris[i];
                localTris[i] = (t.A, t.C, t.B);
            }
        }
        ObjWriter.Write(objPath, localVerts, localTris, legacyParity: false, projectLocal: false, forceFlipX: false);
        if (opts.ExportGltf)
        {
            GltfWriter.WriteGltf(Path.ChangeExtension(objPath, ".gltf"), localVerts, localTris, legacyParity: false, projectLocal: false, forceFlipX: false);
        }
        if (opts.ExportGlb)
        {
            GltfWriter.WriteGlb(Path.ChangeExtension(objPath, ".glb"), localVerts, localTris, legacyParity: false, projectLocal: false, forceFlipX: false);
        }
        facesWritten = written;
        facesSkipped = skipped;
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
                    AssembleAndWrite(scene, allItems, objPath, opts with { SnapToPlane = false }, writerLegacyParity: false, out writtenFaces, out skippedFaces);
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
                        FlipX = opts.FlipXEnabled
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
                    AssembleAndWrite(scene, items, objPath, opts with { SnapToPlane = false }, writerLegacyParity: false, out writtenFaces, out skippedFaces);
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
                        FlipX = opts.FlipXEnabled
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
                    AssembleAndWrite(scene, items, objPath, opts with { SnapToPlane = false }, writerLegacyParity: false, out writtenFaces, out skippedFaces);
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
        AssembleAndWrite(scene, items, objPath, opts with { SnapToPlane = false }, writerLegacyParity: false, out var written, out var skipped);

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
        bool writerLegacyParity,
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
                if (first < 0 || count < 3) continue;

                int end = Math.Min(scene.Indices.Count, first + count);
                int polyCount = end - first;
                if (polyCount < 3) continue;

                int[] poly = new int[polyCount];
                for (int k = 0; k < polyCount; k++) poly[k] = scene.Indices[first + k];

                if (polyCount == 3)
                {
                    EmitTriMapped(scene, poly[0], poly[1], poly[2], g2l, localVerts, localTris, ref written, ref skipped);
                }
                else if (polyCount == 4)
                {
                    EmitTriMapped(scene, poly[0], poly[1], poly[2], g2l, localVerts, localTris, ref written, ref skipped);
                    EmitTriMapped(scene, poly[0], poly[2], poly[3], g2l, localVerts, localTris, ref written, ref skipped);
                }
                else
                {
                    // Triangle fan: (0, i, i+1)
                    for (int i = 1; i + 1 < polyCount; i++)
                        EmitTriMapped(scene, poly[0], poly[i], poly[i + 1], g2l, localVerts, localTris, ref written, ref skipped);
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
                if (first < 0 || count < 3) continue;

                int end = Math.Min(scene.Indices.Count, first + count);
                int polyCount = end - first;
                if (polyCount < 3) continue;

                int[] poly = new int[polyCount];
                for (int k = 0; k < polyCount; k++) poly[k] = scene.Indices[first + k];

                long sKey = (long)(uint)idx; // stable per-surface key

                if (polyCount == 3)
                {
                    EmitTriProjected(scene, entry, sKey, poly[0], poly[1], poly[2], opts.HeightScale, g2l, localVerts, localTris, ref written, ref skipped);
                }
                else if (polyCount == 4)
                {
                    EmitTriProjected(scene, entry, sKey, poly[0], poly[1], poly[2], opts.HeightScale, g2l, localVerts, localTris, ref written, ref skipped);
                    EmitTriProjected(scene, entry, sKey, poly[0], poly[2], poly[3], opts.HeightScale, g2l, localVerts, localTris, ref written, ref skipped);
                }
                else
                {
                    // Triangle fan: (0, i, i+1)
                    for (int i = 1; i + 1 < polyCount; i++)
                        EmitTriProjected(scene, entry, sKey, poly[0], poly[i], poly[i + 1], opts.HeightScale, g2l, localVerts, localTris, ref written, ref skipped);
                }
            }
        }
        // Apply project-local recentering then configurable global transform
        GeometryTransformService.ApplyProjectLocal(localVerts, opts.ProjectLocal);
        GeometryTransformService.ApplyGlobal(localVerts, ToTransformOptions(opts));
        if (opts.FlipXEnabled ^ opts.FlipYEnabled)
        {
            // Preserve normals by swapping winding when a reflection is applied
            for (int i = 0; i < localTris.Count; i++)
            {
                var t = localTris[i];
                localTris[i] = (t.A, t.C, t.B);
            }
        }

        // Writers receive geometry already transformed; optionally apply legacy parity at writer level
        ObjWriter.Write(objPath, localVerts, localTris, legacyParity: writerLegacyParity, projectLocal: false, forceFlipX: false);
        if (opts.ExportGltf)
        {
            GltfWriter.WriteGltf(Path.ChangeExtension(objPath, ".gltf"), localVerts, localTris, legacyParity: writerLegacyParity, projectLocal: false, forceFlipX: false);
        }
        if (opts.ExportGlb)
        {
            GltfWriter.WriteGlb(Path.ChangeExtension(objPath, ".glb"), localVerts, localTris, legacyParity: writerLegacyParity, projectLocal: false, forceFlipX: false);
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

    // Helper to emit one triangle (non-snap) with mapping and degenerate filtering
    private static void EmitTriMapped(
        Pm4Scene scene,
        int ga,
        int gb,
        int gc,
        Dictionary<int, int> g2l,
        List<Vector3> localVerts,
        List<(int A, int B, int C)> localTris,
        ref int written,
        ref int skipped)
    {
        if (!TryMap(scene, ga, g2l, localVerts, out var la) ||
            !TryMap(scene, gb, g2l, localVerts, out var lb) ||
            !TryMap(scene, gc, g2l, localVerts, out var lc))
        { skipped++; return; }
        if (la == lb || lb == lc || lc == la)
        { skipped++; return; }
        localTris.Add((la, lb, lc));
        written++;
    }

    // Helper to emit one triangle (snap-to-plane) with mapping and degenerate filtering
    private static void EmitTriProjected(
        Pm4Scene scene,
        MsurChunk.Entry entry,
        long surfaceKey,
        int ga,
        int gb,
        int gc,
        float heightScale,
        Dictionary<long, int> g2l,
        List<Vector3> localVerts,
        List<(int A, int B, int C)> localTris,
        ref int written,
        ref int skipped)
    {
        if (!TryMapProjected(scene, entry, surfaceKey, ga, heightScale, g2l, localVerts, out var la) ||
            !TryMapProjected(scene, entry, surfaceKey, gb, heightScale, g2l, localVerts, out var lb) ||
            !TryMapProjected(scene, entry, surfaceKey, gc, heightScale, g2l, localVerts, out var lc))
        { skipped++; return; }
        if (la == lb || lb == lc || lc == la)
        { skipped++; return; }
        localTris.Add((la, lb, lc));
        written++;
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

    private static uint CK24(uint key)
    {
        return (key & 0xFFFFFF00u) >> 8;
    }

    private static int? DominantTileIdFor(Pm4Scene scene, IEnumerable<MsurChunk.Entry> entries)
    {
        if (scene.TileIndexOffsetByTileId.Count == 0)
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

    private static void ApplyProjectLocal(List<Vector3> verts, bool projectLocal)
    {
        if (!projectLocal || verts == null || verts.Count == 0) return;
        double sx = 0, sy = 0, sz = 0;
        for (int i = 0; i < verts.Count; i++)
        {
            sx += verts[i].X;
            sy += verts[i].Y;
            sz += verts[i].Z;
        }
        double inv = 1.0 / Math.Max(1, verts.Count);
        var mean = new Vector3((float)(sx * inv), (float)(sy * inv), (float)(sz * inv));
        for (int i = 0; i < verts.Count; i++)
        {
            verts[i] = verts[i] - mean;
        }
    }

    private static void CanonicalizeMscnAxes(List<Vector3> verts, string basis)
    {
        if (verts == null || verts.Count == 0) return;
        for (int i = 0; i < verts.Count; i++)
        {
            var v = verts[i];
            // Basis selection:
            //  - legacy: negate X and Y, preserve Z  => (-X, -Y, Z)
            //  - remap:  swap X and Y, preserve signs => (Y, X, Z)
            switch ((basis ?? "legacy").Trim().ToLowerInvariant())
            {
                case "remap":
                    verts[i] = new Vector3(v.Y, v.X, v.Z);
                    break;
                case "legacy":
                default:
                    if (!string.IsNullOrWhiteSpace(basis) && basis.Trim().ToLowerInvariant() != "legacy")
                        Console.WriteLine($"[pm4-faces] Warning: unknown MSCN basis '{basis}', defaulting to 'legacy'.");
                    verts[i] = new Vector3(-v.X, -v.Y, v.Z);
                    break;
            }
        }
    }

    private static void ApplyMscnPreTransform(List<Vector3> verts, int rotZDeg, string flip, string basis)
    {
        if (verts == null || verts.Count == 0) return;

        // First, bring MSCN points into the canonical basis used by meshes (prior fix)
        CanonicalizeMscnAxes(verts, basis);

        // Normalize rotation to right angles
        int r = rotZDeg;
        r %= 360; if (r < 0) r += 360;
        int rNorm = ((int)MathF.Round(r / 90f)) * 90;
        rNorm %= 360;
        if (rNorm != r && rotZDeg != 0)
        {
            Console.WriteLine($"[pm4-faces] Note: --mscn-pre-rotz normalized from {rotZDeg} to {rNorm} (right-angle).");
        }

        // Build rotation (Z only)
        if (rNorm != 0)
        {
            float rad = rNorm * (MathF.PI / 180f);
            var rz = Matrix4x4.CreateRotationZ(rad);
            for (int i = 0; i < verts.Count; i++)
            {
                verts[i] = Vector3.Transform(verts[i], rz);
            }
        }

        // Determine flip axes
        string f = (flip ?? "none").Trim().ToLowerInvariant();
        bool fx = false, fy = false, fz = false;
        switch (f)
        {
            case "none":
            case "":
                break;
            case "x": fx = true; break;
            case "y": fy = true; break;
            case "z": fz = true; break;
            case "xy": fx = fy = true; break;
            case "xz": fx = fz = true; break;
            case "yz": fy = fz = true; break;
            default:
                Console.WriteLine($"[pm4-faces] Warning: unknown --mscn-pre-flip '{flip}', expected none|x|y|z|xy|xz|yz. Ignoring.");
                break;
        }

        if (fx || fy || fz)
        {
            for (int i = 0; i < verts.Count; i++)
            {
                var v = verts[i];
                if (fx) v.X = -v.X;
                if (fy) v.Y = -v.Y;
                if (fz) v.Z = -v.Z;
                verts[i] = v;
            }
        }
    }

    private static void ApplyGlobalTransform(List<Vector3> verts, Options opts)
    {
        if (verts == null || verts.Count == 0) return;

        bool doFlip = opts.FlipXEnabled;
        bool doRotX = MathF.Abs(opts.RotXDeg) > 1e-6f;
        bool doRotY = MathF.Abs(opts.RotYDeg) > 1e-6f;
        bool doRotZ = MathF.Abs(opts.RotZDeg) > 1e-6f;
        bool doTrans = MathF.Abs(opts.TranslateX) > 1e-6f || MathF.Abs(opts.TranslateY) > 1e-6f || MathF.Abs(opts.TranslateZ) > 1e-6f;

        Matrix4x4 rx = doRotX ? Matrix4x4.CreateRotationX(opts.RotXDeg * (MathF.PI / 180f)) : Matrix4x4.Identity;
        Matrix4x4 ry = doRotY ? Matrix4x4.CreateRotationY(opts.RotYDeg * (MathF.PI / 180f)) : Matrix4x4.Identity;
        Matrix4x4 rz = doRotZ ? Matrix4x4.CreateRotationZ(opts.RotZDeg * (MathF.PI / 180f)) : Matrix4x4.Identity;
        Vector3 t = doTrans ? new Vector3(opts.TranslateX, opts.TranslateY, opts.TranslateZ) : default;

        for (int i = 0; i < verts.Count; i++)
        {
            var v = verts[i];
            if (doFlip) v = new Vector3(-v.X, v.Y, v.Z);
            if (doRotX) v = Vector3.Transform(v, rx);
            if (doRotY) v = Vector3.Transform(v, ry);
            if (doRotZ) v = Vector3.Transform(v, rz);
            if (doTrans) v += t;
            verts[i] = v;
        }
    }
}
