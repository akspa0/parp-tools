using System.Numerics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Maps;
using WowViewer.Core.Mdx;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;
using WowViewer.Core.Wmo;

internal static class Pm4CollisionDumper
{
    public static void Run(string[] args)
    {
        string? tile = GetOption(args, "--tile", "-t");
        string? oidText = GetOption(args, "--oid", "-o");
        string? pm4Path = GetOption(args, "--pm4", "-p");
        string? placementsPath = GetOption(args, "--placements", "-a");
        string? archiveRoot = GetOption(args, "--archive-root", "-r");

        if (string.IsNullOrWhiteSpace(tile) && string.IsNullOrWhiteSpace(pm4Path))
        { Console.Error.WriteLine("Error: --tile or --pm4 required."); return; }
        if (string.IsNullOrWhiteSpace(oidText))
        { Console.Error.WriteLine("Error: --oid required."); return; }
        if (!ushort.TryParse(oidText, out ushort targetOid))
        { Console.Error.WriteLine("Error: --oid must be 16-bit integer."); return; }

        string devDir = @"I:\parp\parp-tools\wow-viewer\test_data\development\World\Maps\development";
        if (string.IsNullOrWhiteSpace(pm4Path))
        {
            string[] parts = tile!.Split('_');
            if (parts.Length != 2) { Console.Error.WriteLine("Error: --tile must be X_Y"); return; }
            pm4Path = Path.Combine(devDir, $"development_{tile}.pm4");
        }
        if (!File.Exists(pm4Path)) { Console.Error.WriteLine("Error: PM4 not found."); return; }
        if (string.IsNullOrWhiteSpace(archiveRoot)) { Console.Error.WriteLine("Error: --archive-root required."); return; }

        string fullArchiveRoot = Path.GetFullPath(archiveRoot);
        string adtDir = string.IsNullOrWhiteSpace(placementsPath) ? devDir : placementsPath;

        Console.WriteLine($"PM4: {pm4Path}");
        Console.WriteLine($"OID: {targetOid} (0x{targetOid:X4})");
        Console.WriteLine();

        // STEP 1: Read PM4
        Console.WriteLine("--- PM4 Surfaces ---");
        var pm4Doc = Pm4ResearchReader.ReadFile(pm4Path);
        var chunks = pm4Doc.KnownChunks;
        var targetSurfaces = chunks.Msur.Where(s => s.Ck24ObjectId == targetOid).ToList();
        if (targetSurfaces.Count == 0) { Console.WriteLine("No surfaces found."); return; }

        var ck24Type = targetSurfaces[0].Ck24Type;
        Console.WriteLine($"Surfaces: {targetSurfaces.Count} (ck24Type=0x{ck24Type:X2})");

        var mscnRefs = targetSurfaces.Select(s => s.MscnRefIndex).Distinct().Where(i => i < chunks.Mscn.Count).ToList();
        if (mscnRefs.Count > 0)
        {
            float mx = mscnRefs.Min(i => chunks.Mscn[(int)i].X), Mx = mscnRefs.Max(i => chunks.Mscn[(int)i].X);
            float my = mscnRefs.Min(i => chunks.Mscn[(int)i].Y), My = mscnRefs.Max(i => chunks.Mscn[(int)i].Y);
            // MSCN (X,Y) maps to ADT (Y,X) — swap
            Console.WriteLine($"MSCN box center (as ADT): ({(my+My)/2:F1}, {(mx+Mx)/2:F1})");
        }

        foreach (var grp in targetSurfaces.GroupBy(s => (s.AttributeMask, s.GroupKey)).OrderBy(g => g.Key.AttributeMask))
        {
            var sl = grp.ToList();
            float ax = sl.Average(s => s.Normal.X), ay = sl.Average(s => s.Normal.Y), az = sl.Average(s => s.Normal.Z);
            var hs = sl.Select(s => s.Height).Distinct().OrderBy(h => h).ToList();
            Console.WriteLine($"  AM=0x{grp.Key.AttributeMask:X2} GK=0x{grp.Key.GroupKey:X2}: {sl.Count} surfaces");
            Console.WriteLine($"    avg_n=({ax:F4},{ay:F4},{az:F4}) h=[{hs.Min():F1},{hs.Max():F1}]");
        }

        // STEP 2: Find placement in _obj0.adt
        Console.WriteLine("\n--- Placement ---");
        string? obj0Path = null;
        string pm4Fn = Path.GetFileNameWithoutExtension(pm4Path);
        var pm4Parts = pm4Fn.Split('_');
        if (pm4Parts.Length >= 3)
        {
            string expected = $"development_{pm4Parts[^2]}_{pm4Parts[^1]}_obj0.adt";
            string candidate = Path.Combine(adtDir, expected);
            if (File.Exists(candidate)) obj0Path = candidate;
        }
        obj0Path ??= Directory.EnumerateFiles(adtDir, "*_obj0.adt").FirstOrDefault();
        if (obj0Path == null) { Console.WriteLine("No _obj0.adt found."); return; }
        Console.WriteLine($"Placements file: {Path.GetFileName(obj0Path)}");

        var placements = AdtPlacementReader.Read(obj0Path);
        float minX = mscnRefs.Min(i => chunks.Mscn[(int)i].X), maxX = mscnRefs.Max(i => chunks.Mscn[(int)i].X);
        float minY = mscnRefs.Min(i => chunks.Mscn[(int)i].Y), maxY = mscnRefs.Max(i => chunks.Mscn[(int)i].Y);
        float adtX = (minY + maxY) / 2, adtY = (minX + maxX) / 2; // MSCN swap
        float mapOrigin = 17066.666f;

        AdtWorldModelPlacement? bestWmo = null;
        AdtModelPlacement? bestM2 = null;
        double bestDist = double.MaxValue;

        foreach (var p in placements.WorldModelPlacements)
        {
            double d = Math.Sqrt(Math.Pow(adtX - (mapOrigin - p.Position.Y), 2) + Math.Pow(adtY - (mapOrigin - p.Position.X), 2));
            if (d < bestDist) { bestDist = d; bestWmo = p; bestM2 = null; }
        }
        foreach (var p in placements.ModelPlacements)
        {
            double d = Math.Sqrt(Math.Pow(adtX - (mapOrigin - p.Position.Y), 2) + Math.Pow(adtY - (mapOrigin - p.Position.X), 2));
            if (d < bestDist) { bestDist = d; bestM2 = p; bestWmo = null; }
        }

        bool isM2 = bestM2 is not null;
        string assetPath = isM2 ? bestM2!.ModelPath : bestWmo!.ModelPath;
        Console.WriteLine($"Closest: dist={bestDist:F2} ({((isM2) ? "M2" : "WMO")})");
        Console.WriteLine($"  Asset: {assetPath}");

        // Read asset from archive
        string normPath = assetPath.Replace('\\', '/').Trim().TrimStart('/').ToLowerInvariant();
        var opts = new ArchiveCatalogBootstrapOptions();
        byte[] rawBytes;
        try { rawBytes = ArchiveVirtualFileReader.ReadVirtualFile(normPath, [fullArchiveRoot], opts); }
        catch (Exception ex) { Console.WriteLine($"Error reading asset: {ex.Message}"); return; }

        if (isM2)
        {
            // M2 collision
            Console.WriteLine("\n--- M2 Collision ---");
            using var ms = new MemoryStream(rawBytes);
            
            MdxCollisionMesh? mesh = null;
            try
            {
                var m2f = MdxCollisionReader.Read(ms, normPath);
                mesh = m2f.Collision;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading M2 collision: {ex.Message}");
                Console.WriteLine("(PM4 targets retail MDLX M2 files; MD20-era models are not supported)");
                return;
            }
            if (mesh is null) { Console.WriteLine("No collision mesh."); return; }

            Console.WriteLine($"Triangles: {mesh.TriangleCount} Vertices: {mesh.VertexCount} FacetNormals: {mesh.FacetNormalCount}");
            var ng = mesh.FacetNormals.GroupBy(n => GetOctant(n)).OrderByDescending(g => g.Count()).ToList();
            foreach (var g2 in ng.Take(5))
                Console.WriteLine($"  {g2.Key}: {g2.Count()} avg_n=({g2.Average(n=>n.X):F3},{g2.Average(n=>n.Y):F3},{g2.Average(n=>n.Z):F3})");
            foreach (var fn in mesh.FacetNormals.Take(10))
                Console.WriteLine($"  n=({fn.X:F3},{fn.Y:F3},{fn.Z:F3})");

            Console.WriteLine($"\n=== SUMMARY ===");
            Console.WriteLine($"PM4 surfaces: {targetSurfaces.Count}");
            Console.WriteLine($"M2 triangles: {mesh.TriangleCount}");
            Console.WriteLine($"Ratio: {(double)mesh.TriangleCount / Math.Max(1, targetSurfaces.Count):F1}x");
        }
        else
        {
            // WMO collision (existing)
            Console.WriteLine("\n--- WMO Groups ---");
            using var ms = new MemoryStream(rawBytes);
            var ws = WmoSummaryReader.Read(ms, normPath);
            Console.WriteLine($"Groups: {ws.ReportedGroupCount}");

            string root = normPath[..^4];
            int totalTri = 0;
            for (int gi = 0; gi < ws.ReportedGroupCount; gi++)
            {
                string gp = $"{root}_{gi:D3}.wmo";
                byte[] gb;
                try { gb = ArchiveVirtualFileReader.ReadVirtualFile(gp, [fullArchiveRoot], opts); }
                catch { continue; }

                using var gsm = new MemoryStream(gb);
                var md = WmoGroupMeshDetailReader.Read(gsm, gp);
                int tc = md.Indices.Count / 3;
                totalTri += tc;

                var ns = new List<(Vector3 N, byte F, byte M)>();
                for (int ti = 0; ti + 2 < md.Indices.Count; ti += 3)
                {
                    int i0 = md.Indices[ti], i1 = md.Indices[ti + 1], i2 = md.Indices[ti + 2];
                    if (i0 >= md.Vertices.Count || i1 >= md.Vertices.Count || i2 >= md.Vertices.Count) continue;
                    var e1 = md.Vertices[i1] - md.Vertices[i0];
                    var e2 = md.Vertices[i2] - md.Vertices[i0];
                    var n = Vector3.Normalize(Vector3.Cross(e1, e2));
                    int fi = ti / 3;
                    byte fl = fi < md.FaceMaterials.Count ? md.FaceMaterials[fi].Flags : (byte)0;
                    byte mt = fi < md.FaceMaterials.Count ? md.FaceMaterials[fi].MaterialId : (byte)0;
                    ns.Add((n, fl, mt));
                }

                var ngs = ns.GroupBy(f => GetOctant(f.N)).OrderByDescending(g => g.Count()).ToList();
                Console.WriteLine($"  Group {gi}: {tc} triangles");
                foreach (var g2 in ngs.Take(3))
                    Console.WriteLine($"    {g2.Key}: {g2.Count()} avg_n=({g2.Average(f=>f.N.X):F3},{g2.Average(f=>f.N.Y):F3},{g2.Average(f=>f.N.Z):F3})");
            }

            Console.WriteLine($"\n=== SUMMARY ===");
            Console.WriteLine($"PM4 surfaces: {targetSurfaces.Count}");
            Console.WriteLine($"WMO triangles: {totalTri}");
            Console.WriteLine($"Ratio: {(double)totalTri / Math.Max(1, targetSurfaces.Count):F1}x");
        }
    }

    private static string GetOctant(Vector3 n) => $"{(n.X>=0?'+':'-')}{(n.Y>=0?'+':'-')}{(n.Z>=0?'+':'-')}";

    private static string? GetOption(string[] args, string longName, string shortName)
    {
        for (int i = 0; i < args.Length; i++)
            if ((args[i] == longName || args[i] == shortName) && i + 1 < args.Length)
                return args[i + 1];
        return null;
    }
}
