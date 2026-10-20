using System.Numerics;
using System.Security.Cryptography;
using System.Text.Json;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

// Investigation only. Existing readers and matching behavior remain unchanged.
// Usage: <map-directory> <report.json> <one-to-five-explicit-pm4-filenames>
if (args.Length is < 3 or > 7)
    throw new ArgumentException("Supply a map directory, report path, and one to five PM4 filenames.");

var tiles = new List<object>();
var controls = new List<object>();
foreach (string name in args.Skip(2))
{
    if (Path.GetFileName(name) != name || !name.EndsWith(".pm4", StringComparison.OrdinalIgnoreCase))
        throw new ArgumentException("Samples must be explicit PM4 filenames, without directories or wildcards.");

    string pm4Path = Path.Combine(args[0], name);
    string adtPath = Pm4CoordinateService.TryGetObj0PathForPm4(pm4Path)
        ?? throw new FileNotFoundException($"No canonical companion for {name}.");
    var chunks = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
    var catalog = AdtPlacementReader.Read(adtPath);
    var placements = catalog.WorldModelPlacements
        .Select(p => new Placement("WMO", p.ModelPath, p.UniqueId, p.Position))
        .Concat(catalog.ModelPlacements.Select(p => new Placement("M2", p.ModelPath, p.UniqueId, p.Position)))
        .ToArray();
    var groups = new List<object>();

    foreach (var group in chunks.Msur.GroupBy(s => s.PackedParams).OrderBy(g => g.Key))
    {
        uint raw = group.Key;
        var surfaces = group.ToList();
        var vertices = Pm4PlacementMath.CollectSurfaceVertices(chunks.Msvt, chunks.Msvi, surfaces);
        if (vertices.Count == 0)
            continue;
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        foreach (Vector3 vertex in vertices)
        {
            Vector3 p = Pm4CoordinateService.Pm4LocalToAdtPlacement(vertex);
            min = Vector3.Min(min, p);
            max = Vector3.Max(max, p);
        }

        byte band = (byte)(raw >> 24);
        string route = raw == 0 ? "M2-connected-components" : band is 0x40 or 0x41 ? "M2-keyed" : "WMO-keyed";
        var inside = placements.Where(p => p.Position.X >= min.X - 1 && p.Position.X <= max.X + 1
            && p.Position.Y >= min.Y - 1 && p.Position.Y <= max.Y + 1).ToArray();
        var exact = raw == 0 ? [] : inside.Where(p => BitConverter.SingleToUInt32Bits(p.Position.Z) == raw).ToArray();
        string[] exactKinds = exact.Select(p => p.Kind).Distinct().ToArray();
        string? supportedKind = exactKinds.Length == 1 ? exactKinds[0] : null;
        bool? conflict = supportedKind is null ? null : (route.StartsWith("M2") ? "M2" : "WMO") != supportedKind;
        var fingerprint = Pm4SurfaceCorrelationExtractor.ExtractFromPm4Group(
            chunks.Msvt, chunks.Msvi, surfaces, raw, band, $"{name}:{raw:X8}", name, "pm4");

        groups.Add(new
        {
            Raw = $"0x{raw:X8}", PlacementZ = BitConverter.UInt32BitsToSingle(raw), HeightByte = $"0x{band:X2}",
            CurrentResolverRoute = route, Surfaces = surfaces.Count, Vertices = vertices.Count,
            Triangles = fingerprint?.TriangleCount,
            BoundsMin = new[] { min.X, min.Y, min.Z }, BoundsMax = new[] { max.X, max.Y, max.Z },
            FootprintCandidates = inside.Length, ExactZCandidates = exact, SupportedKind = supportedKind,
            RoutingConflictsWithExactZKind = conflict,
        });

        // Positive/negative gate controls reuse real geometry, NOT real asset identities.
        // Only the type byte / declared asset kind changes; identical geometry must self-score 1.
        if (fingerprint != null && controls.Count == 0)
        {
            foreach (float height in new[] { 5f, 20f, 100f, -100f, 0f })
            {
                byte heightByte = (byte)(BitConverter.SingleToUInt32Bits(height) >> 24);
                foreach (string kind in new[] { "wmo", "m2" })
                {
                    var query = fingerprint with { AssetKind = "m2", Ck24Type = heightByte };
                    var candidate = fingerprint with { AssetId = "synthetic-self-control", AssetPath = "synthetic-self-control", AssetKind = kind };
                    var match = Pm4SurfaceCorrelationMatcher.MatchOne(query, new[] { candidate }, SurfaceMatchOptions.Default);
                    controls.Add(new
                    {
                        SyntheticHeight = height, HeightByte = $"0x{heightByte:X2}", QueryDeclaredKind = "m2",
                        ReferenceDeclaredKind = kind, match.Status, match.TopScore,
                        SelectedKind = match.TopCandidate?.Candidate.AssetKind,
                    });
                }
            }
        }
    }

    tiles.Add(new
    {
        Pm4 = Identity(pm4Path), Adt = Identity(adtPath),
        WmoPlacements = catalog.WorldModelPlacements.Count, M2Placements = catalog.ModelPlacements.Count,
        Surfaces = chunks.Msur.Count, Groups = groups,
        UnkeyedRemainder = SparseM2Probe.Measure(chunks, placements),
    });
}

var report = new
{
    Purpose = "Bounded height-byte routing investigation; not asset-identification or visual proof.",
    Method = "Existing readers; canonical placement XY footprint +/-1 yard; bit-exact nonzero placement Z; all candidates retained.",
    Limits = "Same-Z/footprint correspondence is evidence, not guaranteed unique identity. Zero remainder is not an object. Synthetic controls deliberately reuse geometry.",
    Tiles = tiles, SyntheticMatcherControls = controls,
};
string output = Path.GetFullPath(args[1]);
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
File.WriteAllText(output, JsonSerializer.Serialize(report, new JsonSerializerOptions
{
    WriteIndented = true,
    IncludeFields = true,
}));
Console.WriteLine($"Wrote {tiles.Count} paired tile reports and {controls.Count} synthetic gate controls to {args[1]}.");

static object Identity(string path) => new
{
    Path = Path.GetRelativePath(Directory.GetCurrentDirectory(), path).Replace('\\', '/'),
    Bytes = new FileInfo(path).Length,
    Sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant(),
};

internal sealed record Placement(string Kind, string ModelPath, int UniqueId, Vector3 Position);
