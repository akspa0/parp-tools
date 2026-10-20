using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

if (args.Length == 3 && args[0] == "--cross-tile")
{
    if (File.Exists(args[2])) throw new IOException("Report exists; refusing overwrite.");
    var timer = Stopwatch.StartNew();
    var map = new List<Pm4MapAuditInput>();
    var fingerprints = new List<object>();
    var mapHashInput = new StringBuilder();
    foreach (string path in Directory.GetFiles(args[1], "*.pm4").Order(StringComparer.Ordinal))
    {
        byte[] bytes = File.ReadAllBytes(path);
        string hash = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        string name = Path.GetFileName(path);
        mapHashInput.Append(name).Append(':').Append(hash).Append('\n');
        fingerprints.Add(new { File = name, Bytes = bytes.Length, Sha256 = hash });
        var doc = Pm4ResearchReader.Read(bytes, path);
        if (doc.Diagnostics.Count != 0) throw new InvalidDataException("Reader diagnostics require review before complete address analysis.");
        map.Add(new Pm4MapAuditInput(name, doc));
    }
    if (map.Count == 0) throw new InvalidDataException("No map inputs.");
    var addressRows = Pm4CrossTileLinkProbe.Analyze(map);
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[2]))!);
    File.WriteAllText(args[2], JsonSerializer.Serialize(new
    {
        Warning = "Address hypotheses checked against all map files, geometry and back-links; not complete PM4 semantics or object partition.",
        FileCount = map.Count,
        MapSha256 = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(mapHashInput.ToString()))).ToLowerInvariant(),
        Rows = addressRows, Files = fingerprints, ElapsedSeconds = timer.Elapsed.TotalSeconds,
    }, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"Cross-tile address probe: {map.Count} files, {addressRows.Count} strata, {timer.Elapsed.TotalSeconds:F3}s.");
    return;
}
if (args.Length != 2) throw new ArgumentException("Usage: <complete-map-pm4-directory> <new-report.json>, or --cross-tile <directory> <new-report.json>");
if (File.Exists(args[1])) throw new IOException("Report already exists; refusing overwrite.");
string[] paths = Directory.GetFiles(args[0], "*.pm4").Order(StringComparer.Ordinal).ToArray();
if (paths.Length == 0) throw new InvalidDataException("No PM4 inputs.");
var clock = Stopwatch.StartNew();
var files = new List<object>();
var allRows = new List<Pm4LinkGeometryRow>();
var identity = new StringBuilder();
foreach (string path in paths)
{
    // A hard bound prevents an unexpectedly heavy analysis from becoming a long unattended job.
    if (clock.Elapsed.TotalSeconds > 30) throw new TimeoutException("30-second analysis budget reached; no complete-map report emitted.");
    byte[] bytes = File.ReadAllBytes(path);
    string hash = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    string name = Path.GetFileName(path);
    identity.Append(name).Append(':').Append(hash).Append('\n');
    var doc = Pm4ResearchReader.Read(bytes, path);
    var result = Pm4LinkGeometryProbe.Analyze(doc);
    allRows.AddRange(result.Rows);
    files.Add(new
    {
        File = name, Bytes = bytes.Length, Sha256 = hash, doc.Diagnostics,
        Chunks = doc.Chunks.Select(c => new { c.Signature, c.Size }),
        result.Surfaces, result.Links, result.InvalidPolygonWindows, result.InvalidLinkWindows, result.InvalidSurfaceReferences,
        ZeroCk24Surfaces = doc.KnownChunks.Msur.Count(s => s.Ck24 == 0),
        RawWordZeroSurfaces = doc.KnownChunks.Msur.Count(s => s.PackedParams == 0),
        result.OutOfRangeReferences,
        PathIndexCounts = doc.KnownChunks.Mslk.Where(l => l.MspiFirstIndex >= 0).GroupBy(l => l.MspiIndexCount)
            .ToDictionary(g => g.Key.ToString(), g => g.Count()),
    });
}
var rows = allRows.GroupBy(r => (r.Relation, r.Ck24Relation, r.RawType, r.RawSubtype, r.NegativePathStart))
    .Select(g => new Pm4LinkGeometryRow(g.Key.Relation, g.Key.Ck24Relation, g.Key.RawType, g.Key.RawSubtype, g.Key.NegativePathStart,
        g.Sum(r => r.Count), g.Sum(r => r.SharedVertex), g.Sum(r => r.SharedBoundaryEdge), g.Sum(r => r.SelfReference),
        g.Sum(r => r.ValidPathWindows), g.Sum(r => r.PathTouchesOwner), g.Sum(r => r.PathTouchesReference),
        g.Sum(r => r.PathSharesOwnerEdge), g.Sum(r => r.PathSharesReferenceEdge)))
    .OrderBy(r => r.Relation, StringComparer.Ordinal).ThenBy(r => r.Ck24Relation, StringComparer.Ordinal)
    .ThenBy(r => r.RawType).ThenBy(r => r.RawSubtype).ThenBy(r => r.NegativePathStart).ToArray();
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[1]))!);
File.WriteAllText(args[1], JsonSerializer.Serialize(new
{
    Warning = "All map files included. This checks decoded window/reference geometry, not complete object partition or full format semantics. Reader diagnostics remain visible.",
    FileCount = paths.Length, MapSha256 = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(identity.ToString()))).ToLowerInvariant(),
    GeometryToleranceYards = 0.01f,
    BoundaryTest = "Pairs of consecutive polygon vertices coincide (either direction); does not detect collinear partial-edge overlaps.",
    Rows = rows, Files = files, ElapsedSeconds = clock.Elapsed.TotalSeconds,
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Measured all {paths.Length} PM4 files in {clock.Elapsed.TotalSeconds:F3}s; {rows.Length} relation/field strata. No asset matches asserted.");
