using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

bool edgesOnly = args.Length == 3 && args[0] == "--edges-only";
bool edgeIdentity = args.Length == 3 && args[0] == "--edge-identity";
bool mshd = args.Length == 3 && args[0] == "--mshd";
bool placements = args.Length == 3 && args[0] == "--placements";
bool m2Partition = args.Length == 3 && args[0] == "--m2-partition";
bool pathNodes = args.Length == 3 && args[0] == "--path-nodes";
if (args.Length == 3 && args[0] == "--mprr-dump")
{
    var dumpDoc = Pm4ResearchReader.ReadFile(args[1]);
    int count = int.Parse(args[2]);
    Console.WriteLine($"MPRR records: {dumpDoc.KnownChunks.Mprr.Count}; MPRL: {dumpDoc.KnownChunks.Mprl.Count}; MSCN: {dumpDoc.KnownChunks.Mscn.Count}");
    foreach (var chunk in dumpDoc.Chunks)
        Console.WriteLine($"  CHUNK {chunk.Signature} size={chunk.Size} stride={(chunk.Size == 0 ? 0 : (double)chunk.Size / Math.Max(1, chunk.Size / 4))}");
    var mprrChunk = dumpDoc.Chunks.FirstOrDefault(c => c.Signature == "MPRR");
    if (mprrChunk is not null)
        Console.WriteLine($"  MPRR payload={mprrChunk.Size} records={dumpDoc.KnownChunks.Mprr.Count} bytes/record={(double)mprrChunk.Size / Math.Max(1, dumpDoc.KnownChunks.Mprr.Count):F3}");
    int run = 0;
    for (int i = 0; i < Math.Min(count, dumpDoc.KnownChunks.Mprr.Count); i++)
    {
        var r = dumpDoc.KnownChunks.Mprr[i];
        if (r.IsSentinel)
        {
            Console.WriteLine($"  [{i}] SENTINEL run={run}");
            run = 0;
        }
        else
        {
            run++;
            Console.WriteLine($"  [{i}] v1={r.Value1} (0x{r.Value1:X4}) v2={r.Value2} (0x{r.Value2:X4})");
        }
    }
    return;
}
if (edgesOnly || edgeIdentity || mshd || placements || m2Partition || pathNodes) args = args[1..];
if (args.Length != 2) throw new ArgumentException("Usage: [--edges-only|--edge-identity|--mshd|--placements|--m2-partition|--path-nodes|--mprr-dump <file> <count>] <complete-map-pm4-directory> <new-report.json>");
if (File.Exists(args[1])) throw new IOException("Report exists; refusing overwrite.");
var timer = Stopwatch.StartNew();
var files = new List<Pm4MapAuditInput>();
var placementInputs = new List<Pm4PlacementAuditInput>();
var hashes = new List<object>();
var sizes = new Dictionary<string, long>(StringComparer.Ordinal);
var identity = new StringBuilder();
int adtRead = 0, adtMissing = 0, adtFailed = 0;
foreach (string path in Directory.GetFiles(args[0], "*.pm4").Order(StringComparer.Ordinal))
{
    if (timer.Elapsed.TotalSeconds > 30) throw new TimeoutException("Read budget exceeded; no complete-map output emitted.");
    byte[] bytes = File.ReadAllBytes(path);
    string name = Path.GetFileName(path), hash = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    identity.Append(name).Append(':').Append(hash).Append('\n');
    var doc = Pm4ResearchReader.Read(bytes, path);
    if (doc.Diagnostics.Count != 0) throw new InvalidDataException("Reader diagnostics require review before spatial inference.");
    files.Add(new Pm4MapAuditInput(name, doc));
    sizes[name] = bytes.Length;
    hashes.Add(new { File = name, Bytes = bytes.Length, Sha256 = hash });

    if (placements || m2Partition)
    {
        AdtPlacementCatalog? catalog = null;
        string? adtPath = Pm4CoordinateService.TryGetObj0PathForPm4(path);
        if (adtPath is null)
        {
            adtMissing++;
        }
        else
        {
            try
            {
                catalog = AdtPlacementReader.Read(adtPath);
                adtRead++;
            }
            catch (InvalidDataException)
            {
                adtFailed++;
            }
        }
        placementInputs.Add(new Pm4PlacementAuditInput(name, doc, catalog));
    }
}
if (files.Count == 0) throw new InvalidDataException("No map inputs.");
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[1]))!);
string mapHash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(identity.ToString()))).ToLowerInvariant();
if (edgeIdentity)
{
    var edgeReport = Pm4EdgeIdentityProbe.Analyze(files);
    File.WriteAllText(args[1], JsonSerializer.Serialize(new
    {
        Warning = "Reciprocal-edge/value-reuse tests of raw +0x04; reverse records resolved independently of the tested value. Not asset identity.",
        MapSha256 = mapHash, Result = edgeReport, Inputs = hashes, ElapsedSeconds = timer.Elapsed.TotalSeconds,
    }, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"Edge-identity probe: {files.Count} files, {edgeReport.Links} links, {timer.Elapsed.TotalSeconds:F3}s. No asset identity asserted.");
    return;
}
if (mshd)
{
    var mshdReport = Pm4MshdProbe.Analyze(files, sizes);
    File.WriteAllText(args[1], JsonSerializer.Serialize(new
    {
        Warning = "MSHD word correlations with file-level counts and tile coordinates. Correlation is not semantics.",
        MapSha256 = mapHash, Result = mshdReport, Inputs = hashes, ElapsedSeconds = timer.Elapsed.TotalSeconds,
    }, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"MSHD probe: {files.Count} files, {mshdReport.Words.Count} words, {timer.Elapsed.TotalSeconds:F3}s. No field meaning asserted.");
    return;
}
if (pathNodes)
{
    var pathReport = Pm4PathNodeProbe.Analyze(files);
    File.WriteAllText(args[1], JsonSerializer.Serialize(new
    {
        Warning = "MPRL/MPRR range and containment tests. In-range values are candidate indices, not proven edges; sentinel retained.",
        MapSha256 = mapHash, Result = pathReport, Inputs = hashes, ElapsedSeconds = timer.Elapsed.TotalSeconds,
    }, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"Path-node probe: {pathReport.Files} files, MPRL {pathReport.MprlCount}, MPRR {pathReport.MprrCount}, " +
        $"inside mesh {pathReport.MprlInsideMesh}/{pathReport.MprlWithFinitePosition}, {timer.Elapsed.TotalSeconds:F3}s.");
    return;
}
if (m2Partition)
{
    var partitionReport = Pm4M2PartitionProbe.Analyze(placementInputs);
    File.WriteAllText(args[1], JsonSerializer.Serialize(new
    {
        Warning = "Topological components of Ck24Type==0 surfaces vs authored M2 origins. Component coverage is not object identity.",
        MapSha256 = mapHash, AdtRead = adtRead, AdtMissing = adtMissing, AdtFailed = adtFailed,
        Result = partitionReport, Inputs = hashes, ElapsedSeconds = timer.Elapsed.TotalSeconds,
    }, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"M2 partition probe: {partitionReport.FilesWithM2} files with M2, {partitionReport.M2Surfaces} surfaces, " +
        $"{partitionReport.Components} components, 0/1/2+ placements = {partitionReport.ComponentsWithZeroPlacements}/" +
        $"{partitionReport.ComponentsWithOnePlacement}/{partitionReport.ComponentsWithMultiplePlacements}, {timer.Elapsed.TotalSeconds:F3}s.");
    return;
}
if (placements)
{
    var placementReport = Pm4PlacementGroundTruthProbe.Analyze(placementInputs);
    File.WriteAllText(args[1], JsonSerializer.Serialize(new
    {
        Warning = "Authored _obj0.adt placements vs PM4 surface features. Coverage is origin-in-polygon, not full footprint; correlation is not identity.",
        MapSha256 = mapHash, AdtRead = adtRead, AdtMissing = adtMissing, AdtFailed = adtFailed,
        Result = placementReport, Inputs = hashes, ElapsedSeconds = timer.Elapsed.TotalSeconds,
    }, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"Placement ground-truth probe: {placementReport.Files} files, {placementReport.FilesWithPlacements} with placements, " +
        $"M2 {placementReport.Coverage.M2Covered}/{placementReport.Coverage.M2Placements} covered, " +
        $"WMO {placementReport.Coverage.WmoCovered}/{placementReport.Coverage.WmoPlacements} covered, {timer.Elapsed.TotalSeconds:F3}s. No identity asserted.");
    return;
}
var report = Pm4PointReferenceProbe.Analyze(files, includePointCandidates: !edgesOnly);
File.WriteAllText(args[1], JsonSerializer.Serialize(new
{
    Warning = "Spatial enrichment tests of raw +0x04 and independent subtype ordinals; not editor-node or asset identity.",
    MapSha256 = mapHash,
    Result = report, Inputs = hashes, EdgesOnly = edgesOnly, ElapsedSeconds = timer.Elapsed.TotalSeconds,
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Point/boundary probe: {files.Count} files, {report.WindowLinks} links, {timer.Elapsed.TotalSeconds:F3}s. No point-layer identity asserted.");
