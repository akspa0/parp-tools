using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

bool edgesOnly = args.Length == 3 && args[0] == "--edges-only";
if (edgesOnly) args = args[1..];
if (args.Length != 2) throw new ArgumentException("Usage: [--edges-only] <complete-map-pm4-directory> <new-report.json>");
if (File.Exists(args[1])) throw new IOException("Report exists; refusing overwrite.");
var timer = Stopwatch.StartNew();
var files = new List<Pm4MapAuditInput>();
var hashes = new List<object>();
var identity = new StringBuilder();
foreach (string path in Directory.GetFiles(args[0], "*.pm4").Order(StringComparer.Ordinal))
{
    if (timer.Elapsed.TotalSeconds > 30) throw new TimeoutException("Read budget exceeded; no complete-map output emitted.");
    byte[] bytes = File.ReadAllBytes(path);
    string name = Path.GetFileName(path), hash = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    identity.Append(name).Append(':').Append(hash).Append('\n');
    var doc = Pm4ResearchReader.Read(bytes, path);
    if (doc.Diagnostics.Count != 0) throw new InvalidDataException("Reader diagnostics require review before spatial inference.");
    files.Add(new Pm4MapAuditInput(name, doc));
    hashes.Add(new { File = name, Bytes = bytes.Length, Sha256 = hash });
}
if (files.Count == 0) throw new InvalidDataException("No map inputs.");
var report = Pm4PointReferenceProbe.Analyze(files, includePointCandidates: !edgesOnly);
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[1]))!);
File.WriteAllText(args[1], JsonSerializer.Serialize(new
{
    Warning = "Spatial enrichment tests of raw +0x04 and independent subtype ordinals; not editor-node or asset identity.",
    MapSha256 = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(identity.ToString()))).ToLowerInvariant(),
    Result = report, Inputs = hashes, EdgesOnly = edgesOnly, ElapsedSeconds = timer.Elapsed.TotalSeconds,
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Point/boundary probe: {files.Count} files, {report.WindowLinks} links, {timer.Elapsed.TotalSeconds:F3}s. No point-layer identity asserted.");
