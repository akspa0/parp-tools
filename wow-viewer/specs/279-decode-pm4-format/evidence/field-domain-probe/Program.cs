using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

if (args.Length != 2) throw new ArgumentException("Usage: <complete-map-pm4-directory> <new-report.json>");
if (File.Exists(args[1])) throw new IOException("Report exists; refusing overwrite.");
var timer = Stopwatch.StartNew();
var files = new List<Pm4MapAuditInput>();
var inventory = new List<object>();
var mapHashInput = new StringBuilder();
foreach (string path in Directory.GetFiles(args[0], "*.pm4").Order(StringComparer.Ordinal))
{
    if (timer.Elapsed.TotalSeconds > 30) throw new TimeoutException("Read budget reached; no complete-map report emitted.");
    byte[] bytes = File.ReadAllBytes(path);
    string hash = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    string name = Path.GetFileName(path);
    mapHashInput.Append(name).Append(':').Append(hash).Append('\n');
    var doc = Pm4ResearchReader.Read(bytes, path);
    if (doc.Diagnostics.Count != 0) throw new InvalidDataException("Reader diagnostics need review before field interpretation.");
    files.Add(new Pm4MapAuditInput(name, doc));
    inventory.Add(new
    {
        File = name, Bytes = bytes.Length, Sha256 = hash, doc.Version,
        Header = doc.KnownChunks.Mshd,
        Chunks = doc.Chunks.Select(c => new { c.Signature, c.Size }),
    });
}
if (files.Count == 0) throw new InvalidDataException("No map inputs.");
var result = Pm4FieldDomainProbe.Analyze(files);
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[1]))!);
File.WriteAllText(args[1], JsonSerializer.Serialize(new
{
    Warning = "Raw MSLK +0x04 hypothesis tests. In-range counts do not identify a semantic domain or object ownership.",
    MapSha256 = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(mapHashInput.ToString()))).ToLowerInvariant(),
    Result = result, Inputs = inventory, ElapsedSeconds = timer.Elapsed.TotalSeconds,
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Raw +0x04 probe: {files.Count} files, {result.Links} links, {timer.Elapsed.TotalSeconds:F3}s. No field meaning asserted.");
