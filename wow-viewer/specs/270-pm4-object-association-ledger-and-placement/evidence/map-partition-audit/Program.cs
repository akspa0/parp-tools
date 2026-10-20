using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

// Operator-run broad read-only audit. No sample/tile-limit switch: all input PM4s form one map.
if (args.Length != 3)
    throw new ArgumentException("Usage: <pm4-map-directory> <authored-Museum-map-directory> <new-output-directory>");
string mapRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(args[0]));
string placementRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(args[1]));
string output = Path.TrimEndingDirectorySeparator(Path.GetFullPath(args[2]));
if (!Directory.Exists(mapRoot) || !Directory.Exists(placementRoot))
    throw new DirectoryNotFoundException("Both explicit map input directories must exist.");
if (Directory.Exists(output) || File.Exists(output))
    throw new IOException("Output must be a new directory; refusing to mix with or overwrite a previous run.");
foreach (string input in new[] { mapRoot, placementRoot })
    if (output.Equals(input, StringComparison.OrdinalIgnoreCase)
        || output.StartsWith(input + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
        throw new IOException("Audit output cannot be written inside an input directory.");

string[] pm4Files = Directory.GetFiles(mapRoot, "*.pm4").Order(StringComparer.Ordinal).ToArray();
string[] adtFiles = Directory.GetFiles(placementRoot, "*.adt")
    .Where(p => !Path.GetFileNameWithoutExtension(p).EndsWith("_tex0", StringComparison.OrdinalIgnoreCase))
    .Order(StringComparer.Ordinal).ToArray();
if (pm4Files.Length == 0 || adtFiles.Length == 0)
    throw new InvalidDataException("Expected map PM4 files and authored placement ADTs; refusing an empty audit.");
Directory.CreateDirectory(output);
var options = new JsonSerializerOptions { IncludeFields = true, NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals };
var files = new List<Pm4MapAuditInput>();
var placements = new List<Pm4MapPlacementInput>();
var manifest = new List<object>();
var failures = new List<object>();
string[] wlFiles = Directory.GetFiles(mapRoot).Where(p => Path.GetExtension(p).StartsWith(".wl", StringComparison.OrdinalIgnoreCase)).Order(StringComparer.Ordinal).ToArray();
foreach (string path in wlFiles)
    try
    {
        byte[] bytes = File.ReadAllBytes(path);
        manifest.Add(new { Kind = "WLContext", FileId = Path.GetFileName(path), Bytes = bytes.Length, Sha256 = Hash(bytes) });
    }
    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
    {
        failures.Add(new { Kind = "WLContext", FileId = Path.GetFileName(path), Error = ex.Message });
    }
File.WriteAllText(Path.Combine(output, "wl-source-hints.json"), JsonSerializer.Serialize(
    Pm4DesignKitHints.FromSourceNames(wlFiles.Select(p => Path.GetFileName(p))), options));
using (var decoded = new StreamWriter(Path.Combine(output, "decoded-documents.jsonl")))
using (var authored = new StreamWriter(Path.Combine(output, "authored-catalogs.jsonl")))
{
    foreach (string path in pm4Files)
    {
        string id = Path.GetFileName(path);
        try
        {
            byte[] bytes = File.ReadAllBytes(path);
            manifest.Add(new { Kind = "PM4", FileId = id, Bytes = bytes.Length, Sha256 = Hash(bytes) });
            var document = Pm4ResearchReader.Read(bytes, path);
            // Full decoded entry sets AND original chunk payloads, not a reduced fingerprint.
            decoded.WriteLine(JsonSerializer.Serialize(new { FileId = id, Document = document }, options));
            if (document.Diagnostics.Count > 0)
                throw new InvalidDataException("PM4 reader diagnostics require review before a complete-map claim: " + string.Join("; ", document.Diagnostics));
            files.Add(new Pm4MapAuditInput(id, document));
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or ArgumentException or OverflowException or UnauthorizedAccessException)
        {
            failures.Add(new { Kind = "PM4", FileId = id, Error = ex.Message });
        }
    }
    foreach (string path in adtFiles)
    {
        string id = Path.GetFileName(path);
        try
        {
            byte[] bytes = File.ReadAllBytes(path);
            manifest.Add(new { Kind = "AuthoredADT", FileId = id, Bytes = bytes.Length, Sha256 = Hash(bytes) });
            var catalog = AdtPlacementAuditReader.Read(bytes, path);
            placements.Add(new Pm4MapPlacementInput(id, catalog));
            authored.WriteLine(JsonSerializer.Serialize(new { FileId = id, Catalog = catalog }, options));
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or NotSupportedException or ArgumentException or OverflowException or UnauthorizedAccessException)
        {
            failures.Add(new { Kind = "AuthoredADT", FileId = id, Error = ex.Message });
        }
    }
}
File.WriteAllText(Path.Combine(output, "manifest.json"), JsonSerializer.Serialize(new
{
    Status = failures.Count == 0 ? "InputsRead" : "Incomplete",
    MapRoot = mapRoot, AuthoredRoot = placementRoot,
    Build = "Not inferred; operator must identify the dataset/client revision.",
    Warning = "Authored rows are preserved, including repeated references across tiles; no identity/partition accuracy claimed.",
    Pm4Files = pm4Files.Length, AuthoredFiles = adtFiles.Length, Inputs = manifest, Failures = failures,
}, options));
if (failures.Count != 0)
    throw new InvalidDataException($"{failures.Count} inputs failed. Evidence retained; no complete-map audit emitted.");

var report = Pm4MapPartitionAudit.Analyze(files, placements);
using (var stream = File.Create(Path.Combine(output, "map-partition-audit.json")))
    JsonSerializer.Serialize(stream, report, options);
File.WriteAllText(Path.Combine(output, "COMPLETE.json"), JsonSerializer.Serialize(new
{
    report.FileCount, report.SurfaceCount, report.LinkCount, report.ZeroCk24SurfaceCount,
    report.RawWordZeroSurfaceCount, report.UnownedLinks, report.InvalidSurfaceWindows,
    Hypotheses = report.Hypotheses.Select(h => new
    {
        h.Name, Components = h.Components.Count,
        ZeroContainingComponents = h.Components.Count(c => c.ZeroCk24Surfaces > 0),
        MixedZeroAndKeyedComponents = h.Components.Count(c => c.ZeroCk24Surfaces > 0 && c.ZeroCk24Surfaces < c.SurfaceIds.Count),
        CrossFileComponents = h.Components.Count(c => c.Files.Count > 1),
        ScaleFamilyCandidates = h.CandidateScaleFamilies.Count,
    }),
    AuthoredPlacements = report.AuthoredPlacements.Count,
    AuthoredM2Families = report.AuthoredM2Families.Count,
    Warning = report.Warning,
}, new JsonSerializerOptions(options) { WriteIndented = true }));
Console.WriteLine($"Complete map inputs: {report.FileCount} PM4 files; {report.SurfaceCount} surfaces; {report.LinkCount} links.");
Console.WriteLine($"Retained {report.AuthoredPlacements.Count} authored placement references; {report.Hypotheses.Count} partition hypotheses. No asset matches asserted.");

static string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
