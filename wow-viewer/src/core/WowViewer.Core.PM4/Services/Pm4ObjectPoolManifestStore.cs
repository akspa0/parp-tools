using System.Text.Json;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Services;

public static class Pm4ObjectPoolManifestStore
{
    private static readonly JsonSerializerOptions Options = new() { WriteIndented = true, PropertyNameCaseInsensitive = true };

    public static Pm4ObjectPoolManifest Read(string path)
    {
        var manifest = JsonSerializer.Deserialize<Pm4ObjectPoolManifest>(File.ReadAllText(path), Options)
            ?? throw new InvalidDataException("Empty object-pool manifest.");
        Pm4ObjectPoolFilter.Validate(manifest);
        return manifest;
    }

    public static void Write(string path, Pm4ObjectPoolManifest manifest)
    {
        Pm4ObjectPoolFilter.Validate(manifest);
        string destination = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        string pending = destination + ".pending";
        using (var stream = new FileStream(pending, FileMode.CreateNew, FileAccess.Write))
            JsonSerializer.Serialize(stream, manifest, Options);
        File.Move(pending, destination, overwrite: true);
    }
}
