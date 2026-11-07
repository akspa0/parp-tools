using System.Text.Json;

namespace WowViewer.Core.Maps;

/// <summary>JSON persistence for one map's manual placements and explicit PM4 hull associations.</summary>
public static class ManualCurationProjectFile
{
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        WriteIndented = true,
        IncludeFields = true,
    };

    public static ManualCurationProject Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ManualCurationProject project = JsonSerializer.Deserialize<ManualCurationProject>(
            File.ReadAllText(path), SerializerOptions)
            ?? throw new InvalidDataException($"Manual curation project '{path}' is empty or invalid.");

        if (project.Version != 1)
            throw new InvalidDataException($"Manual curation project version {project.Version} is not supported.");
        if (string.IsNullOrWhiteSpace(project.MapName))
            throw new InvalidDataException("Manual curation project has no map name.");

        project.Placements ??= [];
        project.Associations ??= [];
        project.ShapeMatchDecisions ??= [];
        project.ShapeMatchAuditHistory ??= [];
        return project;
    }

    public static void Save(string path, ManualCurationProject project)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(project);
        if (project.Version != 1)
            throw new InvalidDataException($"Manual curation project version {project.Version} is not supported.");
        if (string.IsNullOrWhiteSpace(project.MapName))
            throw new InvalidDataException("Manual curation project has no map name.");

        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        if (string.IsNullOrWhiteSpace(directory))
            throw new InvalidOperationException("Manual curation project path has no parent directory.");

        Directory.CreateDirectory(directory);
        string temporaryPath = fullPath + ".tmp";
        try
        {
            File.WriteAllText(temporaryPath, JsonSerializer.Serialize(project, SerializerOptions));
            File.Move(temporaryPath, fullPath, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporaryPath))
                File.Delete(temporaryPath);
        }
    }
}
