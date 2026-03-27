using WowViewer.Core.IO.Files;

namespace WowViewer.Core.IO.Dbc;

public sealed class GroundEffectLookup
{
    private readonly Dictionary<uint, List<uint>> _textureToDoodads = [];
    private readonly Dictionary<uint, string> _doodadModels = [];
    private bool _loaded;

    public bool IsLoaded => _loaded;

    public void Load(IEnumerable<string> searchPaths, IArchiveReader? archiveReader = null)
    {
        ArgumentNullException.ThrowIfNull(searchPaths);

        if (_loaded)
            return;

        try
        {
            byte[]? doodadData = TryReadFromDisk(searchPaths, "GroundEffectDoodad")
                ?? TryReadFromArchive(archiveReader, "GroundEffectDoodad");
            byte[]? textureData = TryReadFromDisk(searchPaths, "GroundEffectTexture")
                ?? TryReadFromArchive(archiveReader, "GroundEffectTexture");

            if (doodadData is null || textureData is null)
            {
                Console.WriteLine("Could not find GroundEffect DBC files on disk or in archives.");
                return;
            }

            ProcessDoodadRows(DbcReader.Load(doodadData));
            ProcessTextureRows(DbcReader.Load(textureData));
            _loaded = true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to load GroundEffects: {ex.Message}");
        }
    }

    public string[]? GetDoodadsEffect(uint effectId)
    {
        if (!_loaded)
            return null;

        if (!_textureToDoodads.TryGetValue(effectId, out List<uint>? doodadIds))
            return null;

        return doodadIds.Where(_doodadModels.ContainsKey).Select(id => _doodadModels[id]).ToArray();
    }

    private static byte[]? TryReadFromArchive(IArchiveReader? archiveReader, string tableName)
    {
        return archiveReader is null ? null : DbClientFileReader.TryReadTable(archiveReader, tableName);
    }

    private static byte[]? TryReadFromDisk(IEnumerable<string> searchPaths, string tableName)
    {
        foreach (string basePath in searchPaths.Where(static path => !string.IsNullOrWhiteSpace(path)))
        {
            foreach (string candidate in EnumerateDiskCandidates(basePath, tableName))
            {
                if (File.Exists(candidate))
                    return File.ReadAllBytes(candidate);
            }
        }

        return null;
    }

    private static IEnumerable<string> EnumerateDiskCandidates(string basePath, string tableName)
    {
        yield return Path.Combine(basePath, "DBFilesClient", $"{tableName}.dbc");
        yield return Path.Combine(basePath, "DBFilesClient", $"{tableName}.db2");
        yield return Path.Combine(basePath, "DBC", $"{tableName}.dbc");
        yield return Path.Combine(basePath, "DBC", $"{tableName}.db2");
        yield return Path.Combine(basePath, $"{tableName}.dbc");
        yield return Path.Combine(basePath, $"{tableName}.db2");
    }

    private void ProcessDoodadRows(DbcReader dbc)
    {
        for (int rowIndex = 0; rowIndex < dbc.Rows.Count; rowIndex++)
        {
            uint id = dbc.GetUInt(rowIndex, 0);

            string model = dbc.GetString(rowIndex, 2);
            if (string.IsNullOrEmpty(model) || !model.Contains('.', StringComparison.Ordinal))
                model = dbc.GetString(rowIndex, 1);

            if (!string.IsNullOrEmpty(model) && model.Length > 4)
                _doodadModels[id] = model;
        }
    }

    private void ProcessTextureRows(DbcReader dbc)
    {
        for (int rowIndex = 0; rowIndex < dbc.Rows.Count; rowIndex++)
        {
            uint id = dbc.GetUInt(rowIndex, 0);
            List<uint> doodads = [];

            for (int fieldIndex = 1; fieldIndex <= 4 && fieldIndex < dbc.Header.FieldCount; fieldIndex++)
            {
                uint doodadId = dbc.GetUInt(rowIndex, fieldIndex);
                if (doodadId > 0 && _doodadModels.ContainsKey(doodadId))
                    doodads.Add(doodadId);
            }

            if (doodads.Count > 0)
                _textureToDoodads[id] = doodads;
        }
    }
}