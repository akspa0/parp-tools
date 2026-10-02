using WowViewer.Core.IO.Files;

namespace WowViewer.Core.IO.Dbc;

public sealed class GroundEffectLookup
{
    private readonly Dictionary<uint, List<uint>> _textureToDoodads = [];
    private readonly Dictionary<uint, string> _doodadModels = [];
    private readonly Dictionary<uint, GroundEffectDoodadRecord> _doodadRecords = [];
    private readonly Dictionary<uint, GroundEffectTextureRecord> _textureRecords = [];
    private bool _loaded;

    public bool IsLoaded => _loaded;

    public IReadOnlyDictionary<uint, GroundEffectDoodadRecord> DoodadRecords => _doodadRecords;
    public IReadOnlyDictionary<uint, GroundEffectTextureRecord> TextureRecords => _textureRecords;

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

    public void Load(Func<string, byte[]?> fileReader)
    {
        ArgumentNullException.ThrowIfNull(fileReader);

        if (_loaded)
            return;

        try
        {
            byte[]? doodadData = fileReader("DBFilesClient\\GroundEffectDoodad.dbc")
                ?? fileReader("DBFilesClient/GroundEffectDoodad.dbc")
                ?? fileReader("DBFilesClient\\GroundEffectDoodad.db2")
                ?? fileReader("DBFilesClient/GroundEffectDoodad.db2")
                ?? fileReader("DBC\\GroundEffectDoodad.dbc")
                ?? fileReader("DBC/GroundEffectDoodad.dbc");

            byte[]? textureData = fileReader("DBFilesClient\\GroundEffectTexture.dbc")
                ?? fileReader("DBFilesClient/GroundEffectTexture.dbc")
                ?? fileReader("DBFilesClient\\GroundEffectTexture.db2")
                ?? fileReader("DBFilesClient/GroundEffectTexture.db2")
                ?? fileReader("DBC\\GroundEffectTexture.dbc")
                ?? fileReader("DBC/GroundEffectTexture.dbc");

            if (doodadData != null && textureData != null)
            {
                ProcessDoodadRows(DbcReader.Load(doodadData));
                ProcessTextureRows(DbcReader.Load(textureData));
                _loaded = true;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to load GroundEffects from file reader: {ex.Message}");
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

    public GroundEffectDoodadRecord? GetDoodadRecord(uint doodadId)
    {
        return _doodadRecords.TryGetValue(doodadId, out GroundEffectDoodadRecord? record) ? record : null;
    }

    public GroundEffectTextureRecord? GetTextureRecord(uint effectId)
    {
        return _textureRecords.TryGetValue(effectId, out GroundEffectTextureRecord? record) ? record : null;
    }

    public IReadOnlyList<GroundEffectDoodadRecord> GetDoodadRecordsForEffect(uint effectId)
    {
        if (!_loaded || !_textureToDoodads.TryGetValue(effectId, out List<uint>? doodadIds))
            return [];

        List<GroundEffectDoodadRecord> result = [];
        foreach (uint id in doodadIds)
        {
            if (_doodadRecords.TryGetValue(id, out GroundEffectDoodadRecord? record))
                result.Add(record);
        }

        return result;
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

    public void ProcessDoodadRows(DbcReader dbc)
    {
        for (int rowIndex = 0; rowIndex < dbc.Rows.Count; rowIndex++)
        {
            uint id = dbc.GetUInt(rowIndex, 0);
            string model = string.Empty;
            uint fileDataId = 0;
            GroundEffectDoodadFlags flags = GroundEffectDoodadFlags.None;
            float animScale = 1.0f;
            float pushScale = 1.0f;

            if (dbc.Header.FieldCount <= 3)
            {
                // 0.5.3 - 1.12 style: Field 0 is ID, Field 1 is tag, Field 2 is doodadpath
                string s2 = dbc.Header.FieldCount >= 3 ? dbc.GetString(rowIndex, 2) : string.Empty;
                string s1 = dbc.GetString(rowIndex, 1);
                if (!string.IsNullOrEmpty(s2) && s2.Contains('.', StringComparison.Ordinal))
                    model = s2;
                else if (!string.IsNullOrEmpty(s1) && s1.Contains('.', StringComparison.Ordinal))
                    model = s1;
            }
            else
            {
                // 3.x / Modern style (>= 4 fields):
                // Field 0: ID
                // Field 1: doodadpath (string) or ModelFileID (uint)
                // Field 2: flags
                // Field 3: animscale
                // Field 4: pushscale
                string s1 = dbc.GetString(rowIndex, 1);
                if (!string.IsNullOrEmpty(s1) && (s1.EndsWith(".m2", StringComparison.OrdinalIgnoreCase) || s1.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase) || s1.Contains('.', StringComparison.Ordinal)))
                {
                    model = s1;
                }
                else
                {
                    fileDataId = dbc.GetUInt(rowIndex, 1);
                    if (fileDataId > 0)
                        model = $"FileDataID:{fileDataId}";
                }

                if (dbc.Header.FieldCount >= 3)
                    flags = (GroundEffectDoodadFlags)dbc.GetUInt(rowIndex, 2);
                if (dbc.Header.FieldCount >= 4)
                    animScale = dbc.GetFloat(rowIndex, 3);
                if (dbc.Header.FieldCount >= 5)
                    pushScale = dbc.GetFloat(rowIndex, 4);
            }


            if (!string.IsNullOrEmpty(model) && model.Length > 4)
            {
                _doodadModels[id] = model;
                _doodadRecords[id] = new GroundEffectDoodadRecord(id, model, fileDataId, flags, animScale, pushScale);
            }
        }
    }

    public void ProcessTextureRows(DbcReader dbc)
    {
        for (int rowIndex = 0; rowIndex < dbc.Rows.Count; rowIndex++)
        {
            uint id = dbc.GetUInt(rowIndex, 0);
            List<uint> doodads = [];

            for (int fieldIndex = 1; fieldIndex <= 4 && fieldIndex < dbc.Header.FieldCount; fieldIndex++)
            {
                uint doodadId = dbc.GetUInt(rowIndex, fieldIndex);
                if (doodadId > 0 && (_doodadModels.ContainsKey(doodadId) || _doodadRecords.ContainsKey(doodadId)))
                    doodads.Add(doodadId);
            }

            uint density = 0;
            if (dbc.Header.FieldCount >= 6)
                density = dbc.GetUInt(rowIndex, 5);

            uint soundId = 0;
            if (dbc.Header.FieldCount >= 7)
                soundId = dbc.GetUInt(rowIndex, 6);

            if (doodads.Count > 0)
            {
                _textureToDoodads[id] = doodads;
                _textureRecords[id] = new GroundEffectTextureRecord(id, doodads, density, soundId);
            }
        }
    }
}

[Flags]
public enum GroundEffectDoodadFlags : uint
{
    None = 0,
    AlignToNormal = 0x1,
    IgnoreMCCV = 0x2,
}

public sealed record GroundEffectDoodadRecord(
    uint Id,
    string ModelPath,
    uint FileDataId,
    GroundEffectDoodadFlags Flags,
    float AnimScale = 1.0f,
    float PushScale = 1.0f)
{
    public bool AlignToNormal => (Flags & GroundEffectDoodadFlags.AlignToNormal) != 0;
    public bool IgnoreMCCV => (Flags & GroundEffectDoodadFlags.IgnoreMCCV) != 0;
}

public sealed record GroundEffectTextureRecord(
    uint Id,
    IReadOnlyList<uint> DoodadIds,
    uint Density = 0,
    uint SoundId = 0);