using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Dbc;

namespace WowViewer.Core.IO.Population;

/// <summary>
/// Resolves creature and gameobject display IDs into canonical model paths and scales
/// using CreatureDisplayInfo, CreatureModelData, and GameObjectDisplayInfo tables.
/// </summary>
public sealed class DbcModelResolver : IDbcModelResolver
{
    private readonly Dictionary<int, (string modelPath, float scale)> _creatureDisplays = new();
    private readonly Dictionary<int, (string modelPath, float scale)> _gameObjectDisplays = new();

    public int CreatureDisplayCount => _creatureDisplays.Count;
    public int GameObjectDisplayCount => _gameObjectDisplays.Count;

    public bool TryResolveCreature(int displayId, out string? modelPath, out float displayScale)
    {
        if (_creatureDisplays.TryGetValue(displayId, out var entry))
        {
            modelPath = entry.modelPath;
            displayScale = entry.scale;
            return true;
        }

        modelPath = null;
        displayScale = 1.0f;
        return false;
    }

    public bool TryResolveGameObject(int displayId, out string? modelPath, out float displayScale)
    {
        if (_gameObjectDisplays.TryGetValue(displayId, out var entry))
        {
            modelPath = entry.modelPath;
            displayScale = entry.scale;
            return true;
        }

        modelPath = null;
        displayScale = 1.0f;
        return false;
    }

    public void RegisterCreature(int displayId, string modelPath, float scale = 1.0f)
    {
        if (displayId > 0 && !string.IsNullOrWhiteSpace(modelPath))
            _creatureDisplays[displayId] = (NormalizeCreaturePath(modelPath), scale > 0 ? scale : 1.0f);
    }

    public void RegisterGameObject(int displayId, string modelPath, float scale = 1.0f)
    {
        if (displayId > 0 && !string.IsNullOrWhiteSpace(modelPath))
            _gameObjectDisplays[displayId] = (NormalizeGameObjectPath(modelPath), scale > 0 ? scale : 1.0f);
    }

    /// <summary>
    /// Loads DBC files from loose files in a directory (e.g. DBFilesClient/).
    /// </summary>
    public static DbcModelResolver LoadFromDirectory(string directory)
    {
        var resolver = new DbcModelResolver();
        if (!Directory.Exists(directory))
            return resolver;

        string cdi = Path.Combine(directory, "CreatureDisplayInfo.dbc");
        string cmd = Path.Combine(directory, "CreatureModelData.dbc");
        string gdi = Path.Combine(directory, "GameObjectDisplayInfo.dbc");

        byte[]? cdiBytes = File.Exists(cdi) ? File.ReadAllBytes(cdi) : null;
        byte[]? cmdBytes = File.Exists(cmd) ? File.ReadAllBytes(cmd) : null;
        byte[]? gdiBytes = File.Exists(gdi) ? File.ReadAllBytes(gdi) : null;

        resolver.LoadData(cdiBytes, cmdBytes, gdiBytes);
        return resolver;
    }

    /// <summary>
    /// Loads DBC files from an MPQ / CASC archive catalog.
    /// </summary>
    public static DbcModelResolver LoadFromArchive(IArchiveCatalog catalog)
    {
        var resolver = new DbcModelResolver();
        byte[]? cdiBytes = catalog.ReadFile("DBFilesClient\\CreatureDisplayInfo.dbc")
            ?? catalog.ReadFile("DBFilesClient/CreatureDisplayInfo.dbc");
        byte[]? cmdBytes = catalog.ReadFile("DBFilesClient\\CreatureModelData.dbc")
            ?? catalog.ReadFile("DBFilesClient/CreatureModelData.dbc");
        byte[]? gdiBytes = catalog.ReadFile("DBFilesClient\\GameObjectDisplayInfo.dbc")
            ?? catalog.ReadFile("DBFilesClient/GameObjectDisplayInfo.dbc");

        resolver.LoadData(cdiBytes, cmdBytes, gdiBytes);
        return resolver;
    }

    public void LoadData(byte[]? cdiBytes, byte[]? cmdBytes, byte[]? gdiBytes)
    {
        // 1. CreatureModelData: ID(0) -> ModelPath(2)
        var modelDataPaths = new Dictionary<int, string>();
        if (cmdBytes != null && cmdBytes.Length > 0)
        {
            var cmd = DbcReader.Load(cmdBytes);
            for (int r = 0; r < cmd.Rows.Count; r++)
            {
                int id = cmd.GetInt(r, 0);
                // In standard 3.3.5a, ModelPath is field 2
                uint strOffset = cmd.GetUInt(r, 2);
                string path = cmd.GetString(strOffset);
                if (string.IsNullOrWhiteSpace(path) && cmd.Header.RecordSize >= 8)
                {
                    // Fallback check field 1 if 2 was 0
                    uint altOffset = cmd.GetUInt(r, 1);
                    path = cmd.GetString(altOffset);
                }

                if (!string.IsNullOrWhiteSpace(path))
                    modelDataPaths[id] = path;
            }
        }

        // 2. CreatureDisplayInfo: ID(0) -> ModelID(1), Scale(4)
        if (cdiBytes != null && cdiBytes.Length > 0)
        {
            var cdi = DbcReader.Load(cdiBytes);
            for (int r = 0; r < cdi.Rows.Count; r++)
            {
                int id = cdi.GetInt(r, 0);
                int modelDataId = cdi.GetInt(r, 1);
                float scale = cdi.GetFloat(r, 4);
                if (scale <= 0) scale = 1.0f;

                if (modelDataPaths.TryGetValue(modelDataId, out var path))
                {
                    RegisterCreature(id, path, scale);
                }
            }
        }

        // 3. GameObjectDisplayInfo: ID(0) -> ModelPath(1)
        if (gdiBytes != null && gdiBytes.Length > 0)
        {
            var gdi = DbcReader.Load(gdiBytes);
            for (int r = 0; r < gdi.Rows.Count; r++)
            {
                int id = gdi.GetInt(r, 0);
                uint strOffset = gdi.GetUInt(r, 1);
                string path = gdi.GetString(strOffset);
                if (!string.IsNullOrWhiteSpace(path))
                {
                    RegisterGameObject(id, path, 1.0f);
                }
            }
        }
    }

    public static string NormalizeCreaturePath(string path)
    {
        string p = path.Trim().Replace('/', '\\');
        if (!p.EndsWith(".m2", StringComparison.OrdinalIgnoreCase) && !p.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            p += ".m2";
        }

        if (!p.StartsWith("Creature\\", StringComparison.OrdinalIgnoreCase) && !p.Contains('\\'))
        {
            string name = Path.GetFileNameWithoutExtension(p);
            p = $"Creature\\{name}\\{p}";
        }

        return p;
    }

    public static string NormalizeGameObjectPath(string path)
    {
        string p = path.Trim().Replace('/', '\\');
        if (!p.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase) && !p.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
        {
            p += ".mdx";
        }

        return p;
    }
}
