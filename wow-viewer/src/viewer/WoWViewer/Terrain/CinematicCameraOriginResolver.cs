using System.Numerics;
using DBCD;
using DBCD.Providers;
using WoWViewer.Logging;
using WoWViewer.Rendering;

namespace WoWViewer.Terrain;

/// <summary>
/// The client stores classic FlyBy camera tracks in the camera asset's local
/// coordinate system. CinematicCamera.dbc supplies the map-space origin and
/// facing for that asset.
/// </summary>
public sealed record CinematicCameraOrigin(
    int Id,
    string Model,
    Vector3 Origin,
    float OriginFacingRadians,
    int TileX,
    int TileY);

public sealed class CinematicCameraOriginResolver
{
    public bool TryResolve(
        IDBCProvider dbcProvider,
        string dbdDir,
        string build,
        string assetPath,
        out CinematicCameraOrigin? result)
    {
        ArgumentNullException.ThrowIfNull(dbcProvider);
        ArgumentException.ThrowIfNullOrWhiteSpace(dbdDir);
        ArgumentException.ThrowIfNullOrWhiteSpace(build);
        ArgumentException.ThrowIfNullOrWhiteSpace(assetPath);

        result = null;
        string normalizedAssetPath = NormalizePath(assetPath);
        try
        {
            var dbcd = new DBCD.DBCD(dbcProvider, new FilesystemDBDProvider(dbdDir));
            IDBCDStorage storage;
            try
            {
                storage = dbcd.Load("CinematicCamera", build, Locale.EnUS);
            }
            catch
            {
                storage = dbcd.Load("CinematicCamera", build, Locale.None);
            }

            List<(int Id, string Model, Vector3 Origin, float Facing)> matches = new();
            foreach (var key in storage.Keys)
            {
                dynamic row = storage[key];
                string model = NormalizePath(TryGetString(row, "Model") ?? string.Empty);
                if (string.IsNullOrWhiteSpace(model) || !PathsMatch(normalizedAssetPath, model))
                    continue;

                if (!TryGetVector3(row, "Origin", out Vector3 origin))
                    continue;
                if (!TryGetFloat(row, "OriginFacing", out float facing) || !float.IsFinite(facing))
                    continue;

                int rowId = TryGetInt(row, "ID") ?? key;
                matches.Add((rowId, model, origin, facing));
            }

            if (matches.Count != 1)
            {
                if (matches.Count > 1)
                    ViewerLog.Error(ViewerLog.Category.Dbc,
                        $"[CinematicCamera] Ambiguous model '{assetPath}' matched {matches.Count} rows in build {build}.");
                return false;
            }

            (int id, string modelPath, Vector3 originPosition, float originFacing) = matches[0];
            int tileX = Math.Clamp((int)MathF.Floor((WoWConstants.MapOrigin - originPosition.X) / WoWConstants.ChunkSize), 0, 63);
            int tileY = Math.Clamp((int)MathF.Floor((WoWConstants.MapOrigin - originPosition.Y) / WoWConstants.ChunkSize), 0, 63);
            result = new CinematicCameraOrigin(id, modelPath, originPosition, originFacing, tileX, tileY);
            ViewerLog.Trace(
                $"[CinematicCamera] {modelPath} id={id} origin=({originPosition.X:F3},{originPosition.Y:F3},{originPosition.Z:F3}) " +
                $"facing={originFacing:F4} tile=({tileX},{tileY}) build={build}");
            return true;
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[CinematicCamera] Could not load CinematicCamera.dbc for build {build}: {ex.Message}");
            return false;
        }
    }

    private static bool PathsMatch(string assetPath, string modelPath)
    {
        if (string.Equals(assetPath, modelPath, StringComparison.OrdinalIgnoreCase))
            return true;

        bool suffixMatch = assetPath.EndsWith('\\' + modelPath, StringComparison.OrdinalIgnoreCase)
            || modelPath.EndsWith('\\' + assetPath, StringComparison.OrdinalIgnoreCase);
        if (suffixMatch)
            return true;

        // Some clients retain the legacy .mdx model name in CinematicCamera
        // while the loaded asset is the equivalent .m2 camera file.
        string assetStem = StripCameraExtension(assetPath);
        string modelStem = StripCameraExtension(modelPath);
        return string.Equals(assetStem, modelStem, StringComparison.OrdinalIgnoreCase)
            || assetStem.EndsWith('\\' + modelStem, StringComparison.OrdinalIgnoreCase)
            || modelStem.EndsWith('\\' + assetStem, StringComparison.OrdinalIgnoreCase);
    }

    private static string StripCameraExtension(string path)
    {
        int separator = path.LastIndexOf('\\');
        int dot = path.LastIndexOf('.');
        if (dot <= separator)
            return path;

        string extension = path[(dot + 1)..];
        return extension.Equals("mdx", StringComparison.OrdinalIgnoreCase)
            || extension.Equals("m2", StringComparison.OrdinalIgnoreCase)
            ? path[..dot]
            : path;
    }

    private static string NormalizePath(string value)
        => value.Replace('/', '\\').Trim().TrimStart('\\');

    private static string? TryGetString(dynamic row, string fieldName)
    {
        try
        {
            object? value = row[fieldName];
            return value?.ToString();
        }
        catch
        {
            return null;
        }
    }

    private static int? TryGetInt(dynamic row, string fieldName)
    {
        try { return Convert.ToInt32(row[fieldName]); }
        catch { return null; }
    }

    private static bool TryGetFloat(dynamic row, string fieldName, out float value)
    {
        try
        {
            value = Convert.ToSingle(row[fieldName]);
            return true;
        }
        catch
        {
            value = 0f;
            return false;
        }
    }

    private static bool TryGetVector3(dynamic row, string fieldName, out Vector3 value)
    {
        try
        {
            object raw = row[fieldName];
            if (raw is Array array && array.Length >= 3)
            {
                value = new Vector3(
                    Convert.ToSingle(array.GetValue(0)),
                    Convert.ToSingle(array.GetValue(1)),
                    Convert.ToSingle(array.GetValue(2)));
                return float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);
            }

            float[] components =
            [
                Convert.ToSingle(row[$"{fieldName}[0]"]),
                Convert.ToSingle(row[$"{fieldName}[1]"]),
                Convert.ToSingle(row[$"{fieldName}[2]"]),
            ];
            value = new Vector3(components[0], components[1], components[2]);
            return float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);
        }
        catch
        {
            value = default;
            return false;
        }
    }
}
