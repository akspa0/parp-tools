using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text.Json;

namespace WowViewer.App;

internal enum WowViewerModelOutputVariant
{
    Predicted = 0,
    WdlBaseline = 1,
}

internal enum WowViewerModelOutputCameraMode
{
    Orbit = 0,
    Fly = 1,
}

internal sealed class WowViewerModelOutputState
{
    public string InputPath { get; set; } = string.Empty;

    public WowViewerModelOutputVariant Variant { get; set; } = WowViewerModelOutputVariant.Predicted;

    public WowViewerModelOutputCameraMode CameraMode { get; set; } = WowViewerModelOutputCameraMode.Orbit;

    public bool ShowObjects { get; set; } = true;

    public bool ShowM2Objects { get; set; } = true;

    public bool ShowWmoObjects { get; set; } = true;

    public float CameraTargetOffsetX { get; set; }

    public float CameraTargetOffsetY { get; set; }

    public float CameraTargetOffsetZ { get; set; }

    public float CameraAzimuthDegrees { get; set; } = 45.0f;

    public float CameraElevationDegrees { get; set; } = 50.0f;

    public float CameraZoomFactor { get; set; } = 1.35f;

    public float FlyPositionX { get; set; }

    public float FlyPositionY { get; set; }

    public float FlyPositionZ { get; set; }

    public float FlyMoveSpeed { get; set; } = 1.0f;

    public void Normalize()
    {
        InputPath = InputPath?.Trim() ?? string.Empty;
        if (!Enum.IsDefined(Variant))
            Variant = WowViewerModelOutputVariant.Predicted;
        if (!Enum.IsDefined(CameraMode))
            CameraMode = WowViewerModelOutputCameraMode.Orbit;

        CameraTargetOffsetX = float.IsFinite(CameraTargetOffsetX) ? CameraTargetOffsetX : 0.0f;
        CameraTargetOffsetY = float.IsFinite(CameraTargetOffsetY) ? CameraTargetOffsetY : 0.0f;
        CameraTargetOffsetZ = float.IsFinite(CameraTargetOffsetZ) ? CameraTargetOffsetZ : 0.0f;
        CameraAzimuthDegrees = float.IsFinite(CameraAzimuthDegrees) ? CameraAzimuthDegrees : 45.0f;
        CameraElevationDegrees = float.IsFinite(CameraElevationDegrees)
            ? Math.Clamp(CameraElevationDegrees, -85.0f, 85.0f)
            : 50.0f;
        CameraZoomFactor = float.IsFinite(CameraZoomFactor)
            ? Math.Clamp(CameraZoomFactor, 0.2f, 6.0f)
            : 1.35f;
        FlyPositionX = float.IsFinite(FlyPositionX) ? FlyPositionX : 0.0f;
        FlyPositionY = float.IsFinite(FlyPositionY) ? FlyPositionY : 0.0f;
        FlyPositionZ = float.IsFinite(FlyPositionZ) ? FlyPositionZ : 0.0f;
        FlyMoveSpeed = float.IsFinite(FlyMoveSpeed)
            ? Math.Clamp(FlyMoveSpeed, 0.1f, 8.0f)
            : 1.0f;
    }

    public Vector3 GetTargetOffset()
    {
        Normalize();
        return new Vector3(CameraTargetOffsetX, CameraTargetOffsetY, CameraTargetOffsetZ);
    }

    public void SetTargetOffset(Vector3 value)
    {
        CameraTargetOffsetX = value.X;
        CameraTargetOffsetY = value.Y;
        CameraTargetOffsetZ = value.Z;
        Normalize();
    }

    public Vector3 GetFlyPosition()
    {
        Normalize();
        return new Vector3(FlyPositionX, FlyPositionY, FlyPositionZ);
    }

    public void SetFlyPosition(Vector3 value)
    {
        FlyPositionX = value.X;
        FlyPositionY = value.Y;
        FlyPositionZ = value.Z;
        Normalize();
    }

    public bool HasInput()
    {
        Normalize();
        return !string.IsNullOrWhiteSpace(InputPath);
    }

    public string Describe()
    {
        Normalize();
        string variant = Variant == WowViewerModelOutputVariant.Predicted ? "predicted" : "wdl baseline";
        return string.IsNullOrWhiteSpace(InputPath)
            ? $"(no model-output root selected, variant={variant})"
            : $"{Path.GetFullPath(InputPath)} [{variant}]";
    }
}

internal readonly record struct ModelOutputVertex(Vector3 Position, Vector3 Normal, Vector2 TexCoord);

internal readonly record struct ModelOutputVertexColor(float R, float G, float B, float A);

internal sealed class ModelOutputObjectPlacement
{
    public required string TileName { get; init; }

    public required int TileX { get; init; }

    public required int TileY { get; init; }

    public required int UniqueId { get; init; }

    public required string Name { get; init; }

    public required string Category { get; init; }

    public required string ModelPath { get; init; }

    public required Vector3 Position { get; init; }

    public required Vector3 BoundsMin { get; init; }

    public required Vector3 BoundsMax { get; init; }
}

internal sealed class ModelOutputTileGeometry
{
    public required string TileName { get; init; }

    public required int TileX { get; init; }

    public required int TileY { get; init; }

    public required string ObjPath { get; init; }

    public required string TexturePath { get; init; }

    public required ModelOutputVertex[] Vertices { get; init; }

    public required uint[] Indices { get; init; }
}

internal sealed class ModelOutputScene
{
    public required string SourcePath { get; init; }

    public required WowViewerModelOutputVariant Variant { get; init; }

    public required float TileWorldSize { get; init; }

    public required bool CenterMesh { get; init; }

    public required TimeSpan LoadDuration { get; init; }

    public required IReadOnlyList<ModelOutputTileGeometry> Tiles { get; init; }

    public required IReadOnlyList<ModelOutputObjectPlacement> Objects { get; init; }

    public required Vector3 BoundsMin { get; init; }

    public required Vector3 BoundsMax { get; init; }

    public required int VertexCount { get; init; }

    public required int TriangleCount { get; init; }

    public required int ObjectCount { get; init; }
}

internal static class ModelOutputSceneLoader
{
    private const float DefaultTileWorldSize = 533.3333333333334f;

    public static ModelOutputScene Load(string inputPath, WowViewerModelOutputVariant variant)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputPath);

        string fullPath = Path.GetFullPath(inputPath);
        Stopwatch stopwatch = Stopwatch.StartNew();

        ModelOutputScene scene = Directory.Exists(fullPath)
            ? LoadFromDirectory(fullPath, variant)
            : LoadFromSummary(fullPath, variant);

        stopwatch.Stop();
        return new ModelOutputScene
        {
            SourcePath = scene.SourcePath,
            Variant = scene.Variant,
            TileWorldSize = scene.TileWorldSize,
            CenterMesh = scene.CenterMesh,
            LoadDuration = stopwatch.Elapsed,
            Tiles = scene.Tiles,
            Objects = scene.Objects,
            BoundsMin = scene.BoundsMin,
            BoundsMax = scene.BoundsMax,
            VertexCount = scene.VertexCount,
            TriangleCount = scene.TriangleCount,
            ObjectCount = scene.ObjectCount,
        };
    }

    private static ModelOutputScene LoadFromDirectory(string directoryPath, WowViewerModelOutputVariant variant)
    {
        string summaryPath = Path.Combine(directoryPath, "batch_inference_summary.json");
        if (File.Exists(summaryPath))
            return LoadFromSummary(summaryPath, variant);

        string suffix = variant == WowViewerModelOutputVariant.Predicted
            ? "_predicted_terrain.obj"
            : "_wdl_baseline_terrain.obj";

        List<ModelOutputTileGeometry> tiles = [];
        foreach (string objPath in Directory.EnumerateFiles(directoryPath, $"*{suffix}", SearchOption.TopDirectoryOnly)
                     .OrderBy(static path => path, StringComparer.OrdinalIgnoreCase))
        {
            string stem = Path.GetFileNameWithoutExtension(objPath);
            if (!TryParseTilePlacement(stem, variant, out string tileName, out int tileX, out int tileY))
                continue;

            string texturePath = ResolveTexturePathFromObj(objPath);
            tiles.Add(LoadTile(tileName, tileX, tileY, objPath, texturePath, DefaultTileWorldSize));
        }

        if (tiles.Count == 0)
            throw new InvalidDataException($"No {(variant == WowViewerModelOutputVariant.Predicted ? "predicted" : "WDL baseline")} OBJ tiles were found under {directoryPath}.");

        return BuildScene(directoryPath, variant, DefaultTileWorldSize, centerMesh: true, tiles, []);
    }

    private static ModelOutputScene LoadFromSummary(string summaryPath, WowViewerModelOutputVariant variant)
    {
        if (!File.Exists(summaryPath))
            throw new FileNotFoundException($"Model-output summary not found: {summaryPath}", summaryPath);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(summaryPath));
        JsonElement root = document.RootElement;
        float tileWorldSize = root.TryGetProperty("tile_world_size", out JsonElement tileWorldSizeElement)
            && tileWorldSizeElement.TryGetDouble(out double tileWorldSizeValue)
                ? (float)tileWorldSizeValue
                : DefaultTileWorldSize;
        bool centerMesh = !root.TryGetProperty("center_mesh", out JsonElement centerMeshElement) || centerMeshElement.GetBoolean();

        if (!root.TryGetProperty("entries", out JsonElement entriesElement) || entriesElement.ValueKind != JsonValueKind.Array)
            throw new InvalidDataException($"Summary does not contain an 'entries' array: {summaryPath}");

        string baseDirectory = Path.GetDirectoryName(summaryPath) ?? Directory.GetCurrentDirectory();
        IReadOnlyDictionary<string, string> sourceJsonLookup = LoadSourceJsonLookup(root, baseDirectory);
        Dictionary<string, IReadOnlyList<ModelOutputObjectPlacement>> sourceObjectCache = new(StringComparer.OrdinalIgnoreCase);
        List<ModelOutputTileGeometry> tiles = [];
        List<ModelOutputObjectPlacement> objects = [];
        foreach (JsonElement entry in entriesElement.EnumerateArray())
        {
            string tileName = entry.TryGetProperty("tile_name", out JsonElement tileNameElement)
                ? tileNameElement.GetString() ?? string.Empty
                : string.Empty;
            if (string.IsNullOrWhiteSpace(tileName) || !TryParseTileCoordinates(tileName, out int tileX, out int tileY))
                continue;

            string objKey = variant == WowViewerModelOutputVariant.Predicted
                ? (entry.TryGetProperty("flat_predicted_mesh_obj_path", out _) ? "flat_predicted_mesh_obj_path" : "predicted_mesh_obj_path")
                : (entry.TryGetProperty("flat_wdl_baseline_mesh_obj_path", out _) ? "flat_wdl_baseline_mesh_obj_path" : "wdl_baseline_mesh_obj_path");
            string textureKey = variant == WowViewerModelOutputVariant.Predicted
                ? (entry.TryGetProperty("flat_predicted_mesh_texture_path", out _) ? "flat_predicted_mesh_texture_path" : "predicted_mesh_texture_path")
                : (entry.TryGetProperty("flat_wdl_baseline_mesh_texture_path", out _) ? "flat_wdl_baseline_mesh_texture_path" : "wdl_baseline_mesh_texture_path");

            if (!TryGetRequiredString(entry, objKey, out string objPathValue))
                continue;

            string objPath = ResolveRelativeOrAbsolutePath(baseDirectory, objPathValue);
            string texturePath = TryGetRequiredString(entry, textureKey, out string texturePathValue)
                ? ResolveRelativeOrAbsolutePath(baseDirectory, texturePathValue)
                : ResolveTexturePathFromObj(objPath);
            tiles.Add(LoadTile(tileName, tileX, tileY, objPath, texturePath, tileWorldSize));

            string? sourceJsonPath = TryGetRequiredString(entry, "source_json", out string sourceJsonValue)
                ? ResolveRelativeOrAbsolutePath(baseDirectory, sourceJsonValue)
                : (sourceJsonLookup.TryGetValue(tileName, out string lookupSourceJson) ? lookupSourceJson : null);
            if (string.IsNullOrWhiteSpace(sourceJsonPath))
                continue;

            if (!sourceObjectCache.TryGetValue(sourceJsonPath, out IReadOnlyList<ModelOutputObjectPlacement>? cachedObjects))
            {
                cachedObjects = LoadSourceObjects(sourceJsonPath, tileName, tileX, tileY);
                sourceObjectCache[sourceJsonPath] = cachedObjects;
            }

            objects.AddRange(cachedObjects);
        }

        if (tiles.Count == 0)
            throw new InvalidDataException($"Summary did not expose any loadable {(variant == WowViewerModelOutputVariant.Predicted ? "predicted" : "WDL baseline")} mesh tiles: {summaryPath}");

        return BuildScene(summaryPath, variant, tileWorldSize, centerMesh, tiles, objects);
    }

    private static ModelOutputScene BuildScene(
        string sourcePath,
        WowViewerModelOutputVariant variant,
        float tileWorldSize,
        bool centerMesh,
        IReadOnlyList<ModelOutputTileGeometry> tiles,
        IReadOnlyList<ModelOutputObjectPlacement> objects)
    {
        Vector3 boundsMin = new(float.MaxValue, float.MaxValue, float.MaxValue);
        Vector3 boundsMax = new(float.MinValue, float.MinValue, float.MinValue);
        int vertexCount = 0;
        int triangleCount = 0;

        foreach (ModelOutputTileGeometry tile in tiles)
        {
            vertexCount += tile.Vertices.Length;
            triangleCount += tile.Indices.Length / 3;
            foreach (ModelOutputVertex vertex in tile.Vertices)
            {
                boundsMin = Vector3.Min(boundsMin, vertex.Position);
                boundsMax = Vector3.Max(boundsMax, vertex.Position);
            }
        }

        foreach (ModelOutputObjectPlacement obj in objects)
        {
            boundsMin = Vector3.Min(boundsMin, obj.BoundsMin);
            boundsMax = Vector3.Max(boundsMax, obj.BoundsMax);
        }

        return new ModelOutputScene
        {
            SourcePath = sourcePath,
            Variant = variant,
            TileWorldSize = tileWorldSize,
            CenterMesh = centerMesh,
            LoadDuration = TimeSpan.Zero,
            Tiles = tiles,
            Objects = objects,
            BoundsMin = boundsMin,
            BoundsMax = boundsMax,
            VertexCount = vertexCount,
            TriangleCount = triangleCount,
            ObjectCount = objects.Count,
        };
    }

    private static ModelOutputTileGeometry LoadTile(string tileName, int tileX, int tileY, string objPath, string texturePath, float tileWorldSize)
    {
        if (!File.Exists(objPath))
            throw new FileNotFoundException($"OBJ tile not found: {objPath}", objPath);

        Vector3 tileOffset = new(tileX * tileWorldSize, 0.0f, tileY * tileWorldSize);
        (ModelOutputVertex[] vertices, uint[] indices) = ParseObjGeometry(objPath, tileOffset);
        return new ModelOutputTileGeometry
        {
            TileName = tileName,
            TileX = tileX,
            TileY = tileY,
            ObjPath = objPath,
            TexturePath = texturePath,
            Vertices = vertices,
            Indices = indices,
        };
    }

    private static (ModelOutputVertex[] vertices, uint[] indices) ParseObjGeometry(string objPath, Vector3 tileOffset)
    {
        List<Vector3> rawPositions = [Vector3.Zero];
        List<Vector2> rawTexCoords = [Vector2.Zero];
        List<Vector3> outputPositions = [];
        List<Vector2> outputTexCoords = [];
        List<Vector3> accumulatedNormals = [];
        List<uint> indices = [];
        Dictionary<ObjVertexKey, int> vertexMap = [];

        foreach (string rawLine in File.ReadLines(objPath))
        {
            string line = rawLine.Trim();
            if (line.Length == 0 || line.StartsWith('#'))
                continue;

            if (line.StartsWith("v ", StringComparison.Ordinal))
            {
                string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 4)
                {
                    rawPositions.Add(new Vector3(
                        float.Parse(parts[1], CultureInfo.InvariantCulture),
                        float.Parse(parts[2], CultureInfo.InvariantCulture),
                        float.Parse(parts[3], CultureInfo.InvariantCulture)) + tileOffset);
                }
                continue;
            }

            if (line.StartsWith("vt ", StringComparison.Ordinal))
            {
                string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 3)
                {
                    rawTexCoords.Add(new Vector2(
                        float.Parse(parts[1], CultureInfo.InvariantCulture),
                        float.Parse(parts[2], CultureInfo.InvariantCulture)));
                }
                continue;
            }

            if (!line.StartsWith("f ", StringComparison.Ordinal))
                continue;

            string[] faceParts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (faceParts.Length < 4)
                continue;

            int[] faceVertexIndices = new int[faceParts.Length - 1];
            for (int partIndex = 1; partIndex < faceParts.Length; partIndex++)
            {
                ObjVertexKey key = ParseFaceVertex(faceParts[partIndex]);
                if (!vertexMap.TryGetValue(key, out int outputIndex))
                {
                    Vector3 position = rawPositions[key.PositionIndex];
                    Vector2 texCoord = key.TexCoordIndex > 0 && key.TexCoordIndex < rawTexCoords.Count
                        ? rawTexCoords[key.TexCoordIndex]
                        : Vector2.Zero;
                    outputIndex = outputPositions.Count;
                    outputPositions.Add(position);
                    outputTexCoords.Add(texCoord);
                    accumulatedNormals.Add(Vector3.Zero);
                    vertexMap.Add(key, outputIndex);
                }

                faceVertexIndices[partIndex - 1] = outputIndex;
            }

            for (int faceIndex = 1; faceIndex < faceVertexIndices.Length - 1; faceIndex++)
            {
                indices.Add((uint)faceVertexIndices[0]);
                indices.Add((uint)faceVertexIndices[faceIndex]);
                indices.Add((uint)faceVertexIndices[faceIndex + 1]);
            }
        }

        for (int index = 0; index < indices.Count; index += 3)
        {
            int a = (int)indices[index + 0];
            int b = (int)indices[index + 1];
            int c = (int)indices[index + 2];
            Vector3 ab = outputPositions[b] - outputPositions[a];
            Vector3 ac = outputPositions[c] - outputPositions[a];
            Vector3 faceNormal = Vector3.Cross(ab, ac);
            if (faceNormal.LengthSquared() <= 1e-10f)
                continue;

            accumulatedNormals[a] += faceNormal;
            accumulatedNormals[b] += faceNormal;
            accumulatedNormals[c] += faceNormal;
        }

        ModelOutputVertex[] vertices = new ModelOutputVertex[outputPositions.Count];
        for (int index = 0; index < outputPositions.Count; index++)
        {
            Vector3 normal = accumulatedNormals[index].LengthSquared() > 1e-10f
                ? Vector3.Normalize(accumulatedNormals[index])
                : Vector3.UnitY;
            vertices[index] = new ModelOutputVertex(outputPositions[index], normal, outputTexCoords[index]);
        }

        return (vertices, [.. indices]);
    }

    private static ObjVertexKey ParseFaceVertex(string token)
    {
        string[] parts = token.Split('/');
        int positionIndex = ParsePositiveObjIndex(parts[0]);
        int texCoordIndex = parts.Length >= 2 && !string.IsNullOrWhiteSpace(parts[1])
            ? ParsePositiveObjIndex(parts[1])
            : 0;
        return new ObjVertexKey(positionIndex, texCoordIndex);
    }

    private static int ParsePositiveObjIndex(string value)
    {
        int index = int.Parse(value, CultureInfo.InvariantCulture);
        if (index <= 0)
            throw new InvalidDataException($"Only positive OBJ indices are supported in exported terrain tiles. Got '{value}'.");
        return index;
    }

    private static string ResolveTexturePathFromObj(string objPath)
    {
        string mtlPath = Path.ChangeExtension(objPath, ".mtl");
        if (File.Exists(mtlPath))
        {
            string objDirectory = Path.GetDirectoryName(objPath) ?? Directory.GetCurrentDirectory();
            foreach (string rawLine in File.ReadLines(mtlPath))
            {
                string line = rawLine.Trim();
                if (!line.StartsWith("map_Kd ", StringComparison.OrdinalIgnoreCase))
                    continue;

                return ResolveRelativeOrAbsolutePath(objDirectory, line[7..].Trim());
            }
        }

        string stem = Path.GetFileNameWithoutExtension(objPath);
        return Path.Combine(Path.GetDirectoryName(objPath) ?? Directory.GetCurrentDirectory(), stem + "_texture.png");
    }

    private static IReadOnlyDictionary<string, string> LoadSourceJsonLookup(JsonElement summaryRoot, string baseDirectory)
    {
        if (!TryGetRequiredString(summaryRoot, "input_manifest", out string manifestPathValue))
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        string manifestPath = ResolveRelativeOrAbsolutePath(baseDirectory, manifestPathValue);
        if (!File.Exists(manifestPath))
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        try
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
            JsonElement root = document.RootElement;
            if (!root.TryGetProperty("entries", out JsonElement entriesElement) || entriesElement.ValueKind != JsonValueKind.Array)
                return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

            string manifestDirectory = Path.GetDirectoryName(manifestPath) ?? baseDirectory;
            Dictionary<string, string> lookup = new(StringComparer.OrdinalIgnoreCase);
            foreach (JsonElement entry in entriesElement.EnumerateArray())
            {
                if (!TryGetRequiredString(entry, "tile_name", out string tileName)
                    || !TryGetRequiredString(entry, "source_json", out string sourceJson))
                    continue;

                lookup[tileName] = ResolveRelativeOrAbsolutePath(manifestDirectory, sourceJson);
            }

            return lookup;
        }
        catch (IOException)
        {
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        }
        catch (JsonException)
        {
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        }
    }

    private static IReadOnlyList<ModelOutputObjectPlacement> LoadSourceObjects(string sourceJsonPath, string tileName, int tileX, int tileY)
    {
        if (!File.Exists(sourceJsonPath))
            return [];

        try
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(sourceJsonPath));
            JsonElement root = document.RootElement;
            JsonElement terrainData = root.TryGetProperty("terrain_data", out JsonElement terrainDataElement)
                ? terrainDataElement
                : root;
            if (!terrainData.TryGetProperty("objects", out JsonElement objectsElement) || objectsElement.ValueKind != JsonValueKind.Array)
                return [];

            List<ModelOutputObjectPlacement> objects = [];
            foreach (JsonElement element in objectsElement.EnumerateArray())
            {
                if (!TryGetSingle(element, "x", out float worldX)
                    || !TryGetSingle(element, "y", out float worldY)
                    || !TryGetSingle(element, "z", out float worldZ))
                    continue;

                Vector3 position = new(worldX, worldY, worldZ);
                float scale = TryGetSingle(element, "scale", out float parsedScale) && float.IsFinite(parsedScale)
                    ? parsedScale
                    : 1.0f;
                Vector3 localBoundsMin = TryGetVector3(element, "bounds_min", out Vector3 parsedBoundsMin)
                    ? parsedBoundsMin * scale
                    : new Vector3(-4.0f, 0.0f, -4.0f);
                Vector3 localBoundsMax = TryGetVector3(element, "bounds_max", out Vector3 parsedBoundsMax)
                    ? parsedBoundsMax * scale
                    : new Vector3(4.0f, 16.0f, 4.0f);

                objects.Add(new ModelOutputObjectPlacement
                {
                    TileName = tileName,
                    TileX = tileX,
                    TileY = tileY,
                    UniqueId = TryGetInt(element, "unique_id", out int uniqueId) ? uniqueId : 0,
                    Name = TryGetRequiredString(element, "name", out string nameValue) ? nameValue : string.Empty,
                    Category = TryGetRequiredString(element, "category", out string categoryValue) ? categoryValue.Trim().ToLowerInvariant() : "unknown",
                    ModelPath = TryGetRequiredString(element, "model_path", out string modelPath) ? modelPath : string.Empty,
                    Position = position,
                    BoundsMin = position + localBoundsMin,
                    BoundsMax = position + localBoundsMax,
                });
            }

            return objects;
        }
        catch (IOException)
        {
            return [];
        }
        catch (JsonException)
        {
            return [];
        }
    }

    private static bool TryGetRequiredString(JsonElement element, string propertyName, out string value)
    {
        value = string.Empty;
        if (!element.TryGetProperty(propertyName, out JsonElement property) || property.ValueKind != JsonValueKind.String)
            return false;

        value = property.GetString() ?? string.Empty;
        return !string.IsNullOrWhiteSpace(value);
    }

    private static bool TryGetInt(JsonElement element, string propertyName, out int value)
    {
        value = 0;
        if (!element.TryGetProperty(propertyName, out JsonElement property))
            return false;

        return property.ValueKind switch
        {
            JsonValueKind.Number => property.TryGetInt32(out value),
            JsonValueKind.String => int.TryParse(property.GetString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out value),
            _ => false,
        };
    }

    private static bool TryGetSingle(JsonElement element, string propertyName, out float value)
    {
        value = 0.0f;
        if (!element.TryGetProperty(propertyName, out JsonElement property))
            return false;

        return property.ValueKind switch
        {
            JsonValueKind.Number => property.TryGetSingle(out value),
            JsonValueKind.String => float.TryParse(property.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out value),
            _ => false,
        };
    }

    private static bool TryGetVector3(JsonElement element, string propertyName, out Vector3 value)
    {
        value = Vector3.Zero;
        if (!element.TryGetProperty(propertyName, out JsonElement property) || property.ValueKind != JsonValueKind.Array)
            return false;

        float[] components = new float[3];
        int index = 0;
        foreach (JsonElement component in property.EnumerateArray())
        {
            if (index >= 3)
                break;

            if (component.ValueKind != JsonValueKind.Number || !component.TryGetSingle(out float parsed))
                return false;

            components[index++] = parsed;
        }

        if (index != 3)
            return false;

        value = new Vector3(components[0], components[1], components[2]);
        return true;
    }

    private static string ResolveRelativeOrAbsolutePath(string baseDirectory, string path)
    {
        if (Path.IsPathRooted(path))
            return Path.GetFullPath(path);

        return Path.GetFullPath(Path.Combine(baseDirectory, path));
    }

    private static bool TryParseTilePlacement(string stem, WowViewerModelOutputVariant variant, out string tileName, out int tileX, out int tileY)
    {
        string suffix = variant == WowViewerModelOutputVariant.Predicted
            ? "_predicted_terrain"
            : "_wdl_baseline_terrain";

        tileName = string.Empty;
        tileX = 0;
        tileY = 0;
        if (!stem.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
            return false;

        string tileStem = stem[..^suffix.Length];
        if (!TryParseTileCoordinates(tileStem, out tileX, out tileY))
            return false;

        tileName = tileStem;
        return true;
    }

    private static bool TryParseTileCoordinates(string tileName, out int tileX, out int tileY)
    {
        tileX = 0;
        tileY = 0;
        int lastUnderscore = tileName.LastIndexOf('_');
        if (lastUnderscore <= 0 || lastUnderscore >= tileName.Length - 1)
            return false;

        int secondLastUnderscore = tileName.LastIndexOf('_', lastUnderscore - 1);
        if (secondLastUnderscore <= 0 || secondLastUnderscore >= lastUnderscore - 1)
            return false;

        return int.TryParse(tileName.AsSpan(secondLastUnderscore + 1, lastUnderscore - secondLastUnderscore - 1), NumberStyles.Integer, CultureInfo.InvariantCulture, out tileX)
            && int.TryParse(tileName.AsSpan(lastUnderscore + 1), NumberStyles.Integer, CultureInfo.InvariantCulture, out tileY);
    }

    private readonly record struct ObjVertexKey(int PositionIndex, int TexCoordIndex);
}
