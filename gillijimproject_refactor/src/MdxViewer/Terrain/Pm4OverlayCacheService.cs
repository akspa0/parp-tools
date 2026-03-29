using System.IO.Compression;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using MdxViewer.DataSources;
using Pm4PlanarTransform = WowViewer.Core.PM4.Models.Pm4PlanarTransform;

namespace MdxViewer.Terrain;

internal sealed class Pm4OverlayCacheService
{
    private const string CacheMagic = "PM4C";
    private const int CacheVersion = 3;
    private readonly string _cacheRoot;

    public Pm4OverlayCacheService(string cacheRoot)
    {
        _cacheRoot = cacheRoot;
        Directory.CreateDirectory(_cacheRoot);
    }

    public static Pm4OverlayCacheService? CreateForDataSource(IDataSource? dataSource)
    {
        if (dataSource == null)
            return null;

        string identity = BuildDataSourceIdentity(dataSource);
        string cacheSegment = string.IsNullOrWhiteSpace(identity)
            ? "default"
            : Convert.ToHexString(SHA1.HashData(Encoding.UTF8.GetBytes(identity))).ToLowerInvariant();
        if (string.IsNullOrWhiteSpace(cacheSegment))
            cacheSegment = "default";

        string cacheRoot = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output", "cache", "pm4-overlay", cacheSegment);
        return new Pm4OverlayCacheService(cacheRoot);
    }

    public static string BuildCandidateSignature(
        IDataSource dataSource,
        IReadOnlyList<string> pm4Candidates,
        bool splitByMdos,
        bool splitByConnectivity)
    {
        var builder = new StringBuilder();
        builder.Append("splitByMdos=").Append(splitByMdos ? '1' : '0').Append('\n');
        builder.Append("splitByConnectivity=").Append(splitByConnectivity ? '1' : '0').Append('\n');

        for (int i = 0; i < pm4Candidates.Count; i++)
        {
            string normalizedPath = NormalizeVirtualPath(pm4Candidates[i]);
            builder.Append(normalizedPath);

            if (TryGetLooseFileStamp(dataSource, normalizedPath, out long fileLength, out long lastWriteTicks))
                builder.Append('|').Append(fileLength).Append('|').Append(lastWriteTicks);

            builder.Append('\n');
        }

        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(builder.ToString()))).ToLowerInvariant();
    }

    public bool TryLoad(string mapName, string candidateSignature, out Pm4OverlayCacheData? data, out string? error)
    {
        data = null;
        error = null;

        string cachePath = GetCachePath(mapName);
        if (!File.Exists(cachePath))
            return false;

        try
        {
            using var fileStream = File.OpenRead(cachePath);
            using var gzipStream = new GZipStream(fileStream, CompressionMode.Decompress, leaveOpen: false);
            using var reader = new BinaryReader(gzipStream, Encoding.UTF8, leaveOpen: false);

            string magic = reader.ReadString();
            if (!string.Equals(magic, CacheMagic, StringComparison.Ordinal))
            {
                error = $"PM4 cache header mismatch for {mapName}.";
                return false;
            }

            int version = reader.ReadInt32();
            if (version != CacheVersion)
            {
                error = $"PM4 cache version mismatch for {mapName}: found {version}, expected {CacheVersion}.";
                return false;
            }

            string cachedMapName = reader.ReadString();
            string cachedSignature = reader.ReadString();
            if (!string.Equals(cachedMapName, mapName, StringComparison.OrdinalIgnoreCase))
            {
                error = $"PM4 cache map mismatch for {mapName}: found {cachedMapName}.";
                return false;
            }

            if (!string.Equals(cachedSignature, candidateSignature, StringComparison.OrdinalIgnoreCase))
            {
                error = $"PM4 cache stale for {mapName}: candidate signature changed.";
                return false;
            }

            data = ReadCacheData(reader, cachedMapName, cachedSignature);
            return true;
        }
        catch (Exception ex)
        {
            error = $"Failed to load PM4 cache for {mapName}: {ex.Message}";
            TryDeleteCorruptCache(cachePath);
            return false;
        }
    }

    public bool TrySave(Pm4OverlayCacheData data, out string? error)
    {
        error = null;

        string cachePath = GetCachePath(data.MapName);
        string tempPath = cachePath + ".tmp";

        try
        {
            Directory.CreateDirectory(_cacheRoot);

            using (var fileStream = File.Create(tempPath))
            using (var gzipStream = new GZipStream(fileStream, CompressionLevel.Fastest, leaveOpen: false))
            using (var writer = new BinaryWriter(gzipStream, Encoding.UTF8, leaveOpen: false))
            {
                writer.Write(CacheMagic);
                writer.Write(CacheVersion);
                writer.Write(data.MapName);
                writer.Write(data.CandidateSignature);
                WriteCacheData(writer, data);
            }

            File.Move(tempPath, cachePath, overwrite: true);
            return true;
        }
        catch (Exception ex)
        {
            error = $"Failed to save PM4 cache for {data.MapName}: {ex.Message}";
            try
            {
                if (File.Exists(tempPath))
                    File.Delete(tempPath);
            }
            catch
            {
            }

            return false;
        }
    }

    public bool TryDelete(string mapName, out string? error)
    {
        error = null;

        string cachePath = GetCachePath(mapName);
        if (!File.Exists(cachePath))
            return true;

        try
        {
            File.Delete(cachePath);
            return true;
        }
        catch (Exception ex)
        {
            error = $"Failed to delete PM4 cache for {mapName}: {ex.Message}";
            return false;
        }
    }

    private string GetCachePath(string mapName)
    {
        string safeFileName = string.Concat(mapName.Select(static ch => char.IsLetterOrDigit(ch) ? ch : '_'));
        if (string.IsNullOrWhiteSpace(safeFileName))
            safeFileName = "default";

        return Path.Combine(_cacheRoot, safeFileName.ToLowerInvariant() + ".bin");
    }

    private static string BuildDataSourceIdentity(IDataSource dataSource)
    {
        if (dataSource is MpqDataSource mpqDataSource)
        {
            var parts = new List<string> { mpqDataSource.GamePath };
            parts.AddRange(mpqDataSource.OverlayRoots.OrderBy(path => path, StringComparer.OrdinalIgnoreCase));
            return string.Join("||", parts);
        }

        return dataSource.Name ?? "default";
    }

    private static bool TryGetLooseFileStamp(IDataSource dataSource, string virtualPath, out long fileLength, out long lastWriteTicks)
    {
        fileLength = 0;
        lastWriteTicks = 0;

        if (dataSource is not MpqDataSource mpqDataSource)
            return false;

        string relativePath = virtualPath.Replace('/', Path.DirectorySeparatorChar);
        for (int i = mpqDataSource.OverlayRoots.Count - 1; i >= 0; i--)
        {
            string candidate = Path.Combine(mpqDataSource.OverlayRoots[i], relativePath);
            if (!File.Exists(candidate))
                continue;

            var fileInfo = new FileInfo(candidate);
            fileLength = fileInfo.Length;
            lastWriteTicks = fileInfo.LastWriteTimeUtc.Ticks;
            return true;
        }

        string baseCandidate = Path.Combine(mpqDataSource.GamePath, relativePath);
        if (!File.Exists(baseCandidate))
            return false;

        var baseFileInfo = new FileInfo(baseCandidate);
        fileLength = baseFileInfo.Length;
        lastWriteTicks = baseFileInfo.LastWriteTimeUtc.Ticks;
        return true;
    }

    private static string NormalizeVirtualPath(string path) => path.Replace('\\', '/').ToLowerInvariant();

    private static void TryDeleteCorruptCache(string cachePath)
    {
        try
        {
            if (File.Exists(cachePath))
                File.Delete(cachePath);
        }
        catch
        {
        }
    }

    private static Pm4OverlayCacheData ReadCacheData(BinaryReader reader, string mapName, string candidateSignature)
    {
        int totalFiles = reader.ReadInt32();
        int loadedFiles = reader.ReadInt32();
        int objectCount = reader.ReadInt32();
        int lineCount = reader.ReadInt32();
        int triangleCount = reader.ReadInt32();
        int positionRefCount = reader.ReadInt32();
        int rejectedLongEdges = reader.ReadInt32();
        float minObjectZ = reader.ReadSingle();
        float maxObjectZ = reader.ReadSingle();

        int tileCount = reader.ReadInt32();
        var tiles = new List<Pm4OverlayCacheTile>(tileCount);
        for (int tileIndex = 0; tileIndex < tileCount; tileIndex++)
        {
            int tileX = reader.ReadInt32();
            int tileY = reader.ReadInt32();

            int positionRefsCount = reader.ReadInt32();
            var positionRefs = new List<Vector3>(positionRefsCount);
            for (int i = 0; i < positionRefsCount; i++)
                positionRefs.Add(ReadVector3(reader));

            int objectEntryCount = reader.ReadInt32();
            var objects = new List<Pm4OverlayCacheObject>(objectEntryCount);
            for (int objectIndex = 0; objectIndex < objectEntryCount; objectIndex++)
            {
                string sourcePath = reader.ReadString();
                uint ck24 = reader.ReadUInt32();
                byte ck24Type = reader.ReadByte();
                int objectPartId = reader.ReadInt32();
                uint linkGroupObjectId = reader.ReadUInt32();
                int linkedPositionRefCount = reader.ReadInt32();

                var linkedSummary = new Pm4LinkedPositionRefSummary(
                    reader.ReadInt32(),
                    reader.ReadInt32(),
                    reader.ReadInt32(),
                    reader.ReadInt32(),
                    reader.ReadInt32(),
                    reader.ReadSingle(),
                    reader.ReadSingle(),
                    reader.ReadSingle());

                int surfaceCount = reader.ReadInt32();
                int totalIndexCount = reader.ReadInt32();
                byte dominantGroupKey = reader.ReadByte();
                byte dominantAttributeMask = reader.ReadByte();
                uint dominantMdosIndex = reader.ReadUInt32();
                float averageSurfaceHeight = reader.ReadSingle();
                Vector3 placementAnchor = ReadVector3(reader);
                float baseRotationRadians = reader.ReadSingle();
                var planarTransform = new Pm4PlanarTransform(reader.ReadBoolean(), reader.ReadBoolean(), reader.ReadBoolean());
                Vector3 boundsMin = ReadVector3(reader);
                Vector3 boundsMax = ReadVector3(reader);

                int connectorCount = reader.ReadInt32();
                var connectorKeys = new List<Pm4ConnectorKey>(connectorCount);
                for (int connectorIndex = 0; connectorIndex < connectorCount; connectorIndex++)
                {
                    connectorKeys.Add(new Pm4ConnectorKey(
                        reader.ReadInt32(),
                        reader.ReadInt32(),
                        reader.ReadInt32()));
                }

                int lineEntryCount = reader.ReadInt32();
                var lines = new List<Pm4LineSegment>(lineEntryCount);
                for (int lineIndex = 0; lineIndex < lineEntryCount; lineIndex++)
                    lines.Add(new Pm4LineSegment(ReadVector3(reader), ReadVector3(reader)));

                int triangleEntryCount = reader.ReadInt32();
                var triangles = new List<Pm4Triangle>(triangleEntryCount);
                for (int triangleIndex = 0; triangleIndex < triangleEntryCount; triangleIndex++)
                {
                    triangles.Add(new Pm4Triangle(
                        ReadVector3(reader),
                        ReadVector3(reader),
                        ReadVector3(reader)));
                }

                objects.Add(new Pm4OverlayCacheObject(
                    sourcePath,
                    ck24,
                    ck24Type,
                    objectPartId,
                    linkGroupObjectId,
                    linkedPositionRefCount,
                    linkedSummary,
                    lines,
                    triangles,
                    surfaceCount,
                    totalIndexCount,
                    dominantGroupKey,
                    dominantAttributeMask,
                    dominantMdosIndex,
                    averageSurfaceHeight,
                    placementAnchor,
                    baseRotationRadians,
                    planarTransform,
                    boundsMin,
                    boundsMax,
                    connectorKeys));
            }

            tiles.Add(new Pm4OverlayCacheTile(tileX, tileY, objects, positionRefs));
        }

        return new Pm4OverlayCacheData(
            mapName,
            candidateSignature,
            totalFiles,
            loadedFiles,
            objectCount,
            lineCount,
            triangleCount,
            positionRefCount,
            rejectedLongEdges,
            minObjectZ,
            maxObjectZ,
            tiles);
    }

    private static void WriteCacheData(BinaryWriter writer, Pm4OverlayCacheData data)
    {
        writer.Write(data.TotalFiles);
        writer.Write(data.LoadedFiles);
        writer.Write(data.ObjectCount);
        writer.Write(data.LineCount);
        writer.Write(data.TriangleCount);
        writer.Write(data.PositionRefCount);
        writer.Write(data.RejectedLongEdges);
        writer.Write(data.MinObjectZ);
        writer.Write(data.MaxObjectZ);

        writer.Write(data.Tiles.Count);
        for (int tileIndex = 0; tileIndex < data.Tiles.Count; tileIndex++)
        {
            Pm4OverlayCacheTile tile = data.Tiles[tileIndex];
            writer.Write(tile.TileX);
            writer.Write(tile.TileY);

            writer.Write(tile.PositionRefs.Count);
            for (int i = 0; i < tile.PositionRefs.Count; i++)
                WriteVector3(writer, tile.PositionRefs[i]);

            writer.Write(tile.Objects.Count);
            for (int objectIndex = 0; objectIndex < tile.Objects.Count; objectIndex++)
            {
                Pm4OverlayCacheObject obj = tile.Objects[objectIndex];
                writer.Write(obj.SourcePath);
                writer.Write(obj.Ck24);
                writer.Write(obj.Ck24Type);
                writer.Write(obj.ObjectPartId);
                writer.Write(obj.LinkGroupObjectId);
                writer.Write(obj.LinkedPositionRefCount);

                writer.Write(obj.LinkedPositionRefSummary.TotalCount);
                writer.Write(obj.LinkedPositionRefSummary.NormalCount);
                writer.Write(obj.LinkedPositionRefSummary.TerminatorCount);
                writer.Write(obj.LinkedPositionRefSummary.FloorMin);
                writer.Write(obj.LinkedPositionRefSummary.FloorMax);
                writer.Write(obj.LinkedPositionRefSummary.HeadingMinDegrees);
                writer.Write(obj.LinkedPositionRefSummary.HeadingMaxDegrees);
                writer.Write(obj.LinkedPositionRefSummary.HeadingMeanDegrees);

                writer.Write(obj.SurfaceCount);
                writer.Write(obj.TotalIndexCount);
                writer.Write(obj.DominantGroupKey);
                writer.Write(obj.DominantAttributeMask);
                writer.Write(obj.DominantMdosIndex);
                writer.Write(obj.AverageSurfaceHeight);
                WriteVector3(writer, obj.PlacementAnchor);
                writer.Write(obj.BaseRotationRadians);
                writer.Write(obj.PlanarTransform.SwapPlanarAxes);
                writer.Write(obj.PlanarTransform.InvertU);
                writer.Write(obj.PlanarTransform.InvertV);
                WriteVector3(writer, obj.BoundsMin);
                WriteVector3(writer, obj.BoundsMax);

                writer.Write(obj.ConnectorKeys.Count);
                for (int connectorIndex = 0; connectorIndex < obj.ConnectorKeys.Count; connectorIndex++)
                {
                    Pm4ConnectorKey connector = obj.ConnectorKeys[connectorIndex];
                    writer.Write(connector.X);
                    writer.Write(connector.Y);
                    writer.Write(connector.Z);
                }

                writer.Write(obj.Lines.Count);
                for (int lineIndex = 0; lineIndex < obj.Lines.Count; lineIndex++)
                {
                    WriteVector3(writer, obj.Lines[lineIndex].From);
                    WriteVector3(writer, obj.Lines[lineIndex].To);
                }

                writer.Write(obj.Triangles.Count);
                for (int triangleIndex = 0; triangleIndex < obj.Triangles.Count; triangleIndex++)
                {
                    WriteVector3(writer, obj.Triangles[triangleIndex].A);
                    WriteVector3(writer, obj.Triangles[triangleIndex].B);
                    WriteVector3(writer, obj.Triangles[triangleIndex].C);
                }
            }
        }
    }

    private static Vector3 ReadVector3(BinaryReader reader) => new(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());

    private static void WriteVector3(BinaryWriter writer, Vector3 value)
    {
        writer.Write(value.X);
        writer.Write(value.Y);
        writer.Write(value.Z);
    }
}

internal sealed class Pm4OverlayCacheData
{
    public Pm4OverlayCacheData(
        string mapName,
        string candidateSignature,
        int totalFiles,
        int loadedFiles,
        int objectCount,
        int lineCount,
        int triangleCount,
        int positionRefCount,
        int rejectedLongEdges,
        float minObjectZ,
        float maxObjectZ,
        List<Pm4OverlayCacheTile> tiles)
    {
        MapName = mapName;
        CandidateSignature = candidateSignature;
        TotalFiles = totalFiles;
        LoadedFiles = loadedFiles;
        ObjectCount = objectCount;
        LineCount = lineCount;
        TriangleCount = triangleCount;
        PositionRefCount = positionRefCount;
        RejectedLongEdges = rejectedLongEdges;
        MinObjectZ = minObjectZ;
        MaxObjectZ = maxObjectZ;
        Tiles = tiles;
    }

    public string MapName { get; }
    public string CandidateSignature { get; }
    public int TotalFiles { get; }
    public int LoadedFiles { get; }
    public int ObjectCount { get; }
    public int LineCount { get; }
    public int TriangleCount { get; }
    public int PositionRefCount { get; }
    public int RejectedLongEdges { get; }
    public float MinObjectZ { get; }
    public float MaxObjectZ { get; }
    public List<Pm4OverlayCacheTile> Tiles { get; }
}

internal sealed class Pm4OverlayCacheTile
{
    public Pm4OverlayCacheTile(int tileX, int tileY, List<Pm4OverlayCacheObject> objects, List<Vector3> positionRefs)
    {
        TileX = tileX;
        TileY = tileY;
        Objects = objects;
        PositionRefs = positionRefs;
    }

    public int TileX { get; }
    public int TileY { get; }
    public List<Pm4OverlayCacheObject> Objects { get; }
    public List<Vector3> PositionRefs { get; }
}

internal sealed class Pm4OverlayCacheObject
{
    public Pm4OverlayCacheObject(
        string sourcePath,
        uint ck24,
        byte ck24Type,
        int objectPartId,
        uint linkGroupObjectId,
        int linkedPositionRefCount,
        Pm4LinkedPositionRefSummary linkedPositionRefSummary,
        List<Pm4LineSegment> lines,
        List<Pm4Triangle> triangles,
        int surfaceCount,
        int totalIndexCount,
        byte dominantGroupKey,
        byte dominantAttributeMask,
        uint dominantMdosIndex,
        float averageSurfaceHeight,
        Vector3 placementAnchor,
        float baseRotationRadians,
        Pm4PlanarTransform planarTransform,
        Vector3 boundsMin,
        Vector3 boundsMax,
        List<Pm4ConnectorKey> connectorKeys)
    {
        SourcePath = sourcePath;
        Ck24 = ck24;
        Ck24Type = ck24Type;
        ObjectPartId = objectPartId;
        LinkGroupObjectId = linkGroupObjectId;
        LinkedPositionRefCount = linkedPositionRefCount;
        LinkedPositionRefSummary = linkedPositionRefSummary;
        Lines = lines;
        Triangles = triangles;
        SurfaceCount = surfaceCount;
        TotalIndexCount = totalIndexCount;
        DominantGroupKey = dominantGroupKey;
        DominantAttributeMask = dominantAttributeMask;
        DominantMdosIndex = dominantMdosIndex;
        AverageSurfaceHeight = averageSurfaceHeight;
        PlacementAnchor = placementAnchor;
        BaseRotationRadians = baseRotationRadians;
        PlanarTransform = planarTransform;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        ConnectorKeys = connectorKeys;
    }

    public string SourcePath { get; }
    public uint Ck24 { get; }
    public byte Ck24Type { get; }
    public int ObjectPartId { get; }
    public uint LinkGroupObjectId { get; }
    public int LinkedPositionRefCount { get; }
    public Pm4LinkedPositionRefSummary LinkedPositionRefSummary { get; }
    public List<Pm4LineSegment> Lines { get; }
    public List<Pm4Triangle> Triangles { get; }
    public int SurfaceCount { get; }
    public int TotalIndexCount { get; }
    public byte DominantGroupKey { get; }
    public byte DominantAttributeMask { get; }
    public uint DominantMdosIndex { get; }
    public float AverageSurfaceHeight { get; }
    public Vector3 PlacementAnchor { get; }
    public float BaseRotationRadians { get; }
    public Pm4PlanarTransform PlanarTransform { get; }
    public Vector3 BoundsMin { get; }
    public Vector3 BoundsMax { get; }
    public List<Pm4ConnectorKey> ConnectorKeys { get; }
}