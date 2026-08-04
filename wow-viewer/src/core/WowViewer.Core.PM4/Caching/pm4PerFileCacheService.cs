using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Caching;

/// <summary>
/// On-disk per-file PM4 overlay cache. Layout is a directory of small
/// gzip blobs, one per PM4 file, at
/// <c>output/cache/pm4-overlay/{dataSourceSegment}/{mapName}/files/{normalizedPath}.pm4cache</c>.
///
/// The cache key is the (dataSourceSegment, mapName, normalizedPath)
/// triple, so two different data sources or two different maps do not
/// collide. The on-disk entry is content-stamped: the entry's
/// <see cref="Pm4PerFileCacheEntry.FileLength"/> and
/// <see cref="Pm4PerFileCacheEntry.LastWriteTicks"/> are recorded at
/// write time and compared at read time; a mismatch is treated as a miss.
///
/// The reader/writer pair uses magic <c>PM4F</c> and version <c>8</c>.
/// This intentionally diverges from the per-window PM4 overlay cache
/// (magic <c>PM4C</c>, version <c>7</c>) so the two layers do not collide
/// on disk.
/// </summary>
public sealed class Pm4PerFileCacheService
{
    private const string EntryMagic = "PM4F";

    /// <summary>
    /// Bump this whenever the MEANING of a cached value changes, not only its layout.
    /// </summary>
    /// <remarks>
    /// Entries store geometry that has already been through the placement transform — triangles,
    /// lines, <c>PlacementAnchor</c>, <c>BoundsMin</c>/<c>BoundsMax</c> and the resolved planar
    /// flags are all post-transform. A cached tile therefore replays whatever placement was in
    /// effect when it was written, so a change to placement that leaves the layout untouched will
    /// silently keep rendering the old positions until this number moves.
    ///
    /// 8 -> 9: PM4 placement stopped being fitted per object against MPRL and is now pinned to the
    /// canonical frame (WorldSpace, identity planar transform, zero yaw). Every entry written
    /// before that carries fitted positions and must be discarded.
    /// </remarks>
    private const int EntryVersion = 9;

    private readonly string _cacheRoot;

    public Pm4PerFileCacheService(string cacheRoot)
    {
        _cacheRoot = cacheRoot;
        Directory.CreateDirectory(_cacheRoot);
    }

    public string CacheRoot => _cacheRoot;

    public static Pm4PerFileCacheService? CreateForDataSource(
        string cacheRoot,
        string dataSourceIdentity,
        string mapName)
    {
        if (string.IsNullOrWhiteSpace(cacheRoot))
            return null;
        if (string.IsNullOrWhiteSpace(mapName))
            return null;

        string dataSourceSegment = string.IsNullOrWhiteSpace(dataSourceIdentity)
            ? "default"
            : Convert.ToHexString(SHA1.HashData(Encoding.UTF8.GetBytes(dataSourceIdentity))).ToLowerInvariant();
        if (string.IsNullOrWhiteSpace(dataSourceSegment))
            dataSourceSegment = "default";

        string perFileRoot = Path.Combine(
            cacheRoot,
            dataSourceSegment,
            SanitizePathSegment(mapName),
            "files");
        return new Pm4PerFileCacheService(perFileRoot);
    }

    public bool TryRead(string normalizedPath, out Pm4PerFileCacheEntry? entry)
    {
        entry = null;
        string entryPath = GetEntryPath(normalizedPath);
        if (!File.Exists(entryPath))
            return false;

        try
        {
            using FileStream fileStream = File.OpenRead(entryPath);
            using GZipStream gzipStream = new(fileStream, CompressionMode.Decompress, leaveOpen: false);
            using BinaryReader reader = new(gzipStream, Encoding.UTF8, leaveOpen: false);

            string magic = reader.ReadString();
            if (!string.Equals(magic, EntryMagic, System.StringComparison.Ordinal))
                return false;

            int version = reader.ReadInt32();
            if (version != EntryVersion)
                return false;

            string storedNormalizedPath = reader.ReadString();
            if (!string.Equals(storedNormalizedPath, normalizedPath, System.StringComparison.OrdinalIgnoreCase))
                return false;

            long fileLength = reader.ReadInt64();
            long lastWriteTicks = reader.ReadInt64();

            int tileCount = reader.ReadInt32();
            var tiles = new List<Pm4CachedTile>(tileCount);
            for (int tileIndex = 0; tileIndex < tileCount; tileIndex++)
            {
                int tileX = reader.ReadInt32();
                int tileY = reader.ReadInt32();

                int positionRefsCount = reader.ReadInt32();
                var positionRefs = new List<System.Numerics.Vector3>(positionRefsCount);
                for (int i = 0; i < positionRefsCount; i++)
                    positionRefs.Add(ReadVector3(reader));

                int objectCount = reader.ReadInt32();
                var objects = new List<Pm4CachedObject>(objectCount);
                for (int objectIndex = 0; objectIndex < objectCount; objectIndex++)
                    objects.Add(ReadCachedObject(reader));

                tiles.Add(new Pm4CachedTile(tileX, tileY, positionRefs, objects));
            }

            entry = new Pm4PerFileCacheEntry(fileLength, lastWriteTicks, tiles);
            return true;
        }
        catch
        {
            return false;
        }
    }

    public bool Write(string normalizedPath, Pm4PerFileCacheEntry entry)
    {
        try
        {
            string entryPath = GetEntryPath(normalizedPath);
            Directory.CreateDirectory(Path.GetDirectoryName(entryPath)!);

            using FileStream fileStream = File.Create(entryPath);
            using GZipStream gzipStream = new(fileStream, CompressionLevel.Optimal, leaveOpen: false);
            using BinaryWriter writer = new(gzipStream, Encoding.UTF8, leaveOpen: false);

            writer.Write(EntryMagic);
            writer.Write(EntryVersion);
            writer.Write(normalizedPath);
            writer.Write(entry.FileLength);
            writer.Write(entry.LastWriteTicks);

            writer.Write(entry.Tiles.Count);
            for (int tileIndex = 0; tileIndex < entry.Tiles.Count; tileIndex++)
            {
                Pm4CachedTile tile = entry.Tiles[tileIndex];
                writer.Write(tile.TileX);
                writer.Write(tile.TileY);

                writer.Write(tile.PositionRefs.Count);
                for (int i = 0; i < tile.PositionRefs.Count; i++)
                    WriteVector3(writer, tile.PositionRefs[i]);

                writer.Write(tile.Objects.Count);
                for (int objectIndex = 0; objectIndex < tile.Objects.Count; objectIndex++)
                    WriteCachedObject(writer, tile.Objects[objectIndex]);
            }

            return true;
        }
        catch
        {
            return false;
        }
    }

    public bool Delete(string normalizedPath)
    {
        try
        {
            string entryPath = GetEntryPath(normalizedPath);
            if (File.Exists(entryPath))
            {
                File.Delete(entryPath);
                return true;
            }
            return false;
        }
        catch
        {
            return false;
        }
    }

    public void ClearForMap()
    {
        try
        {
            if (!Directory.Exists(_cacheRoot))
                return;
            foreach (string file in Directory.EnumerateFiles(_cacheRoot, "*.pm4cache", SearchOption.TopDirectoryOnly))
                File.Delete(file);
        }
        catch
        {
            // best effort
        }
    }

    private string GetEntryPath(string normalizedPath)
    {
        string sanitized = SanitizePathSegment(normalizedPath);
        if (string.IsNullOrWhiteSpace(sanitized))
            sanitized = "root";
        return Path.Combine(_cacheRoot, sanitized + ".pm4cache");
    }

    private static string SanitizePathSegment(string segment)
    {
        char[] invalid = Path.GetInvalidFileNameChars();
        var builder = new StringBuilder(segment.Length);
        foreach (char c in segment)
            builder.Append(Array.IndexOf(invalid, c) >= 0 ? '_' : c);
        return builder.ToString();
    }

    private static System.Numerics.Vector3 ReadVector3(BinaryReader reader)
    {
        float x = reader.ReadSingle();
        float y = reader.ReadSingle();
        float z = reader.ReadSingle();
        return new System.Numerics.Vector3(x, y, z);
    }

    private static void WriteVector3(BinaryWriter writer, System.Numerics.Vector3 v)
    {
        writer.Write(v.X);
        writer.Write(v.Y);
        writer.Write(v.Z);
    }

    private static Pm4CachedObject ReadCachedObject(BinaryReader reader)
    {
        string sourcePath = reader.ReadString();
        uint mshdField00 = reader.ReadUInt32();
        uint mshdRegionId = reader.ReadUInt32();
        uint mshdField08 = reader.ReadUInt32();
        uint ck24 = reader.ReadUInt32();
        byte ck24Type = reader.ReadByte();
        int objectPartId = reader.ReadInt32();
        uint linkGroupObjectId = reader.ReadUInt32();
        int linkedPositionRefCount = reader.ReadInt32();

        int totalCount = reader.ReadInt32();
        int normalCount = reader.ReadInt32();
        int terminatorCount = reader.ReadInt32();
        int floorMin = reader.ReadInt32();
        int floorMax = reader.ReadInt32();
        float headingMin = reader.ReadSingle();
        float headingMax = reader.ReadSingle();
        float headingMean = reader.ReadSingle();
        var summary = new Pm4LinkedPositionRefSummary(totalCount, normalCount, terminatorCount, floorMin, floorMax, headingMin, headingMax, headingMean);

        int surfaceCount = reader.ReadInt32();
        int totalIndexCount = reader.ReadInt32();
        byte dominantGroupKey = reader.ReadByte();
        byte dominantAttributeMask = reader.ReadByte();
        uint dominantMscnRefIndex = reader.ReadUInt32();
        float averageSurfaceHeight = reader.ReadSingle();
        var placementAnchor = ReadVector3(reader);
        float baseRotationRadians = reader.ReadSingle();
        bool swap = reader.ReadBoolean();
        bool invU = reader.ReadBoolean();
        bool invV = reader.ReadBoolean();
        var boundsMin = ReadVector3(reader);
        var boundsMax = ReadVector3(reader);

        int connectorCount = reader.ReadInt32();
        var connectorKeys = new List<Pm4CachedConnectorKey>(connectorCount);
        for (int i = 0; i < connectorCount; i++)
            connectorKeys.Add(new Pm4CachedConnectorKey(reader.ReadInt32(), reader.ReadInt32(), reader.ReadInt32()));

        int lineCount = reader.ReadInt32();
        var lines = new List<Pm4CachedLineSegment>(lineCount);
        for (int i = 0; i < lineCount; i++)
            lines.Add(new Pm4CachedLineSegment(ReadVector3(reader), ReadVector3(reader)));

        int triangleCount = reader.ReadInt32();
        var triangles = new List<Pm4CachedTriangle>(triangleCount);
        for (int i = 0; i < triangleCount; i++)
            triangles.Add(new Pm4CachedTriangle(ReadVector3(reader), ReadVector3(reader), ReadVector3(reader)));

        return new Pm4CachedObject(
            sourcePath, mshdField00, mshdRegionId, mshdField08,
            ck24, ck24Type, objectPartId, linkGroupObjectId, linkedPositionRefCount,
            summary, surfaceCount, totalIndexCount, dominantGroupKey, dominantAttributeMask,
            dominantMscnRefIndex, averageSurfaceHeight, placementAnchor, baseRotationRadians,
            swap, invU, invV, boundsMin, boundsMax, connectorKeys, lines, triangles);
    }

    private static void WriteCachedObject(BinaryWriter writer, Pm4CachedObject obj)
    {
        writer.Write(obj.SourcePath);
        writer.Write(obj.MshdField00);
        writer.Write(obj.MshdRegionId);
        writer.Write(obj.MshdField08);
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
        writer.Write(obj.DominantMscnRefIndex);
        writer.Write(obj.AverageSurfaceHeight);
        WriteVector3(writer, obj.PlacementAnchor);
        writer.Write(obj.BaseRotationRadians);
        writer.Write(obj.PlanarSwapPlanarAxes);
        writer.Write(obj.PlanarInvertU);
        writer.Write(obj.PlanarInvertV);
        WriteVector3(writer, obj.BoundsMin);
        WriteVector3(writer, obj.BoundsMax);

        writer.Write(obj.ConnectorKeys.Count);
        for (int i = 0; i < obj.ConnectorKeys.Count; i++)
        {
            writer.Write(obj.ConnectorKeys[i].X);
            writer.Write(obj.ConnectorKeys[i].Y);
            writer.Write(obj.ConnectorKeys[i].Z);
        }

        writer.Write(obj.Lines.Count);
        for (int i = 0; i < obj.Lines.Count; i++)
        {
            WriteVector3(writer, obj.Lines[i].From);
            WriteVector3(writer, obj.Lines[i].To);
        }

        writer.Write(obj.Triangles.Count);
        for (int i = 0; i < obj.Triangles.Count; i++)
        {
            WriteVector3(writer, obj.Triangles[i].A);
            WriteVector3(writer, obj.Triangles[i].B);
            WriteVector3(writer, obj.Triangles[i].C);
        }
    }
}
