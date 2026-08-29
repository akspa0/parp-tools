using System.Text;

namespace WowViewer.Core.IO.Dbc;

/// <summary>
/// A map entry to be encoded into <c>Map.dbc</c>.
/// </summary>
public sealed record RosettaMapDbcEntry(
    uint Id,
    string Directory,
    uint InstanceType = 0,
    uint Pvp = 0,
    string MapName = "");

/// <summary>
/// An area entry to be encoded into <c>AreaTable.dbc</c>.
/// </summary>
public sealed record RosettaAreaTableDbcEntry(
    uint Id,
    uint ContinentId,
    uint ParentAreaId = 0,
    uint AreaBit = 0,
    uint Flags = 0,
    uint SoundAmbience = 0,
    uint ZoneMusic = 0,
    uint ZoneIntroMusic = 0,
    uint Level = 0,
    string AreaName = "");

/// <summary>
/// Generates authentic binary WDBC files (<c>Map.dbc</c> and <c>AreaTable.dbc</c>) for World of Warcraft Alpha 0.5.3.
/// </summary>
public static class RosettaDbcGenerator
{
    private const uint WdbcMagic = 0x43424457; // "WDBC"

    /// <summary>
    /// Builds a binary <c>Map.dbc</c> byte array in Alpha 0.5.3 format (5 uint32 fields per record).
    /// </summary>
    public static byte[] BuildAlphaMapDbc(IReadOnlyList<RosettaMapDbcEntry> entries)
    {
        ArgumentNullException.ThrowIfNull(entries);

        const uint fieldCount = 5;
        const uint recordSize = fieldCount * 4;

        using MemoryStream stringStream = new();
        stringStream.WriteByte(0); // leading null byte

        var rows = new List<uint[]>(entries.Count);
        foreach (RosettaMapDbcEntry entry in entries)
        {
            uint dirOffset = WriteDbcString(stringStream, entry.Directory);
            uint nameOffset = WriteDbcString(stringStream, string.IsNullOrWhiteSpace(entry.MapName) ? entry.Directory : entry.MapName);

            rows.Add([
                entry.Id,
                dirOffset,
                entry.InstanceType,
                entry.Pvp,
                nameOffset
            ]);
        }

        return AssembleDbc(fieldCount, recordSize, rows, stringStream);
    }

    /// <summary>
    /// Builds a binary <c>AreaTable.dbc</c> byte array in Alpha 0.5.3 format (14 uint32 fields per record).
    /// </summary>
    public static byte[] BuildAlphaAreaTableDbc(IReadOnlyList<RosettaAreaTableDbcEntry> entries)
    {
        ArgumentNullException.ThrowIfNull(entries);

        const uint fieldCount = 14;
        const uint recordSize = fieldCount * 4;

        using MemoryStream stringStream = new();
        stringStream.WriteByte(0); // leading null byte

        var rows = new List<uint[]>(entries.Count);
        foreach (RosettaAreaTableDbcEntry entry in entries)
        {
            uint nameOffset = WriteDbcString(stringStream, entry.AreaName);

            rows.Add([
                entry.Id,
                entry.ContinentId,
                entry.ParentAreaId,
                entry.AreaBit,
                entry.Flags,
                entry.SoundAmbience,
                entry.ZoneMusic,
                entry.ZoneIntroMusic,
                entry.Level,
                nameOffset,
                0u, 0u, 0u, 0u // string localization / padding fields in 0.5.3
            ]);
        }

        return AssembleDbc(fieldCount, recordSize, rows, stringStream);
    }

    /// <summary>
    /// Writes <c>Map.dbc</c> to the specified file path.
    /// </summary>
    public static void WriteMapDbc(string filePath, IReadOnlyList<RosettaMapDbcEntry> entries)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
        string? dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);

        byte[] bytes = BuildAlphaMapDbc(entries);
        File.WriteAllBytes(filePath, bytes);
    }

    /// <summary>
    /// Writes <c>AreaTable.dbc</c> to the specified file path.
    /// </summary>
    public static void WriteAreaTableDbc(string filePath, IReadOnlyList<RosettaAreaTableDbcEntry> entries)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
        string? dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);

        byte[] bytes = BuildAlphaAreaTableDbc(entries);
        File.WriteAllBytes(filePath, bytes);
    }

    private static uint WriteDbcString(MemoryStream stringStream, string value)
    {
        if (string.IsNullOrEmpty(value))
            return 0;

        uint offset = checked((uint)stringStream.Position);
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        stringStream.Write(bytes, 0, bytes.Length);
        stringStream.WriteByte(0);
        return offset;
    }

    private static byte[] AssembleDbc(
        uint fieldCount, uint recordSize, List<uint[]> rows, MemoryStream stringStream)
    {
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(WdbcMagic);
        writer.Write(checked((uint)rows.Count));
        writer.Write(fieldCount);
        writer.Write(recordSize);
        writer.Write(checked((uint)stringStream.Length));

        foreach (uint[] row in rows)
        {
            foreach (uint val in row)
            {
                writer.Write(val);
            }
        }

        stringStream.Position = 0;
        stringStream.CopyTo(stream);
        writer.Flush();

        return stream.ToArray();
    }
}
