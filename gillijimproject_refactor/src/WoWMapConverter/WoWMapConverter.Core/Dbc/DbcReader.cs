using System.Text;

namespace WoWMapConverter.Core.Dbc;

/// <summary>
/// Simple DBC file reader for Alpha and LK formats.
/// Supports reading AreaTable.dbc and Map.dbc without external dependencies.
/// </summary>
public class DbcReader
{
    private const uint DBC_MAGIC = 0x43424457; // 'WDBC'

    public DbcHeader Header { get; private set; }
    public List<Dictionary<int, object>> Rows { get; } = new();
    public List<string> StringTable { get; } = new();

    private byte[] _stringBlock = Array.Empty<byte>();

    /// <summary>
    /// Load a DBC file from disk.
    /// </summary>
    public static DbcReader Load(string path)
    {
        using var fs = File.OpenRead(path);
        using var reader = new BinaryReader(fs);
        return Load(reader);
    }

    /// <summary>
    /// Load a DBC file from a stream.
    /// </summary>
    public static DbcReader Load(BinaryReader reader)
    {
        var dbc = new DbcReader();

        // Read header
        var magic = reader.ReadUInt32();
        if (magic != DBC_MAGIC)
            throw new InvalidDataException($"Invalid DBC magic: 0x{magic:X8}");

        dbc.Header = new DbcHeader
        {
            RecordCount = reader.ReadUInt32(),
            FieldCount = reader.ReadUInt32(),
            RecordSize = reader.ReadUInt32(),
            StringBlockSize = reader.ReadUInt32()
        };

        // Read records
        var recordData = reader.ReadBytes((int)(dbc.Header.RecordCount * dbc.Header.RecordSize));

        // Read string block
        dbc._stringBlock = reader.ReadBytes((int)dbc.Header.StringBlockSize);

        // Parse records (treat all fields as uint32 for now)
        int fieldsPerRecord = (int)(dbc.Header.RecordSize / 4);
        for (int i = 0; i < dbc.Header.RecordCount; i++)
        {
            var row = new Dictionary<int, object>();
            int offset = (int)(i * dbc.Header.RecordSize);

            for (int f = 0; f < fieldsPerRecord; f++)
            {
                int fieldOffset = offset + (f * 4);
                uint value = BitConverter.ToUInt32(recordData, fieldOffset);
                row[f] = value;
            }

            dbc.Rows.Add(row);
        }

        return dbc;
    }

    /// <summary>
    /// Get a string from the string block at the given offset.
    /// </summary>
    public string GetString(uint offset)
    {
        if (offset >= _stringBlock.Length)
            return string.Empty;

        int end = (int)offset;
        while (end < _stringBlock.Length && _stringBlock[end] != 0)
            end++;

        return Encoding.UTF8.GetString(_stringBlock, (int)offset, end - (int)offset);
    }

    /// <summary>
    /// Get a field value as uint.
    /// </summary>
    public uint GetUInt(int row, int field)
    {
        if (row < 0 || row >= Rows.Count)
            return 0;
        if (!Rows[row].TryGetValue(field, out var val))
            return 0;
        return (uint)val;
    }

    /// <summary>
    /// Get a field value as int.
    /// </summary>
    public int GetInt(int row, int field)
    {
        return (int)GetUInt(row, field);
    }

    /// <summary>
    /// Get a field value as float.
    /// </summary>
    public float GetFloat(int row, int field)
    {
        var bytes = BitConverter.GetBytes(GetUInt(row, field));
        return BitConverter.ToSingle(bytes, 0);
    }

    /// <summary>
    /// Get a field value as string (from string block).
    /// </summary>
    public string GetString(int row, int field)
    {
        return GetString(GetUInt(row, field));
    }
}

public struct DbcHeader
{
    public uint RecordCount;
    public uint FieldCount;
    public uint RecordSize;
    public uint StringBlockSize;
}
