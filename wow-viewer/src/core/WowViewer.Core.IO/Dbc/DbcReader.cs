using System.Text;

namespace WowViewer.Core.IO.Dbc;

public sealed class DbcReader
{
    private const uint DbcMagic = 0x43424457;

    private byte[] _stringBlock = [];

    public DbcHeader Header { get; private set; }

    public List<Dictionary<int, object>> Rows { get; } = [];

    public static DbcReader Load(byte[] data)
    {
        ArgumentNullException.ThrowIfNull(data);

        using MemoryStream stream = new(data, writable: false);
        using BinaryReader reader = new(stream);
        return Load(reader);
    }

    public static DbcReader Load(string path)
    {
        using FileStream stream = File.OpenRead(path);
        using BinaryReader reader = new(stream);
        return Load(reader);
    }

    public static DbcReader Load(BinaryReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);

        DbcReader dbc = new();

        uint magic = reader.ReadUInt32();
        if (magic != DbcMagic)
            throw new InvalidDataException($"Invalid DBC magic: 0x{magic:X8}");

        dbc.Header = new DbcHeader(
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32(),
            reader.ReadUInt32());

        byte[] recordData = reader.ReadBytes(checked((int)(dbc.Header.RecordCount * dbc.Header.RecordSize)));
        dbc._stringBlock = reader.ReadBytes(checked((int)dbc.Header.StringBlockSize));

        int fieldsPerRecord = checked((int)(dbc.Header.RecordSize / 4));
        for (int rowIndex = 0; rowIndex < dbc.Header.RecordCount; rowIndex++)
        {
            Dictionary<int, object> row = [];
            int rowOffset = checked((int)(rowIndex * dbc.Header.RecordSize));

            for (int fieldIndex = 0; fieldIndex < fieldsPerRecord; fieldIndex++)
            {
                int fieldOffset = rowOffset + (fieldIndex * 4);
                row[fieldIndex] = BitConverter.ToUInt32(recordData, fieldOffset);
            }

            dbc.Rows.Add(row);
        }

        return dbc;
    }

    public string GetString(uint offset)
    {
        if (offset >= _stringBlock.Length)
            return string.Empty;

        int end = checked((int)offset);
        while (end < _stringBlock.Length && _stringBlock[end] != 0)
            end++;

        return Encoding.UTF8.GetString(_stringBlock, checked((int)offset), end - checked((int)offset));
    }

    public uint GetUInt(int row, int field)
    {
        if (row < 0 || row >= Rows.Count)
            return 0;

        if (!Rows[row].TryGetValue(field, out object? value))
            return 0;

        return (uint)value;
    }

    public int GetInt(int row, int field)
    {
        return (int)GetUInt(row, field);
    }

    public float GetFloat(int row, int field)
    {
        return BitConverter.ToSingle(BitConverter.GetBytes(GetUInt(row, field)), 0);
    }

    public string GetString(int row, int field)
    {
        return GetString(GetUInt(row, field));
    }
}

public readonly record struct DbcHeader(
    uint RecordCount,
    uint FieldCount,
    uint RecordSize,
    uint StringBlockSize);