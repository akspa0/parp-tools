# D002: DBC Records

## Type
DBC Structure Component

## Source
DBC Format Documentation

## Description
Records in DBC files are arranged in a tabular structure where each record represents a row of data with a fixed number of fields (columns). The record section follows immediately after the DBC header and contains `recordCount` records, each with `fieldCount` fields. Every record has an identical structure specified by an external schema definition, as DBC files do not contain schema information within the file itself.

## Structure
Records are structured as a sequence of 4-byte (32-bit) fields, with each field potentially representing different data types:

```csharp
struct DBCRecord
{
    /*0x00*/ uint32_t field1;   // First field (type depends on schema)
    /*0x04*/ uint32_t field2;   // Second field
    // ... Additional fields as defined by fieldCount ...
    /*0xNN*/ uint32_t fieldN;   // Last field
};
```

## Field Types
Although all fields are stored as 32-bit values (uint32_t), they can represent different data types based on the schema definition:

| Logical Type | Storage Type | Description |
|--------------|--------------|-------------|
| Integer | uint32_t / int32_t | Signed or unsigned 32-bit integer |
| Float | float | 32-bit floating point number |
| String | uint32_t | Offset into the string block |
| Boolean | uint32_t | 0 for false, 1 for true |
| Flags | uint32_t | Bit flags where each bit represents a boolean |
| Foreign Key | uint32_t | Reference to an ID in another table |
| Enum | uint32_t | Enumerated value (interpreted based on schema) |

## Record Layout
- Records start immediately after the header (at offset 20)
- Each record is `recordSize` bytes (as specified in the header)
- Records are arranged sequentially without any padding or delimiters
- The total size of the record section is `recordCount * recordSize` bytes

## String References
String fields are stored as offsets into the string block:
- An offset of 0 typically represents an empty string or NULL value
- The offset is relative to the start of the string block
- The actual string is a null-terminated sequence of characters at the given offset

## Implementation Notes
- Records are always accessed by index (position) in the file
- Primary key fields (usually the first field) provide unique identification of records
- Foreign key relationships are not enforced by the file format but by the application logic
- Many DBC files sort records by primary key for efficient binary search
- Record fields are 4-byte aligned and have no padding between them
- The record structure is identical across all records in a given DBC file

## Implementation Example
```csharp
public class DBCRecord
{
    private uint[] _fields;
    
    public DBCRecord(uint fieldCount)
    {
        _fields = new uint[fieldCount];
    }
    
    public void Parse(BinaryReader reader, uint fieldCount)
    {
        _fields = new uint[fieldCount];
        
        for (int i = 0; i < fieldCount; i++)
        {
            _fields[i] = reader.ReadUInt32();
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (uint field in _fields)
        {
            writer.Write(field);
        }
    }
    
    // Access field as unsigned integer
    public uint GetUInt32(int fieldIndex)
    {
        if (fieldIndex < 0 || fieldIndex >= _fields.Length)
            throw new ArgumentOutOfRangeException(nameof(fieldIndex));
            
        return _fields[fieldIndex];
    }
    
    // Access field as signed integer
    public int GetInt32(int fieldIndex)
    {
        return (int)GetUInt32(fieldIndex);
    }
    
    // Access field as float
    public float GetFloat(int fieldIndex)
    {
        if (fieldIndex < 0 || fieldIndex >= _fields.Length)
            throw new ArgumentOutOfRangeException(nameof(fieldIndex));
            
        // Reinterpret the bits as a float
        return BitConverter.ToSingle(BitConverter.GetBytes(_fields[fieldIndex]), 0);
    }
    
    // Access field as boolean
    public bool GetBoolean(int fieldIndex)
    {
        return GetUInt32(fieldIndex) != 0;
    }
    
    // Check if a specific flag bit is set
    public bool HasFlag(int fieldIndex, uint flagBit)
    {
        return (GetUInt32(fieldIndex) & flagBit) != 0;
    }
    
    // Get string offset (to be resolved using string block)
    public uint GetStringOffset(int fieldIndex)
    {
        return GetUInt32(fieldIndex);
    }
    
    // Set field value
    public void SetField(int fieldIndex, uint value)
    {
        if (fieldIndex < 0 || fieldIndex >= _fields.Length)
            throw new ArgumentOutOfRangeException(nameof(fieldIndex));
            
        _fields[fieldIndex] = value;
    }
}
```

## DBC File Reader Example
```csharp
public class DBCReader
{
    public DBCHeader Header { get; private set; }
    public List<DBCRecord> Records { get; private set; }
    public byte[] StringBlock { get; private set; }
    
    public DBCReader()
    {
        Header = new DBCHeader();
        Records = new List<DBCRecord>();
    }
    
    public void Load(string filePath)
    {
        using (FileStream stream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
        using (BinaryReader reader = new BinaryReader(stream))
        {
            // Parse header
            Header.Parse(reader);
            
            // Parse records
            Records.Clear();
            for (int i = 0; i < Header.RecordCount; i++)
            {
                var record = new DBCRecord(Header.FieldCount);
                record.Parse(reader, Header.FieldCount);
                Records.Add(record);
            }
            
            // Read string block
            StringBlock = reader.ReadBytes((int)Header.StringBlockSize);
        }
    }
    
    // Helper method to get a string from the string block
    public string GetString(uint stringOffset)
    {
        if (stringOffset >= StringBlock.Length)
            return string.Empty;
            
        // Find the null terminator
        int stringEnd = (int)stringOffset;
        while (stringEnd < StringBlock.Length && StringBlock[stringEnd] != 0)
            stringEnd++;
            
        int stringLength = stringEnd - (int)stringOffset;
        return Encoding.UTF8.GetString(StringBlock, (int)stringOffset, stringLength);
    }
    
    // Helper method to resolve a string field in a record
    public string GetRecordString(int recordIndex, int fieldIndex)
    {
        if (recordIndex < 0 || recordIndex >= Records.Count)
            throw new ArgumentOutOfRangeException(nameof(recordIndex));
            
        uint stringOffset = Records[recordIndex].GetStringOffset(fieldIndex);
        return GetString(stringOffset);
    }
}
```

## Accessing String Data
Since string fields are stored as offsets into the string block, resolving a string requires two steps:

1. Get the string offset from the field value
2. Read the null-terminated string from the string block at that offset

For example:
```csharp
// To access a string field in a record:
uint stringOffset = record.GetUInt32(fieldIndex);
string value = string.Empty;

if (stringOffset < stringBlock.Length)
{
    // Find the null terminator
    int start = (int)stringOffset;
    int end = start;
    while (end < stringBlock.Length && stringBlock[end] != 0)
        end++;
        
    value = Encoding.UTF8.GetString(stringBlock, start, end - start);
}
```

## Common Field Patterns
DBC tables typically follow these common patterns:

1. First field is usually a unique ID (primary key)
2. String fields store offsets to the string block
3. Related records in different tables use matching IDs as foreign keys
4. Flag fields use bit positions to store multiple boolean values
5. Enum fields use integer values with predefined meanings

## Common Record Structures
While each DBC table has its own record structure, common patterns include:

- **Item Records**: ID, class, subclass, name offset, display info ID, quality, flags
- **Spell Records**: ID, school, name offset, description offset, icon ID, attributes
- **Creature Records**: ID, name offset, guild offset, flags, type, family, rank

## Relationship to Other Components
- **Header**: Defines recordCount, fieldCount, and recordSize
- **String Block**: Contains actual strings referenced by string fields

## Version Differences
- The record structure remains consistent from Classic through Wrath of the Lich King
- DB2 format introduces more complex record structures with variable sizes and sparse tables 