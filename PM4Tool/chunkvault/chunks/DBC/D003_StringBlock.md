# D003: DBC String Block

## Type
DBC Structure Component

## Source
DBC Format Documentation

## Description
The string block is a contiguous section at the end of a DBC file that contains all string data referenced by the records. It consists of a sequence of null-terminated strings, with each string referenced by an offset from record fields. The string block allows efficient storage of variable-length strings by storing them in a dedicated area instead of embedding them directly in the fixed-size records.

## Structure
The string block has no header or internal structure - it's simply a collection of null-terminated strings arranged sequentially:

```
+---------------+---+---------------+---+-----+---------------+---+
| String 1 data | 0 | String 2 data | 0 | ... | String N data | 0 |
+---------------+---+---------------+---+-----+---------------+---+
```

## Characteristics
- Located at the end of the file, after all records
- Size is defined by the `stringBlockSize` field in the DBC header
- Starts at offset `20 + (recordCount * recordSize)`
- Contains null-terminated strings in UTF-8 encoding
- First byte of the string block (offset 0) is typically a null byte representing empty strings
- Strings are not sorted or organized in any specific order

## String References
String fields in records contain offsets (uint32_t values) that point to positions within the string block:
- Offset 0 typically means an empty string or NULL value
- Offsets are relative to the start of the string block
- The referenced string continues until the next null byte (0x00)
- Multiple record fields can reference the same string by using the same offset

## Implementation Notes
- The string block may contain unused strings or padding
- There is no directory or index for the strings; they are only accessed via offsets from records
- Duplicate strings may be stored multiple times or may share the same offset for optimization
- The UTF-8 encoding allows for internationalization, including non-Latin characters
- The string block's total size is limited to 4GB due to 32-bit offsets
- There is no explicit string length stored; strings must be read until a null terminator

## Implementation Example
```csharp
public class StringBlock
{
    private byte[] _data;
    private Dictionary<string, uint> _stringToOffsetCache; // Optional optimization for writing
    
    public StringBlock(byte[] data)
    {
        _data = data;
        _stringToOffsetCache = new Dictionary<string, uint>();
    }
    
    public StringBlock()
    {
        _data = new byte[] { 0 }; // Initialize with a single null byte for empty strings
        _stringToOffsetCache = new Dictionary<string, uint>();
    }
    
    // Get a string at a specific offset
    public string GetString(uint offset)
    {
        if (offset >= _data.Length)
            return string.Empty;
            
        // Find the null terminator
        int stringEnd = (int)offset;
        while (stringEnd < _data.Length && _data[stringEnd] != 0)
            stringEnd++;
            
        int stringLength = stringEnd - (int)offset;
        if (stringLength == 0)
            return string.Empty;
            
        return Encoding.UTF8.GetString(_data, (int)offset, stringLength);
    }
    
    // Add a string to the block and return its offset
    public uint AddString(string value)
    {
        if (string.IsNullOrEmpty(value))
            return 0; // Empty strings point to offset 0
            
        // Check if we already have this string
        if (_stringToOffsetCache.TryGetValue(value, out uint existingOffset))
            return existingOffset;
            
        // Append the string to the block
        uint offset = (uint)_data.Length;
        byte[] stringBytes = Encoding.UTF8.GetBytes(value);
        
        // Resize the data array to accommodate the new string
        byte[] newData = new byte[_data.Length + stringBytes.Length + 1]; // +1 for null terminator
        Array.Copy(_data, newData, _data.Length);
        Array.Copy(stringBytes, 0, newData, _data.Length, stringBytes.Length);
        newData[newData.Length - 1] = 0; // Add null terminator
        
        _data = newData;
        _stringToOffsetCache[value] = offset;
        
        return offset;
    }
    
    // Get the total size of the string block
    public uint Size
    {
        get { return (uint)_data.Length; }
    }
    
    // Get the raw byte data
    public byte[] Data
    {
        get { return _data; }
    }
    
    // Write the string block to a binary writer
    public void Write(BinaryWriter writer)
    {
        writer.Write(_data);
    }
    
    // Create an empty string block with just a null terminator for empty strings
    public static StringBlock CreateEmpty()
    {
        return new StringBlock(new byte[] { 0 });
    }
}
```

## String Block Management Example
```csharp
public class DBCFile
{
    public DBCHeader Header { get; private set; }
    public List<DBCRecord> Records { get; private set; }
    public StringBlock Strings { get; private set; }
    
    public DBCFile()
    {
        Header = new DBCHeader();
        Records = new List<DBCRecord>();
        Strings = StringBlock.CreateEmpty();
    }
    
    // Set a string field in a record
    public void SetRecordString(int recordIndex, int fieldIndex, string value)
    {
        if (recordIndex < 0 || recordIndex >= Records.Count)
            throw new ArgumentOutOfRangeException(nameof(recordIndex));
            
        // Add the string to the string block
        uint offset = Strings.AddString(value);
        
        // Update the record field with the string offset
        Records[recordIndex].SetField(fieldIndex, offset);
    }
    
    // Get a string from a record
    public string GetRecordString(int recordIndex, int fieldIndex)
    {
        if (recordIndex < 0 || recordIndex >= Records.Count)
            throw new ArgumentOutOfRangeException(nameof(recordIndex));
            
        uint offset = Records[recordIndex].GetUInt32(fieldIndex);
        return Strings.GetString(offset);
    }
    
    // Save the DBC file
    public void Save(string filePath)
    {
        using (FileStream stream = new FileStream(filePath, FileMode.Create, FileAccess.Write))
        using (BinaryWriter writer = new BinaryWriter(stream))
        {
            // Update header with current counts
            Header.RecordCount = (uint)Records.Count;
            Header.StringBlockSize = Strings.Size;
            
            // Write header
            Header.Write(writer);
            
            // Write records
            foreach (var record in Records)
            {
                record.Write(writer);
            }
            
            // Write string block
            Strings.Write(writer);
        }
    }
}
```

## String Handling Considerations
When working with DBC string blocks, consider the following:

1. **Empty Strings**: Generally represented as offset 0, which is a null byte at the start of the string block.

2. **Internationalization**: DBC files often have different versions for different languages, with the string block containing localized text.

3. **String Deduplication**: When creating DBC files, it's efficient to store identical strings only once by reusing offsets.

4. **UTF-8 Encoding**: Strings are stored in UTF-8, which allows for variable-byte encoding of characters.

5. **Performance**: For better performance when reading, consider caching strings by their offset after first access.

## Optimization Techniques
When implementing string block handling, these optimizations can improve performance:

1. **String Caching**: Cache strings by their offset to avoid repeated parsing of the same string.

2. **Offset Mapping**: When writing files, maintain a map of strings to offsets to deduplicate strings.

3. **Lazy Parsing**: Only parse strings when needed rather than extracting all strings at load time.

4. **String Interning**: Use string interning to reduce memory usage for duplicate strings in the application.

## String Block in WDBC vs. Other Formats
The string block concept is found in several World of Warcraft file formats with some variations:

- **DBC**: Simple concatenated null-terminated strings
- **DB2**: Similar to DBC but may include additional string tables for different locales
- **WDB**: Similar structure but may have different string lookup mechanisms
- **WCH**: Similar concept but with potentially different encoding or structure

## Usage with Localization
World of Warcraft uses DBC/DB2 files for localization:

1. Non-localized DBC files have a single string block with text in one language.
2. Localized DBC files might exist in multiple versions, one per supported language.
3. Later DB2 formats support multiple string blocks for different languages within the same file.

## Relationship to Other Components
- **Header**: Defines `stringBlockSize` and helps locate the string block
- **Records**: Contain fields with offsets that reference strings in the string block 