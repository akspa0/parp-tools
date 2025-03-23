# D005: Field Storage Info

## Type
DB2 Structure Component

## Source
DB2 Format Documentation

## Description
The Field Storage Info structure is a component introduced in the WDB5 format (and continued in WDB6) that provides metadata about how each field in a record is stored. This component enables more sophisticated storage techniques like bit-packing, palletized data, and common data, allowing for significantly smaller file sizes compared to the fixed-field approach of DBC and early DB2 formats.

## Structure
The Field Storage Info appears immediately after the header in WDB5/WDB6 formats and contains an array of field definitions:

```csharp
struct FieldStorageInfo
{
    /*0x00*/ uint16_t fieldOffset;   // Bit offset within the record
    /*0x02*/ uint16_t fieldSize;     // Size in bits
    /*0x04*/ uint32_t additionalDataSize; // Size of additional data for this field
    /*0x08*/ uint32_t storageType;   // How the field is stored (see below)
    /*0x0C*/ uint32_t compressedSize; // Size after compression or 0 if not applicable
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| fieldOffset | uint16_t | Bit offset within the record where this field starts |
| fieldSize | uint16_t | Size of the field in bits (not necessarily a multiple of 8) |
| additionalDataSize | uint32_t | Size of additional data for this field (depends on storage type) |
| storageType | uint32_t | Indicates how the field is stored (see storage types below) |
| compressedSize | uint32_t | Size of data after compression (or 0 if not compressed) |

## Storage Types
The `storageType` field defines how a particular field is stored and accessed:

| Value | Type Name | Description |
|-------|-----------|-------------|
| 0 | None | Field is stored as-is within the record |
| 1 | Unsigned | Field is stored as an unsigned integer in the record |
| 2 | Signed | Field is stored as a signed integer in the record |
| 3 | Float | Field is stored as a floating-point value in the record |
| 4 | BitpackedSigned | Field is bit-packed as a signed integer (may use fewer bits) |
| 5 | BitpackedUnsigned | Field is bit-packed as an unsigned integer (may use fewer bits) |
| 6 | BitpackedFloat | Field is bit-packed as a float (may use fewer bits) |
| 7 | Palletized | Field uses a shared palette table for values |
| 8 | PalletizedIndexed | Field uses an indexed pallete table for values |
| 9 | StringOffset | Field is an offset into the string block |
| 10 | CommonDataIndex | Field is an index into a common data table (shared values) |
| 11 | ForeignKey | Field is a foreign key reference to another table |
| 12 | StringWithID | Field is a string reference with ID (WDB6+) |
| 13 | Complex | Field uses a complex storage type, requiring special handling |

## Field Storage Info Size
The total size of the Field Storage Info section is calculated as:
```
fieldStorageInfoSize = sizeof(FieldStorageInfo) * fieldCount
```

Where `fieldCount` is from the main DB2 header. Each `FieldStorageInfo` struct is 16 bytes (0x10).

## Implementation Notes
- The Field Storage Info section provides a way to optimize storage by using variable bit widths
- Bit-packed fields allow using exactly the number of bits needed to store a value's range
- Palletized data is used when a field has a small number of unique values
- Common data is used when many records share the same value for a field
- This structure enables significant compression compared to earlier formats
- The same field can be stored differently in different versions of the same file

## Implementation Example
```csharp
public class FieldStorageInfo
{
    public enum StorageType : uint
    {
        None = 0,
        Unsigned = 1,
        Signed = 2,
        Float = 3,
        BitpackedSigned = 4,
        BitpackedUnsigned = 5, 
        BitpackedFloat = 6,
        Palletized = 7,
        PalletizedIndexed = 8,
        StringOffset = 9,
        CommonDataIndex = 10,
        ForeignKey = 11,
        StringWithID = 12,
        Complex = 13
    }
    
    public ushort FieldOffset { get; set; }
    public ushort FieldSize { get; set; }
    public uint AdditionalDataSize { get; set; }
    public StorageType Storage { get; set; }
    public uint CompressedSize { get; set; }
    
    // Calculated properties
    public bool IsBitPacked => Storage == StorageType.BitpackedSigned || 
                              Storage == StorageType.BitpackedUnsigned || 
                              Storage == StorageType.BitpackedFloat;
    
    public bool IsPalletized => Storage == StorageType.Palletized || 
                               Storage == StorageType.PalletizedIndexed;
    
    public bool IsString => Storage == StorageType.StringOffset || 
                           Storage == StorageType.StringWithID;
    
    public void Parse(BinaryReader reader)
    {
        FieldOffset = reader.ReadUInt16();
        FieldSize = reader.ReadUInt16();
        AdditionalDataSize = reader.ReadUInt32();
        Storage = (StorageType)reader.ReadUInt32();
        CompressedSize = reader.ReadUInt32();
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(FieldOffset);
        writer.Write(FieldSize);
        writer.Write(AdditionalDataSize);
        writer.Write((uint)Storage);
        writer.Write(CompressedSize);
    }
    
    // Helper to get a field value from a record based on its storage type
    public object GetValue(byte[] recordData, Dictionary<int, object[]> pallets, 
                          Dictionary<int, object> commonData, string[] stringTable)
    {
        // Get the raw bits from the record data based on field offset and size
        ulong rawValue = ExtractBits(recordData, FieldOffset, FieldSize);
        
        switch (Storage)
        {
            case StorageType.None:
                return null;
                
            case StorageType.Unsigned:
                return rawValue;
                
            case StorageType.Signed:
                // Convert to signed based on bit size
                return ConvertToSigned(rawValue, FieldSize);
                
            case StorageType.Float:
                // Reinterpret bits as float
                return BitConverter.ToSingle(BitConverter.GetBytes((uint)rawValue), 0);
                
            case StorageType.BitpackedSigned:
                return ConvertToSigned(rawValue, FieldSize);
                
            case StorageType.BitpackedUnsigned:
                return rawValue;
                
            case StorageType.BitpackedFloat:
                // Special handling for bit-packed floats
                return UnpackFloat(rawValue, FieldSize);
                
            case StorageType.Palletized:
                // Look up value in pallet array using the raw value as index
                return pallets.TryGetValue((int)FieldOffset, out var pallet) ? pallet[rawValue] : null;
                
            case StorageType.PalletizedIndexed:
                // More complex indexed pallet lookup
                return LookupPalletizedIndexed(rawValue, pallets);
                
            case StorageType.StringOffset:
                // Get string from string table using the offset
                return rawValue < stringTable.Length ? stringTable[rawValue] : "";
                
            case StorageType.CommonDataIndex:
                // Look up in common data table
                return commonData.TryGetValue((int)rawValue, out var value) ? value : null;
                
            case StorageType.ForeignKey:
                // Just return the ID for the foreign table
                return rawValue;
                
            case StorageType.StringWithID:
                // Complex string with ID handling
                return HandleStringWithID(rawValue, stringTable);
                
            case StorageType.Complex:
                // Would need special handling based on field
                return null;
                
            default:
                return null;
        }
    }
    
    // Helper method to extract bits from byte array
    private ulong ExtractBits(byte[] data, int bitOffset, int bitCount)
    {
        // Calculate byte-aligned position
        int byteOffset = bitOffset / 8;
        int bitShift = bitOffset % 8;
        
        // Read enough bytes to cover the field
        int bytesToRead = (bitShift + bitCount + 7) / 8;
        ulong result = 0;
        
        for (int i = 0; i < bytesToRead && byteOffset + i < data.Length; i++)
        {
            result |= (ulong)data[byteOffset + i] << (i * 8);
        }
        
        // Shift out unwanted bits at the start
        result >>= bitShift;
        
        // Mask to keep only the desired bits
        ulong mask = bitCount == 64 ? ulong.MaxValue : ((1UL << bitCount) - 1);
        return result & mask;
    }
    
    // Helper to convert unsigned to signed value
    private long ConvertToSigned(ulong value, int bitSize)
    {
        // Check if the highest bit is set (negative number)
        bool negative = ((value >> (bitSize - 1)) & 1) != 0;
        
        if (negative)
        {
            // Create a mask for the sign extension
            ulong mask = ulong.MaxValue << bitSize;
            return (long)(value | mask);
        }
        return (long)value;
    }
    
    // Helper for float unpacking (simplified)
    private float UnpackFloat(ulong value, int bitSize)
    {
        // This would implement the specific bit-packing algorithm
        // used by WoW for compressed floats
        return 0.0f; // Placeholder
    }
    
    // Helper for palletized indexed lookup
    private object LookupPalletizedIndexed(ulong index, Dictionary<int, object[]> pallets)
    {
        // This would implement the specific indexed pallet lookup
        return null; // Placeholder
    }
    
    // Helper for string with ID handling
    private string HandleStringWithID(ulong value, string[] stringTable)
    {
        // This would implement the string with ID handling
        return ""; // Placeholder
    }
}
```

## Usage Example
```csharp
// Reading field storage info from a DB2 file
public void ReadFieldStorageInfo(BinaryReader reader, DB2Header header)
{
    // Position after the header
    reader.BaseStream.Position = header.GetHeaderSize();
    
    // Array to store field info
    var fieldInfos = new FieldStorageInfo[header.FieldCount];
    
    // Read field info for each field
    for (int i = 0; i < header.FieldCount; i++)
    {
        var fieldInfo = new FieldStorageInfo();
        fieldInfo.Parse(reader);
        fieldInfos[i] = fieldInfo;
    }
    
    Console.WriteLine($"Read {fieldInfos.Length} field storage info entries");
    
    // Display some stats about storage types
    int bitpacked = fieldInfos.Count(f => f.IsBitPacked);
    int palletized = fieldInfos.Count(f => f.IsPalletized);
    int strings = fieldInfos.Count(f => f.IsString);
    
    Console.WriteLine($"Fields by storage type:");
    Console.WriteLine($"- Bit-packed: {bitpacked}");
    Console.WriteLine($"- Palletized: {palletized}");
    Console.WriteLine($"- Strings: {strings}");
    Console.WriteLine($"- Standard: {fieldInfos.Length - bitpacked - palletized - strings}");
    
    // Now position at the end of the field storage info section
    reader.BaseStream.Position = header.GetHeaderSize() + header.FieldStorageInfoSize;
}
```

## Field Bit-Packing Benefits
Bit-packing allows for significant data compression:

1. **Fixed Byte Boundaries**: Traditional databases store data in fixed-size fields (1, 2, 4, or 8 bytes) 
2. **Bit-Level Storage**: DB2 can use exactly the number of bits needed:
   - Boolean values only need 1 bit instead of 8 bits (1 byte)
   - Values 0-3 only need 2 bits instead of 8 bits (1 byte)
   - Values 0-100 only need 7 bits instead of 8 bits (1 byte)
   - Values 0-1000 only need 10 bits instead of 16 bits (2 bytes)

For example, a table with many small-range fields (like booleans or small enums) might use only 25-30% of the space compared to a traditional fixed-field format.

## Palletized Data Concept
Palletized data is a compression technique where fields with a limited set of unique values store indices into a shared table of values:

1. **Traditional Storage**: Each record contains the full value, even if many records share the same value
2. **Palletized Storage**:
   - A palette table stores each unique value once
   - Record fields store an index into the palette
   - Greatly reduces size when the number of unique values is much smaller than the number of records

For example, a "Class" field in a character database might have only 12 unique values but appear in thousands of records. Instead of storing the class ID in each record, records store an index (0-11) into a palette table that contains the actual class IDs.

## Relationship to Other Components
- **DB2 Header**: Defines the size and location of the Field Storage Info block
- **Record Data**: Field Storage Info dictates how to interpret the bits in record data
- **Additional Data**: Points to palette tables, common data, and other special structures
- **String Block**: Some storage types reference the string block

## Evolution in Later Formats
The concept of Field Storage Info has evolved in later formats:

- **WDB5**: Introduced the basic bit-packing and palletized data concepts
- **WDB6**: Added more storage types and refined the approach
- **WDC1/2/3**: Further extended the concept with additional optimization techniques

## Validation Requirements
- Total Field Storage Info size must match the `fieldStorageInfoSize` value in the DB2 header
- Each field must have a valid storage type
- Bit offsets and sizes must be consistent with the record size
- For special storage types (palletized, etc.), the additional data must be present and correctly sized 