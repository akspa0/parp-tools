# P001: MCRC (CRC)

## Type
PD4 Integrity Chunk

## Source
PD4 Format Documentation

## Description
The MCRC chunk contains CRC (Cyclic Redundancy Check) data that is used to verify the integrity of the PD4 file. This chunk is specific to the PD4 format and is not present in PM4 files. It allows the game to detect corrupted files and ensure that the data read from the file is valid.

## Structure
The MCRC chunk has the following structure:

```csharp
struct MCRC
{
    /*0x00*/ uint32_t crc_value;
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| crc_value | uint32_t | The CRC-32 value calculated for the file content |

## Dependencies
None directly, but relates to all other chunks in the file as it provides integrity verification.

## Implementation Notes
- The chunk contains a single 32-bit unsigned integer representing the CRC-32 value
- The CRC value is typically calculated over the entire file, excluding the MCRC chunk itself
- For calculating the CRC, the standard CRC-32 algorithm (same as used in PNG, ZIP, etc.) is likely used
- When writing a file, the CRC should be calculated after all other chunks have been written
- When reading a file, the CRC can be verified to ensure the file has not been corrupted

## C# Implementation Example

```csharp
public class McrcChunk : IChunk
{
    public const string Signature = "MCRC";
    public uint CrcValue { get; set; }

    public McrcChunk()
    {
        CrcValue = 0;
    }

    public void Read(BinaryReader reader)
    {
        CrcValue = reader.ReadUInt32();
    }

    public void Write(BinaryWriter writer)
    {
        writer.Write(CrcValue);
    }

    // Calculate the CRC-32 value for a byte array
    public static uint CalculateCrc32(byte[] data)
    {
        // Standard CRC-32 polynomial
        const uint Polynomial = 0xEDB88320;
        uint crc = 0xFFFFFFFF;

        foreach (byte b in data)
        {
            crc ^= b;
            
            for (int i = 0; i < 8; i++)
            {
                if ((crc & 1) == 1)
                {
                    crc = (crc >> 1) ^ Polynomial;
                }
                else
                {
                    crc >>= 1;
                }
            }
        }

        return ~crc; // Finalize the CRC by inverting all bits
    }

    // Calculate the CRC-32 for a file, excluding the MCRC chunk itself
    public static uint CalculateFileCrc(Stream fileStream, long mcrcPosition, long mcrcSize)
    {
        using (var crcStream = new MemoryStream())
        {
            // Save the current position
            long originalPosition = fileStream.Position;
            
            // Reset to start of the file
            fileStream.Position = 0;
            
            byte[] buffer = new byte[8192]; // 8KB buffer
            long remaining = fileStream.Length;
            
            while (remaining > 0)
            {
                // Skip the MCRC chunk bytes
                if (fileStream.Position == mcrcPosition)
                {
                    fileStream.Position += mcrcSize;
                    remaining -= mcrcSize;
                    continue;
                }

                // Determine how many bytes to read
                int toRead = (int)Math.Min(buffer.Length, remaining);
                
                if (fileStream.Position < mcrcPosition && fileStream.Position + toRead > mcrcPosition)
                {
                    // Reading would cross MCRC chunk boundary
                    toRead = (int)(mcrcPosition - fileStream.Position);
                }
                
                // Read bytes
                int read = fileStream.Read(buffer, 0, toRead);
                
                if (read <= 0)
                    break; // End of file or error
                    
                // Write to CRC calculation stream
                crcStream.Write(buffer, 0, read);
                remaining -= read;
            }
            
            // Restore original position
            fileStream.Position = originalPosition;
            
            // Calculate CRC on the accumulated data
            return CalculateCrc32(crcStream.ToArray());
        }
    }

    // Verify that the stored CRC matches the calculated CRC
    public bool VerifyCrc(Stream fileStream, long mcrcPosition)
    {
        uint calculatedCrc = CalculateFileCrc(fileStream, mcrcPosition, 8); // 8 bytes: 4 for 'MCRC' + 4 for size
        return CrcValue == calculatedCrc;
    }
}
```

## Related Information
- This chunk is specific to the PD4 format and not present in PM4 files
- The CRC-32 algorithm is a standard error-detection mechanism used in many file formats
- The CRC value is calculated excluding the MCRC chunk itself to avoid a circular dependency
- This chunk likely appears near the beginning or end of the file
- When implementing a PD4 writer, the MCRC chunk should be updated whenever the file content changes
- The presence of a CRC check indicates that file integrity is important for PD4 files, suggesting they may be used in critical game systems 