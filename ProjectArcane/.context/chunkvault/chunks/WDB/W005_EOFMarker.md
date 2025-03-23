# W005: EOF Marker

## Type
WDB/ADB Structure Component

## Source
WDB/ADB Format Documentation

## Description
The EOF (End-Of-File) marker is a simple structure found at the end of both WDB and ADB cache files. It serves as a terminal indicator that signals to the parser that it has reached the end of valid data in the file. The marker is critical for ensuring proper parsing and validation of cache files, especially in formats that contain variable-length records.

## Structure
The EOF marker is a fixed sequence of 8 null (zero) bytes:

```csharp
struct EOFMarker
{
    /*0x00*/ byte[8] nullBytes;    // 8 consecutive zero bytes
};
```

This consistent pattern allows parsers to identify when they've reached the end of the valid data in the file, even when the file content might have varied lengths or when handling truncated or corrupted files.

## Properties
| Name | Type | Description |
|------|------|-------------|
| nullBytes | byte[8] | Eight consecutive bytes with value 0 |

## Detection and Validation
A parser typically identifies the EOF marker by looking for 8 consecutive zero bytes. This can be particularly important when:

1. Reading records with variable lengths
2. Recovering from partially corrupted files
3. Determining if a file is complete or was truncated during transfer/saving

## Implementation Notes
- The EOF marker is identical in both WDB and ADB formats
- The marker should always be present, even in empty files (files with no records)
- Software that writes WDB/ADB files must remember to include the EOF marker
- Parsers should be cautious of data that might appear to be an EOF marker but is actually part of record data
- The presence of the EOF marker is often used as a file integrity check

## Implementation Example
```csharp
public class CacheFileParser
{
    // Constants
    private const int EOF_MARKER_SIZE = 8;
    
    // Helper method to check if current position contains EOF marker
    public static bool IsEOFMarker(BinaryReader reader)
    {
        // Save current position
        long currentPosition = reader.BaseStream.Position;
        
        // Check if there are at least 8 bytes left to read
        if (reader.BaseStream.Length - currentPosition < EOF_MARKER_SIZE)
            return false;
        
        // Read 8 bytes
        byte[] potentialEOF = reader.ReadBytes(EOF_MARKER_SIZE);
        
        // Restore original position
        reader.BaseStream.Position = currentPosition;
        
        // Check if all bytes are zero
        return potentialEOF.All(b => b == 0);
    }
    
    // Helper method to write EOF marker to a file
    public static void WriteEOFMarker(BinaryWriter writer)
    {
        for (int i = 0; i < EOF_MARKER_SIZE; i++)
            writer.Write((byte)0);
    }
    
    // Example method to parse WDB/ADB file
    public bool ParseCacheFile(string filePath)
    {
        try
        {
            using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            using (BinaryReader reader = new BinaryReader(fs))
            {
                // Parse header (WDB or ADB specific)
                // ...
                
                // Parse records until EOF marker is found
                while (!IsEOFMarker(reader) && reader.BaseStream.Position < reader.BaseStream.Length)
                {
                    // Parse record (implementation depends on WDB or ADB format)
                    // ...
                }
                
                // Validate EOF marker is present
                if (!IsEOFMarker(reader))
                {
                    Console.WriteLine("Warning: No EOF marker found at the end of the file");
                    return false;
                }
                
                // Skip EOF marker
                reader.ReadBytes(EOF_MARKER_SIZE);
                
                // Validate nothing comes after EOF marker
                if (reader.BaseStream.Position < reader.BaseStream.Length)
                {
                    Console.WriteLine("Warning: Data found after EOF marker");
                    return false;
                }
                
                return true;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error parsing cache file: {ex.Message}");
            return false;
        }
    }
    
    // Example method to check file integrity based on EOF marker
    public bool ValidateCacheFileIntegrity(string filePath)
    {
        try
        {
            using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            using (BinaryReader reader = new BinaryReader(fs))
            {
                // Get file length
                long fileLength = reader.BaseStream.Length;
                
                // Small files can't be valid (must at least have header + EOF marker)
                if (fileLength < 24) // Minimum size: 16-byte header + 8-byte EOF
                    return false;
                
                // Check for EOF marker at the end of the file
                reader.BaseStream.Position = fileLength - EOF_MARKER_SIZE;
                byte[] eofBytes = reader.ReadBytes(EOF_MARKER_SIZE);
                
                return eofBytes.All(b => b == 0);
            }
        }
        catch
        {
            return false;
        }
    }
}
```

## Usage Example
```csharp
// Example of using the EOF marker to validate a cache file
public class CacheManager
{
    private CacheFileParser _parser = new CacheFileParser();
    
    public bool VerifyAndRepairCache(string filePath)
    {
        // Check if file has a valid EOF marker
        bool hasValidEOF = _parser.ValidateCacheFileIntegrity(filePath);
        
        if (!hasValidEOF)
        {
            Console.WriteLine("Cache file is missing EOF marker or is corrupted");
            
            // Attempt to repair by loading and re-saving
            if (TryRepairCacheFile(filePath))
            {
                Console.WriteLine("Successfully repaired cache file");
                return true;
            }
            else
            {
                Console.WriteLine("Could not repair cache file, it will be rebuilt");
                return false;
            }
        }
        
        return true;
    }
    
    private bool TryRepairCacheFile(string filePath)
    {
        try
        {
            // Create backup
            string backupPath = filePath + ".bak";
            File.Copy(filePath, backupPath, true);
            
            // Try to parse the file to recover as much data as possible
            ADBFile adbFile = new ADBFile();
            if (adbFile.LoadFromFile(filePath, true)) // true = repair mode
            {
                // If we loaded anything successfully, save it back with proper EOF marker
                return adbFile.SaveToFile(filePath);
            }
            
            return false;
        }
        catch
        {
            return false;
        }
    }
}
```

## File Truncation Detection
The EOF marker enables detection of truncated files:

1. If a file is truncated in the middle of a record, the EOF marker will be missing
2. If a file is truncated after a complete record but before the EOF marker, this can also be detected
3. If a file appears to have data after the EOF marker, it may indicate corruption or improper concatenation

## Relationship to Other Components
- **WDB/ADB Header**: The header is at the beginning of the file, while the EOF marker is at the end
- **Records**: The EOF marker follows immediately after the last record in the file
- **Data Integrity**: Works with the header and record lengths to validate file completeness

## Validation Requirements
- The EOF marker must be exactly 8 bytes, all set to zero
- No data should exist in the file after the EOF marker
- The marker must be present for a cache file to be considered valid and complete
- Its position should be consistent with the record count and sizes defined in the header

## Historical Context
The 8-byte EOF marker has been a consistent feature across multiple versions of Blizzard's cache file formats. While other aspects of the formats have evolved from WDB to ADB and beyond, the EOF marker has remained unchanged, providing backward and forward compatibility for parsing tools. 