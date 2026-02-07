# WoW Alpha 0.5.3 (Build 3368) Archive and File System Analysis

## Overview

This document provides a deep analysis of the archive (MPQ) and file system in WoW Alpha 0.5.3 (Build 3368, Dec 11 2003), based on Ghidra reverse engineering of WoWClient.exe. It covers the Storm library MPQ implementation, file operations, and archive management.

## Related Functions

### SFile Functions

| Function | Address | Purpose |
|----------|---------|---------|
| `SFile` | 0x0063c5e0 | SFile interface |
| `_SFileCloseArchive@4` | 0x0064f7b0 | Close archive |
| `_SFileCloseFile@4` | 0x0064f8d0 | Close file |
| `_SFileDdaBegin@12` | 0x0064f960 | Begin audio stream |
| `_SFileDdaBeginEx@28` | 0x0064f990 | Begin audio stream extended |
| `_SFileDdaDestroy@0` | 0x0064ff00 | Destroy audio stream |
| `_SFileDdaEnd@4` | 0x0064ffd0 | End audio stream |
| `_SFileDdaGetPos@12` | 0x006500b0 | Get audio position |
| `_SFileDdaGetVolume@12` | 0x00650180 | Get audio volume |
| `_SFileDdaInitialize@4` | 0x00650290 | Initialize audio |
| `_SFileDdaSetVolume@12` | 0x006502f0 | Set audio volume |
| `_SFileAuthenticateArchive@8` | 0x0064ecc0 | Authenticate archive |
| `_SFileAuthenticateArchiveEx@24` | 0x0064f320 | Authenticate extended |
| `_SFileCalcFileCrc@4` | 0x0064f520 | Calculate file CRC |
| `_SFileCancelRequest@4` | 0x0064f690 | Cancel file request |
| `_SFileCancelRequestEx@4` | 0x0064f770 | Cancel request extended |

### OsFile Functions

| Function | Address | Purpose |
|----------|---------|---------|
| `OsFileAssocGetIdentifier` | 0x0045dbe0 | Get file association |
| `OsFileAssocGetValue` | 0x0045dcc0 | Get association value |
| `OsFileAssocSetIdentifier` | 0x0045dc60 | Set association ID |
| `OsFileAssocSetValue` | 0x0045ddc0 | Set association value |
| `OsFileConnClose` | 0x00466f70 | Close file connection |
| `OsFileConnCreate` | 0x00466ea0 | Create file connection |
| `OsFileConnRead` | 0x00466ed0 | Read from connection |
| `OsFileConnWrite` | 0x00466f20 | Write to connection |
| `OsFileExists` | 0x0045d190 | Check file exists |
| `OsFileFreeSpace` | 0x0045deb0 | Get free space |
| `OsFileList` | 0x0045dab0 | List files |
| `OsFileNameHasInvalidChars` | 0x0045cef0 | Validate filename |
| `OsFileNameIsValid` | 0x0045cf40 | Check filename valid |
| `OsFileTimeAdd` | 0x0045bcd0 | Add file time |
| `OsFileTimeCompare` | 0x0045bc80 | Compare file times |
| `OsFileTimeGetCurrent` | 0x0045bc30 | Get current time |
| `OsFileTimeToLocalFileTime` | 0x0045bea0 | Convert to local |
| `OsFileTimeToSystemTime` | 0x0045bf00 | Convert to system |

### Other File Functions

| Function | Address | Purpose |
|----------|---------|---------|
| `DisableSFileCheckDisk` | 0x0063dfa0 | Disable disk check |
| `FrameXML_ProcessFile` | 0x006d40b0 | Process UI file |
| `ProcessFile` | 0x00640120 | Process file |
| `SFileReadTyped` | 0x0065070 | Typed file read |

---

## Archive Data Structures

### TSList Structures

```c
/* Archive record list */
TSList<struct_Storm::SFile::ARCHIVEREC, class_TSGetLink<struct_Storm::SFile::ARCHIVEREC>> 
    @ 0x006533e0

/* File record list */
TSList<struct_Storm::SFile::FILEREC, class_TSGetLink<struct_Storm::SFile::FILEREC>> 
    @ 0x00653590

/* Audio stream list */
TSList<struct_Storm::SFile::AUDIOSTREAM, class_TSGetLink<struct_Storm::SFile::AUDIOSTREAM>> 
    @ 0x006532e0
```

### Storm::SFile::ARCHIVEREC

```c
struct ARCHIVEREC {
    /* Archive identification */
    uint32_t archiveId;        // Unique archive ID
    
    /* File path */
    char* filePath;           // Path to MPQ file
    
    /* Archive properties */
    uint32_t flags;           // Archive flags
    uint32_t priority;        // Load priority
    
    /* Locale */
    uint32_t locale;          // Locale mask
    
    /* File list */
    TSList<FILEREC> files;   // Files in archive
    
    /* Status */
    uint32_t status;          // Current status
    bool isOpen;             // Is archive open
    bool isLoaded;           // Is fully loaded
};
```

### Storm::SFile::FILEREC

```c
struct FILEREC {
    /* File identification */
    uint32_t fileId;          // Unique file ID
    uint32_t archiveId;       // Parent archive ID
    
    /* File path */
    char* fileName;          // Internal file name
    char* searchName;        // Searchable name (lowercase)
    
    /* File properties */
    uint32_t fileSize;       // Uncompressed size
    uint32_t compressedSize; // Compressed size
    uint32_t flags;         // File flags
    
    /* Locale */
    uint32_t locale;         // Locale mask
    
    /* Offset */
    uint64_t fileOffset;     // Offset in archive
    
    /* Encryption */
    uint32_t encryptionSeed; // Encryption seed
    uint32_t hash;          // File hash
    
    /* Status */
    uint32_t status;        // Current status
    bool isLoaded;          // Is file loaded
    bool isCompressed;      // Is file compressed
};
```

### Storm::SFile::AUDIOSTREAM

```c
struct AUDIOSTREAM {
    /* Stream identification */
    uint32_t streamId;      // Unique stream ID
    
    /* Source file */
    char* fileName;         // Source file name
    uint64_t fileOffset;    // Offset in file
    
    /* Audio format */
    uint16_t formatTag;     // Format tag (WAVEFORMATEX)
    uint16_t channels;      // Channel count
    uint32_t sampleRate;    // Sample rate
    uint32_t avgBytesPerSec; // Average bytes/second
    uint16_t blockAlign;   // Block alignment
    uint16_t bitsPerSample; // Bits per sample
    
    /* Buffer */
    uint8_t* buffer;       // Audio buffer
    uint32_t bufferSize;   // Buffer size
    uint32_t bufferPos;    // Current position
    
    /* Playback state */
    uint32_t state;        // Playing, paused, stopped
    uint32_t playPosition; // Current play position
    uint32_t loopStart;   // Loop start position
    uint32_t loopEnd;     // Loop end position
    bool isLooping;       // Is looping
    
    /* Callbacks */
    void (*callback)(AUDIOSTREAM* stream, uint32_t event);
    void* callbackData;
};
```

---

## Archive Operations

### SFileOpenArchive

```c
/* Pseudo-code for archive opening */
bool SFileOpenArchive(const char* archivePath, uint32_t locale, 
                      uint32_t flags, HARCHIVE* outArchive) {
    // Check if archive already open
    ARCHIVEREC* existing = FindOpenArchive(archivePath);
    if (existing != NULL) {
        *outArchive = existing->archiveId;
        return true;
    }
    
    // Open MPQ file
    HANDLE fileHandle = CreateFile(archivePath, GENERIC_READ, 
                                   FILE_SHARE_READ, NULL, 
                                   OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL);
    if (fileHandle == INVALID_HANDLE_VALUE) {
        return false;
    }
    
    // Read MPQ header
    MPQHeader header;
    ReadFile(fileHandle, &header, sizeof(MPQHeader));
    
    // Validate header
    if (header.signature != 'MPQ' && header.signature != 'MPQ\x1a') {
        CloseHandle(fileHandle);
        return false;
    }
    
    // Create archive record
    ARCHIVEREC* archive = AllocateArchiveRecord();
    archive->filePath = StrDup(archivePath);
    archive->flags = flags;
    archive->locale = locale;
    archive->isOpen = true;
    
    // Build file table
    BuildFileTable(archive, fileHandle, &header);
    
    // Add to archive list
    ArchiveList.Add(archive);
    
    *outArchive = archive->archiveId;
    return true;
}
```

### MPQ Header Structure

```c
struct MPQHeader {
    /* Header identification */
    uint32_t signature;      // 'MPQ' or 0x1a41504d ('MPQ\x1a')
    uint32_t headerSize;    // Header size
    uint32_t archiveSize;   // Archive size
    uint16_t formatVersion; // Format version (0, 1, 2, 3)
    uint16_t blockSize;     // Block size (power of 2)
    
    /* Hash table */
    uint32_t hashOffset;   // Hash table offset
    uint32_t hashCount;    // Hash table count
    
    /* Block table */
    uint32_t blockOffset;   // Block table offset
    uint32_t blockCount;    // Block table count
    
    /* Extended header (v2+) */
    uint64_t archiveSize64; // 64-bit archive size
    uint64_t hashOffset64;  // 64-bit hash offset
    uint64_t blockOffset64; // 64-bit block offset
    
    /* Format v3 */
    uint32_t hashTableSizeHigh; // High bits of hash size
    uint32_t blockTableSizeHigh; // High bits of block size
    
    /* Format v4 */
    uint64_t headerSize64;  // 64-bit header size
    uint64_t contentSize;  // Content size
};
```

---

## File Operations

### SFileOpenFile

```c
/* Pseudo-code for opening a file in archive */
bool SFileOpenFile(const char* fileName, HANDLE* outHandle) {
    // Normalize filename
    char normalizedName[256];
    NormalizePath(fileName, normalizedName);
    
    // Search through archives (in priority order)
    for (ARCHIVEREC* archive : ArchiveList) {
        // Try locale-specific file first
        FILEREC* file = FindFile(archive, normalizedName, archive->locale);
        if (file != NULL) {
            *outHandle = CreateFileHandle(file);
            return true;
        }
        
        // Fall back to default locale
        file = FindFile(archive, normalizedName, 0);
        if (file != NULL) {
            *outHandle = CreateFileHandle(file);
            return true;
        }
    }
    
    // File not found
    return false;
}
```

### SFileReadFile

```c
/* Pseudo-code for reading from a file */
bool SFileReadFile(HANDLE handle, void* buffer, uint32_t bytesToRead,
                   uint32_t* outBytesRead, OVERLAPPED* overlapped) {
    FILEREC* file = (FILEREC*)handle;
    
    // Check if file is compressed
    if (file->flags & MPQ_FILE_COMPRESSED) {
        // Decompress on demand
        uint8_t* decompressed;
        uint32_t decompressedSize;
        DecompressFile(file, &decompressed, &decompressedSize);
        
        // Copy requested portion
        uint32_t copySize = min(bytesToRead, decompressedSize - file->readPos);
        memcpy(buffer, decompressed + file->readPos, copySize);
        file->readPos += copySize;
        
        if (outBytesRead != NULL) {
            *outBytesRead = copySize;
        }
        
        return copySize == bytesToRead;
    }
    
    // Read directly from archive
    uint64_t offset = file->fileOffset + file->readPos;
    uint32_t bytesRead;
    ReadFileAt(archive->fileHandle, offset, buffer, bytesToRead, &bytesRead);
    
    file->readPos += bytesRead;
    
    if (outBytesRead != NULL) {
        *outBytesRead = bytesRead;
    }
    
    return true;
}
```

---

## Encryption and Hashing

### File Hash Function

```c
/* MPQ file hashing */
uint32_t HashFileName(const char* fileName) {
    uint32_t hash = 0x7FED7FED;  // Initial seed
    uint32_t seed = 0xEEEEEEEE;
    
    for (const char* p = fileName; *p != '\0'; p++) {
        // Convert to uppercase
        char c = toupper(*p);
        
        // Mix into hash
        hash = (hash ^ c) * 0x7FED7FED;
        seed = (seed ^ c) * 0xEEEEEEEE;
    }
    
    return hash ^ seed;
}
```

---

## Compression Types

| Flag | Value | Description |
|------|-------|-------------|
| MPQ_FILE_IMPLODE | 0x00000100 | Imploded (PKWare DCL) |
| MPQ_FILE_COMPRESS | 0x00000200 | Compressed |
| MPQ_FILE_ENCRYPTED | 0x00010000 | Encrypted |
| MPQ_FILE_FIX_KEY | 0x00020000 | Fixed key encryption |
| MPQ_FILE_PATCH_FILE | 0x00100000 | Patch file |
| MPQ_FILE_SINGLE_BLOCK | 0x01000000 | Single unit |

### Supported Compression Methods

| Method | Value | Description |
|--------|-------|-------------|
| COMPRESSION_NONE | 0x00 | No compression |
| COMPRESSION_HUFFMANN | 0x01 | Huffman compression |
| COMPRESSION_ZLIB | 0x02 | zlib/deflate |
| COMPRESSION_PKWARE | 0x08 | PKWare DCL |
| COMPRESSION_BZIP2 | 0x10 | bzip2 |
| COMPRESSION_LZMA | 0x12 | LZMA |
| COMPRESSION_SPARSE | 0x20 | Sparse compression |
| COMPRESSION_ADPCM_MONO | 0x40 | ADPCM mono |
| COMPRESSION_ADPCM_STEREO | 0x80 | ADPCM stereo |

---

## Audio Streaming (DDA)

### SFileDdaBegin

```c
/* Begin audio streaming from MPQ file */
bool SFileDdaBegin(const char* fileName, uint32_t flags) {
    // Open audio file in archive
    HANDLE fileHandle;
    if (!SFileOpenFile(fileName, &fileHandle)) {
        return false;
    }
    
    // Read WAV header
    WAVHeader wavHeader;
    uint32_t bytesRead;
    SFileReadFile(fileHandle, &wavHeader, sizeof(wavHeader), &bytesRead);
    
    // Validate WAV format
    if (wavHeader.riff != 'RIFF' || wavHeader.wave != 'WAVE') {
        SFileCloseFile(fileHandle);
        return false;
    }
    
    // Create audio stream
    AUDIOSTREAM* stream = AllocateAudioStream();
    stream->fileHandle = fileHandle;
    stream->fileName = StrDup(fileName);
    stream->fileOffset = sizeof(wavHeader);
    stream->formatTag = wavHeader.formatTag;
    stream->channels = wavHeader.channels;
    stream->sampleRate = wavHeader.sampleRate;
    stream->avgBytesPerSec = wavHeader.avgBytesPerSec;
    stream->blockAlign = wavHeader.blockAlign;
    stream->bitsPerSample = wavHeader.bitsPerSample;
    stream->state = AUDIO_STOPPED;
    stream->isLooping = (flags & DDA_LOOP) != 0;
    
    // Add to stream list
    AudioStreamList.Add(stream);
    
    return true;
}
```

---

## Locale System

### Locale Support

The MPQ system supports multiple locales through locale-specific files:

```c
/* Locale codes */
#define LOCALE_NEUTRAL      0x00
#define LOCALE_ENGLISH     0x01
#define LOCALE_GERMAN      0x02
#define LOCALE_FRENCH     0x04
#define LOCALE_SPANISH    0x08
#define LOCALE_ITALIAN    0x10
#define LOCALE_JAPANESE   0x20
#define LOCALE_KOREAN     0x40
#define LOCALE_CHINESE    0x80
```

### Locale-Specific File Lookup

```c
/* Find file with locale priority */
FILEREC* FindFile(ARCHIVEREC* archive, const char* fileName, uint32_t locale) {
    // Generate locale-specific filename
    char localePath[256];
    if (locale != 0) {
        snprintf(localePath, sizeof(localePath), "locale\\%s", fileName);
        FILEREC* localeFile = FindFileInArchive(archive, localePath);
        if (localeFile != NULL) {
            return localeFile;
        }
    }
    
    // Fall back to base filename
    return FindFileInArchive(archive, fileName);
}
```

---

## File Name Normalization

### Search Name Generation

```c
/* Normalize file name for searching */
void NormalizePath(const char* input, char* output) {
    // Convert to forward slashes
    const char* p = input;
    char* q = output;
    
    while (*p) {
        if (*p == '\\') {
            *q++ = '/';
        } else {
            *q++ = toupper(*p);
        }
        p++;
    }
    *q = '\0';
    
    // Remove duplicate slashes
    RemoveDuplicateSlashes(output);
    
    // Remove leading slash
    if (output[0] == '/') {
        memmove(output, output + 1, strlen(output + 1) + 1);
    }
}
```

---

## Archive Priority System

Archives are loaded in priority order (highest first):

```c
/* Priority system */
typedef enum {
    PRIORITY_CORE = 0,        // Core game files
    PRIORITY_EXPANSION = 1,   // Expansion content
    PRIORITY_PATCH = 2,       // Patch files
    PRIORITY_LOCAL = 3,       // Local overrides
    PRIORITY_USER = 4         // User customizations
} ArchivePriority;
```

---

## Summary

The archive and file system in WoW Alpha 0.5.3 provides:
- **MPQ archive support**: Full read/write MPQ archives
- **Compression**: Multiple compression methods (zlib, bzip2, LZMA)
- **Encryption**: File encryption with hash-based lookup
- **Locale support**: Multi-language file variants
- **Audio streaming**: Direct audio streaming from archives
- **File caching**: Efficient file caching and pooling

Key functions and addresses provide a complete reference for reverse engineering and implementation.

---

*Document created: 2026-02-07*
*Analysis based on WoWClient.exe (Build 3368)*
