# Task 1: Storm/MPQ Function List

This document lists the Storm and MPQ related functions identified in the binary.

## SFile Functions (File & Archive Operations)

These functions handle MPQ archive management and file I/O.

| Address | Function Name | Description |
|---|---|---|
| 0063c5e0 | `SFile` | Constructor? |
| 0063c630 | `~SFile` | Destructor? |
| 0063dfa0 | `DisableSFileCheckDisk` | |
| 0064ecc0 | `_SFileAuthenticateArchive@8` | |
| 0064f320 | `_SFileAuthenticateArchiveEx@24` | |
| 0064f520 | `_SFileCalcFileCrc@4` | |
| 0064f690 | `_SFileCancelRequest@4` | |
| 0064f770 | `_SFileCancelRequestEx@4` | |
| 0064f7b0 | `_SFileCloseArchive@4` | |
| 0064f8d0 | `_SFileCloseFile@4` | |
| 0064f960 | `_SFileDdaBegin@12` | Direct Data Access? |
| 0064f990 | `_SFileDdaBeginEx@28` | |
| 0064ff00 | `_SFileDdaDestroy@0` | |
| 0064ffd0 | `_SFileDdaEnd@4` | |
| 006500b0 | `_SFileDdaGetPos@12` | |
| 00650180 | `_SFileDdaGetVolume@12` | |
| 00650290 | `_SFileDdaInitialize@4` | |
| 006502f0 | `_SFileDdaSetVolume@12` | |
| 006503e0 | `_SFileDestroy@0` | |
| 00650620 | `_SFileEnableArchive@8` | |
| 00650680 | `_SFileEnableDirectAccess@4` | |
| 006506b0 | `_SFileEnableSeekOptimization@4` | |
| 006506e0 | `_SFileFileExists@4` | |
| 00650770 | `_SFileFileExistsEx@12` | |
| 00651010 | `_SFileGetActualFileName@12` | |
| 00650c70 | `_SFileGetArchiveInfo@12` | |
| 00650d10 | `_SFileGetArchiveName@12` | |
| 00650da0 | `_SFileGetBasePath@8` | |
| 00650e00 | `_SFileGetFileArchive@8` | |
| 006510c0 | `_SFileGetFileCompressedSize@8` | |
| 00650e90 | `_SFileGetFileCrc@4` | |
| 00650f00 | `_SFileGetFileMD5@8` | |
| 00650f80 | `_SFileGetFileName@12` | |
| 00651120 | `_SFileGetFileSize@8` | |
| 006511c0 | `_SFileGetFileTime@8` | |
| 00652ec0 | `_SFileGetLocale@0` | |
| 00652f70 | `_SFileLoadDump@0` | |
| 00651290 | `_SFileLoadFile@20` | |
| 006512f0 | `_SFileLoadFileEx2@32` | |
| 006512c0 | `_SFileLoadFileEx@28` | |
| 00651420 | `_SFileOpenArchive@16` | |
| 00651dd0 | `_SFileOpenFile@8` | |
| 00651df0 | `_SFileOpenFileAsArchive@20` | |
| 00652030 | `_SFileOpenFileEx@16` | |
| 00651710 | `_SFileOpenPathAsArchive@20` | |
| 006523b0 | `_SFilePrioritizeRequest@8` | |
| 00652460 | `_SFileReadFile@20` | |
| 006524c0 | `_SFileReadFileEx2@28` | |
| 00652490 | `_SFileReadFileEx@24` | |
| 0064f920 | `_SFileRegisterLoadNotifyProc@8` | |
| 00652b30 | `_SFileSetAsyncBudget@4` | |
| 00652b60 | `_SFileSetBasePath@4` | |
| 00652c60 | `_SFileSetDataChunkSize@4` | |
| 00652cc0 | `_SFileSetFilePointer@16` | |
| 00652e50 | `_SFileSetIoErrorMode@8` | |
| 00652e90 | `_SFileSetLocale@4` | |
| 00652ef0 | `_SFileSetPlatform@4` | |
| 00652f20 | `_SFileUnloadFile@4` | |
| 00565070 | `SFileReadTyped` | |
| 00565080 | `SFileReadTyped` | |
| 00589870 | `SFileReadTyped` | |

## SComp Functions (Compression Dispatch)

These functions likely handle the dispatching to specific compression algorithms.

| Address | Function Name | Description |
|---|---|---|
| 006497e0 | `_SCompCompress@28` | |
| 00649b70 | `_SCompDecompress@16` | |
| 00649b90 | `SCompDecompress2` | |
| 00649d80 | `_SCompDestroy@0` | |

## Storm Functions (System & Initialization)

| Address | Function Name | Description |
|---|---|---|
| 00637e30 | `AddStormFacility` | |
| 00637e10 | `AddStormMessages` | |
| 0045cc20 | `OsGetStormName` | |
| 0064d1f0 | `StormOptCdThread` | |
| 00634880 | `StormRtlDestroy` | |
| 006348a0 | `StormRtlInitialize` | |
| 0065abc0 | `_StormCallService` | |
| 0063bc90 | `_StormDestroy@0` | |
| 0063bcc0 | `_StormGetInstance@0` | |
| 0063bd10 | `_StormGetOption@12` | |
| 0063bc80 | `_StormInitialize@0` | |
| 0063beb0 | `_StormSetOption@12` | |
| 00401000 | `_StormStaticEntryPoint` | |

## Compression Algorithms

### Huffman
| Address | Function Name | Description |
|---|---|---|
| 00648390 | `CHuffman` | |
| 00648760 | `CHuffmanDecoder` | |
| 006496d0 | `CHuffmanEncoder` | |
| 00649670 | `HuffmanCompress` | |
| 00649730 | `HuffmanDecompress` | |
| 00648a50 | `?Compress@CHuffmanEncoder...` | |
| 00648790 | `?Decompress@CHuffmanDecoder...` | |

### Pkware (DCL/Implode)
| Address | Function Name | Description |
|---|---|---|
| 00648c00 | `PkwareCompress` | |
| 00648d50 | `PkwareDecompress` | |
| 00648cc0 | `PkwareBufferRead` | |
| 00648d10 | `PkwareBufferWrite` | |
| 00657f70 | `implode` | |
| 00658a50 | `explode` | |

### Zlib
| Address | Function Name | Description |
|---|---|---|
| 00648f40 | `ZlibCompress` | |
| 00649060 | `ZlibDecompress` | |
| 00648dd0 | `zlib_compress` | |
| 00648fa0 | `zlib_uncompress` | |

### IMA ADPCM (Audio)
| Address | Function Name | Description |
|---|---|---|
| 00649140 | `ImaAdpcmCompress` | |
| 00649400 | `ImaAdpcmDecompress` | |
| 006490b0 | `ImaAdpcmMonoCompress` | |
| 006493e0 | `ImaAdpcmMonoDecompress` | |
| 00649610 | `ImaAdpcmStereoCompress` | |
| 00649650 | `ImaAdpcmStereoDecompress` | |

### LZW
| Address | Function Name | Description |
|---|---|---|
| 0064dfa0 | `DecompressLzw` | |
| 0064e030 | `DecompressLzw_BufferRead` | |
| 0064e070 | `DecompressLzw_BufferWrite` | |

### Other
| Address | Function Name | Description |
|---|---|---|
| 00649a20 | `s_AllocDecompressBuffer` | |
| 00649b20 | `s_FreeDecompressBuffer` | |
