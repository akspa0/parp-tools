using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using GillijimProject.WowFiles.Alpha;
using WoWRollback.LkToAlphaModule.Liquids;
using WoWRollback.LkToAlphaModule.Models;

namespace WoWRollback.LkToAlphaModule.Services;

/// <summary>
/// Extracts raw chunk data from Alpha ADT files to populate LkMcnkSource for round-trip testing.
/// </summary>
public static class AlphaDataExtractor
{
    private const int ChunkHeaderSize = 8; // FourCC (4) + Size (4)
    private const int McnkHeaderSize = 128;
    
    /// <summary>
    /// Reads an Alpha MCNK chunk and extracts raw data to populate LkMcnkSource.
    /// This enables round-trip testing: Alpha → LK → Alpha.
    /// </summary>
    public static LkMcnkSource ExtractFromAlphaMcnk(string alphaAdtPath, int mcnkOffset, int mcnkIndex)
    {
        if (string.IsNullOrWhiteSpace(alphaAdtPath))
            throw new ArgumentException("Alpha ADT path required", nameof(alphaAdtPath));
        if (!File.Exists(alphaAdtPath))
            throw new FileNotFoundException("Alpha ADT not found", alphaAdtPath);

        using var fs = File.OpenRead(alphaAdtPath);
        
        fs.Seek(mcnkOffset, SeekOrigin.Begin);
        Span<byte> chunkHeader = stackalloc byte[ChunkHeaderSize];
        if (fs.Read(chunkHeader) != ChunkHeaderSize)
        {
            throw new InvalidDataException($"[{Path.GetFileName(alphaAdtPath)}] MCNK {mcnkIndex}: unable to read chunk header at offset 0x{mcnkOffset:X}.");
        }

        string fourCC = Encoding.ASCII.GetString(chunkHeader[..4]);
        if (!string.Equals(fourCC, "KNCM", StringComparison.Ordinal))
        {
            throw new InvalidDataException($"[{Path.GetFileName(alphaAdtPath)}] MCNK {mcnkIndex}: expected 'KNCM' FourCC at offset 0x{mcnkOffset:X}, found '{fourCC}'.");
        }

        int payloadSize = BitConverter.ToInt32(chunkHeader[4..8]);
        if (payloadSize < McnkHeaderSize)
        {
            throw new InvalidDataException($"[{Path.GetFileName(alphaAdtPath)}] MCNK {mcnkIndex}: invalid payload size {payloadSize} (< {McnkHeaderSize}).");
        }

        long headerStart = mcnkOffset + ChunkHeaderSize;
        long dataStart = headerStart + McnkHeaderSize;
        long chunkEnd = headerStart + payloadSize;
        int chunkDataLength = payloadSize - McnkHeaderSize;
        if (chunkEnd > fs.Length)
        {
            throw new InvalidDataException($"[{Path.GetFileName(alphaAdtPath)}] MCNK {mcnkIndex}: chunk extends beyond file length (end=0x{chunkEnd:X}, len=0x{fs.Length:X}).");
        }

        fs.Seek(headerStart, SeekOrigin.Begin);
        var headerBytes = new byte[McnkHeaderSize];
        fs.Read(headerBytes, 0, McnkHeaderSize);
        
        // Parse header fields we need
        int indexX = BitConverter.ToInt32(headerBytes, 0x04);
        int indexY = BitConverter.ToInt32(headerBytes, 0x08);
        int areaId = BitConverter.ToInt32(headerBytes, 0x38); // Unknown3 used as AreaId
        
        // Read offsets from header (relative to start of MCNK chunk)
        int mcvtOffset = BitConverter.ToInt32(headerBytes, 0x18);
        int mcnrOffset = BitConverter.ToInt32(headerBytes, 0x1C);
        int mclyOffset = BitConverter.ToInt32(headerBytes, 0x20);
        int mcrfOffset = BitConverter.ToInt32(headerBytes, 0x24);
        int mcalOffset = BitConverter.ToInt32(headerBytes, 0x28);
        int mcalSize = BitConverter.ToInt32(headerBytes, 0x2C);
        int mcshOffset = BitConverter.ToInt32(headerBytes, 0x30);
        int mcshSize = BitConverter.ToInt32(headerBytes, 0x34);
        int nLayers = BitConverter.ToInt32(headerBytes, 0x10);
        
        // Only log if there's actual texture data (nLayers > 1 or mcalSize > 0)
        if (nLayers > 1 || mcalSize > 0)
        {
            Console.WriteLine($"[AlphaExtract] MCNK {indexX},{indexY}: nLayers={nLayers}, mclyOffset=0x{mclyOffset:X}, mcalOffset=0x{mcalOffset:X}, mcalSize={mcalSize}");
        }
        
        uint mcnkFlags = BitConverter.ToUInt32(headerBytes, 0x00);
        int doodadRefs = BitConverter.ToInt32(headerBytes, 0x14);
        float radius = BitConverter.ToSingle(headerBytes, 0x0C);
        int mapObjRefs = BitConverter.ToInt32(headerBytes, 0x3C);
        ushort holes = BitConverter.ToUInt16(headerBytes, 0x40);
        uint offsSndEmit = BitConverter.ToUInt32(headerBytes, 0x5C);
        uint sndEmitCount = BitConverter.ToUInt32(headerBytes, 0x60);
        uint offsLiquid = BitConverter.ToUInt32(headerBytes, 0x64);

        var source = new LkMcnkSource
        {
            IndexX = indexX,
            IndexY = indexY,
            AreaId = (uint)areaId,
            Flags = mcnkFlags,
            HolesLowRes = holes,
            Radius = radius,
            DoodadRefCount = (uint)Math.Max(doodadRefs, 0),
            MapObjectRefs = (uint)Math.Max(mapObjRefs, 0),
            OffsLiquid = offsLiquid,
            OffsSndEmitters = offsSndEmit,
            SndEmitterCount = sndEmitCount
        };

        // Helper to compute size from current offset to next populated offset (or chunk end)
        int ComputeSize(int currentOffset, params int[] potentialNextOffsets)
        {
            if (currentOffset < 0)
            {
                return 0;
            }

            if (currentOffset > chunkDataLength)
            {
                return 0;
            }

            int next = chunkDataLength;
            foreach (int candidate in potentialNextOffsets)
            {
                if (candidate > currentOffset && candidate < next)
                {
                    next = candidate;
                }
            }

            int size = next - currentOffset;
            if (size < 0)
            {
                throw new InvalidDataException($"[{Path.GetFileName(alphaAdtPath)}] MCNK {mcnkIndex}: computed negative size for offset {currentOffset} (next={next}).");
            }

            return size;
        }

        // Helper local to load raw slices (supports both headerless and headered layouts)
        ReadOnlySpan<byte> ReadRawSlice(int relativeOffset, int declaredSize, string label)
        {
            if (relativeOffset < 0)
            {
                return ReadOnlySpan<byte>.Empty;
            }

            long sliceStart = dataStart + relativeOffset;
            if (sliceStart < dataStart || sliceStart > chunkEnd)
            {
                throw new InvalidDataException($"[{Path.GetFileName(alphaAdtPath)}] MCNK {mcnkIndex}: {label} offset 0x{relativeOffset:X} points outside chunk (start=0x{sliceStart:X}, dataStart=0x{dataStart:X}, end=0x{chunkEnd:X}).");
            }

            // If declaredSize is zero (Alpha headerless), infer from next chunk or chunk end
            int size = declaredSize;
            if (size <= 0)
            {
                size = (int)Math.Min(chunkEnd - sliceStart, int.MaxValue);
            }

            long sliceEnd = sliceStart + size;
            if (sliceEnd > chunkEnd)
            {
                throw new InvalidDataException($"[{Path.GetFileName(alphaAdtPath)}] MCNK {mcnkIndex}: {label} overflows chunk (end=0x{sliceEnd:X} > 0x{chunkEnd:X}).");
            }

            var buffer = new byte[size];
            fs.Seek(sliceStart, SeekOrigin.Begin);
            fs.Read(buffer, 0, size);

            // Handle embedded LK-style chunk headers if present (e.g., 'MCLY', 'MCAL')
            if (size >= ChunkHeaderSize)
            {
                string embeddedFourCc = Encoding.ASCII.GetString(buffer, 0, 4);
                if ((embeddedFourCc == "KNCM" || embeddedFourCc == "YLCM" || embeddedFourCc == "LACM" || embeddedFourCc == "HSCM" || embeddedFourCc == "ESCM")
                    && BitConverter.ToInt32(buffer, 4) >= 0
                    && BitConverter.ToInt32(buffer, 4) <= size - ChunkHeaderSize)
                {
                    int embeddedSize = BitConverter.ToInt32(buffer, 4);
                    if (embeddedFourCc == "KNCM")
                    {
                        throw new InvalidDataException($"[{Path.GetFileName(alphaAdtPath)}] MCNK {mcnkIndex}: nested MCNK detected inside {label}.");
                    }

                    int dataLength = embeddedSize;
                    var trimmed = new byte[dataLength];
                    Buffer.BlockCopy(buffer, ChunkHeaderSize, trimmed, 0, dataLength);
                    return trimmed;
                }
            }

            return buffer;
        }

        // Extract MCVT raw data (580 bytes, no chunk header in Alpha)
        if (mcvtOffset >= 0)
        {
            var mcvt = ReadRawSlice(mcvtOffset, 580, "MCVT");
            if (mcvt.Length != 580)
            {
                Console.WriteLine($"[AlphaExtract] Warning: MCNK {indexX},{indexY} MCVT expected 580 bytes but read {mcvt.Length}.");
            }
            source.McvtRaw = mcvt.ToArray();
        }

        // Extract MCNR raw data (448 bytes, no chunk header in Alpha)
        if (mcnrOffset >= 0)
        {
            var mcnr = ReadRawSlice(mcnrOffset, 448, "MCNR");
            if (mcnr.Length != 448)
            {
                Console.WriteLine($"[AlphaExtract] Warning: MCNK {indexX},{indexY} MCNR expected 448 bytes but read {mcnr.Length}.");
            }
            source.McnrRaw = mcnr.ToArray();
        }

        // Extract MCLY table (Alpha MCLY HAS chunk header on disk)
        // On disk: "YLCM" (reversed) + size + data
        // Reference: McnkAlpha.cs line 54-55 uses new Chunk(adtFile, offsetInFile)
        if (mclyOffset >= 0 && nLayers > 0)
        {
            long mclyAbsoluteOffset = dataStart + mclyOffset;
            
            try
            {
                // Chunk constructor will seek to offset and read header + data
                var mclyChunk = new GillijimProject.WowFiles.Chunk(fs, (int)mclyAbsoluteOffset);
                
                // After reading, Letters will be "MCLY" (Chunk constructor reverses FourCC)
                if (mclyChunk.Letters == "MCLY")
                {
                    source.MclyRaw = mclyChunk.Data;
                    
                    int expectedSize = nLayers * 16;
                    if (source.MclyRaw.Length != expectedSize)
                    {
                        Console.WriteLine($"[AlphaExtract] Warning: MCNK {indexX},{indexY} expected MCLY data size {expectedSize}, read {source.MclyRaw.Length}.");
                    }
                    
                }
                else
                {
                    Console.WriteLine($"[AlphaExtract] Warning: MCNK {indexX},{indexY} expected MCLY chunk at offset 0x{mclyOffset:X}, found '{mclyChunk.Letters}' size={mclyChunk.Data.Length}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[AlphaExtract] Error reading MCLY for MCNK {indexX},{indexY} at offset 0x{mclyOffset:X}: {ex.Message}");
            }
        }
        else if (nLayers > 1)
        {
            Console.WriteLine($"[AlphaExtract] Warning: MCNK {indexX},{indexY} has nLayers={nLayers} but mclyOffset={mclyOffset} (invalid)");
        }

        // Extract MCRF table (Alpha MCRF HAS chunk header on disk)
        // On disk: "FRCM" (reversed) + size + data
        // Reference: McnkAlpha.cs line 57-59 uses new Mcrf(adtFile, offsetInFile)
        if (mcrfOffset >= 0)
        {
            long mcrfAbsoluteOffset = dataStart + mcrfOffset;
            
            try
            {
                // Mcrf inherits from Chunk, so it reads header + data
                var mcrfChunk = new GillijimProject.WowFiles.Mcrf(fs, (int)mcrfAbsoluteOffset);
                source.McrfRaw = mcrfChunk.Data;
                
                if (source.McrfRaw.Length > 0)
                {
                    Console.WriteLine($"[AlphaExtract] MCRF read {source.McrfRaw.Length} bytes for MCNK {indexX},{indexY}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[AlphaExtract] Error reading MCRF for MCNK {indexX},{indexY}: {ex.Message}");
            }
        }

        // Extract MCSH data
        if (mcshOffset >= 0 && mcshSize > 0)
        {
            var mcsh = ReadRawSlice(mcshOffset, mcshSize, "MCSH");
            source.McshRaw = mcsh.ToArray();
        }

        // Extract MCAL chunk
        // Alpha MCAL: raw bytes (NO chunk header), size from MCNK header mcalSize field
        // MCAL stores compressed/partial alpha data per layer based on MCLY props
        if (mcalOffset >= 0 && mcalSize > 0)
        {
            long mcalAbsoluteOffset = dataStart + mcalOffset;
            
            // Clamp read size to chunk bounds
            int readSize = mcalSize;
            long mcalEnd = mcalAbsoluteOffset + readSize;
            if (mcalEnd > chunkEnd)
            {
                readSize = (int)(chunkEnd - mcalAbsoluteOffset);
                Console.WriteLine($"[AlphaExtract] Warning: MCNK {indexX},{indexY} MCAL size clamped from {mcalSize} to {readSize} (chunk boundary)");
            }
            
            if (readSize > 0)
            {
                fs.Seek(mcalAbsoluteOffset, SeekOrigin.Begin);
                var mcalArray = new byte[readSize];
                fs.Read(mcalArray, 0, readSize);
                
                // Preserve raw MCAL for passthrough
                source.McalRaw = mcalArray;

                // Only log non-zero MCAL data
                if (mcalArray.Length > 0 && mcalArray.Any(b => b != 0))
                {
                    Console.WriteLine($"[AlphaExtract] MCAL raw[0..31] MCNK {indexX},{indexY}: {BitConverter.ToString(mcalArray, 0, Math.Min(32, mcalArray.Length))}");
                }
            }

            int layerCount = Math.Max(0, source.MclyRaw.Length / 16);
            source.AlphaLayers.Clear();

            // In Alpha, MCAL contains raw compressed alpha data for ALL layers that use alpha
            // We need to parse MCLY to determine which layers use alpha and their offsets
            if (source.McalRaw.Length > 0 && source.MclyRaw.Length > 0)
            {
                for (int layerIdx = 0; layerIdx < layerCount; layerIdx++)
                {
                    int mclyEntryOffset = layerIdx * 16;
                    if (mclyEntryOffset + 16 > source.MclyRaw.Length) break;

                    uint props = BitConverter.ToUInt32(source.MclyRaw, mclyEntryOffset + 4);
                    uint alphaOffset = BitConverter.ToUInt32(source.MclyRaw, mclyEntryOffset + 8);
                    
                    // In Alpha MCLY: props field bit 0x4 = use_alpha_map (set for layers after first)
                    // Layer 0 (base) never has alpha
                    bool usesAlpha = layerIdx > 0 && (props & 0x4) != 0;
                    
                    if (usesAlpha && alphaOffset < source.McalRaw.Length)
                    {
                        // Determine size: from this offset to next layer's offset (or end of MCAL)
                        int nextOffset = source.McalRaw.Length;
                        for (int nextIdx = layerIdx + 1; nextIdx < layerCount; nextIdx++)
                        {
                            int nextMclyOffset = nextIdx * 16;
                            if (nextMclyOffset + 12 <= source.MclyRaw.Length)
                            {
                                uint nextProps = BitConverter.ToUInt32(source.MclyRaw, nextMclyOffset + 4);
                                if (nextIdx > 0 && (nextProps & 0x4) != 0)
                                {
                                    uint nextAlphaOffset = BitConverter.ToUInt32(source.MclyRaw, nextMclyOffset + 8);
                                    if (nextAlphaOffset > alphaOffset)
                                    {
                                        nextOffset = (int)nextAlphaOffset;
                                        break;
                                    }
                                }
                            }
                        }
                        
                        int layerSize = Math.Min(nextOffset - (int)alphaOffset, source.McalRaw.Length - (int)alphaOffset);
                        if (layerSize > 0)
                        {
                            var layerData = new byte[layerSize];
                            Buffer.BlockCopy(source.McalRaw, (int)alphaOffset, layerData, 0, layerSize);
                            
                            source.AlphaLayers.Add(new LkMcnkAlphaLayer
                            {
                                LayerIndex = layerIdx,
                                ColumnMajorAlpha = layerData,
                                OverrideFlags = props
                            });
                        }
                    }
                }
            }
        }
        else
        {
            source.AlphaLayers.Clear();
        }

        // MCSE not commonly used in Alpha, leave empty
        source.McseRaw = Array.Empty<byte>();

        if (source.MclyRaw.Length == 0 && source.AlphaLayers.Count > 0)
        {
            Console.WriteLine($"[AlphaExtract] Warning: MCNK {indexX},{indexY} produced alpha layers without MCLY entries.");
        }

        return source;
    }

    /// <summary>
    /// Extracts all 256 MCNK chunks from an Alpha ADT file.
    /// </summary>
    public static LkAdtSource ExtractFromAlphaAdt(string alphaAdtPath)
    {
        if (string.IsNullOrWhiteSpace(alphaAdtPath))
            throw new ArgumentException("Alpha ADT path required", nameof(alphaAdtPath));
        if (!File.Exists(alphaAdtPath))
            throw new FileNotFoundException("Alpha ADT not found", alphaAdtPath);

        var source = new LkAdtSource
        {
            MapName = Path.GetFileNameWithoutExtension(alphaAdtPath),
            TileX = 0, // TODO: Parse from filename
            TileY = 0
        };
        
        using var fs = File.OpenRead(alphaAdtPath);
        var allBytes = new byte[fs.Length];
        fs.Read(allBytes, 0, (int)fs.Length);

        ParseTopLevelChunks(alphaAdtPath, allBytes, source);

        // Populate MH2O sources (per MCNK) via Alpha liquid extraction
        var liquidsOptions = new LiquidsOptions();
        var mclqExtractor = new AlphaMclqExtractor(alphaAdtPath);
        var mclqs = mclqExtractor.Extract();
        if (mclqs is not null)
        {
            int count = Math.Min(mclqs.Length, source.Mh2oByChunk.Length);
            for (int i = 0; i < count; i++)
            {
                var mclq = mclqs[i];
                if (mclq is null) continue;
                source.Mh2oByChunk[i] = LiquidsConverter.MclqToMh2o(mclq, liquidsOptions);
            }
        }

        return source;
    }

    private static void ParseTopLevelChunks(string alphaAdtPath, byte[] allBytes, LkAdtSource source)
    {
        var mcnkOffsets = new List<int>();
        int position = 0;
        while (position + 8 <= allBytes.Length)
        {
            string fourCC = Encoding.ASCII.GetString(allBytes, position, 4);
            int size = BitConverter.ToInt32(allBytes, position + 4);
            if (size < 0 || position + 8 + size > allBytes.Length)
            {
                break;
            }

            int dataStart = position + 8;

            switch (fourCC)
            {
                case "XDDM":
                    if (source.MmdxFilenames.Count == 0)
                        ParseStringTable(allBytes, dataStart, size, source.MmdxFilenames);
                    break;
                case "DIMM":
                    if (source.MmidOffsets.Count == 0)
                        ParseOffsetTable(allBytes, dataStart, size, source.MmidOffsets);
                    break;
                case "OMWM":
                    if (source.MwmoFilenames.Count == 0)
                        ParseStringTable(allBytes, dataStart, size, source.MwmoFilenames);
                    break;
                case "DIWM":
                    if (source.MwidOffsets.Count == 0)
                        ParseOffsetTable(allBytes, dataStart, size, source.MwidOffsets);
                    break;
                case "FDDM":
                    ParseMddfEntries(allBytes, dataStart, size, source.MddfPlacements);
                    break;
                case "FDOM":
                    ParseModfEntries(allBytes, dataStart, size, source.ModfPlacements);
                    break;
                case "KNCM":
                    mcnkOffsets.Add(position);
                    break;
            }

            position = AlignPosition(position + 8 + size);
        }

        int index = 0;
        foreach (int mcnkOffset in mcnkOffsets)
        {
            if (index >= 256)
            {
                break;
            }

            var mcnkSource = ExtractFromAlphaMcnk(alphaAdtPath, mcnkOffset, index);
            source.Mcnks.Add(mcnkSource);
            index++;
        }

        while (source.Mcnks.Count < 256)
        {
            int i = source.Mcnks.Count;
            source.Mcnks.Add(new LkMcnkSource
            {
                IndexX = i % 16,
                IndexY = i / 16
            });
        }
    }

    private static void ParseStringTable(byte[] data, int start, int size, List<string> destination)
    {
        destination.Clear();
        int end = start + size;
        int cursor = start;
        while (cursor < end)
        {
            int terminator = cursor;
            while (terminator < end && data[terminator] != 0)
            {
                terminator++;
            }

            int length = terminator - cursor;
            if (length > 0)
            {
                string value = Encoding.UTF8.GetString(data, cursor, length);
                destination.Add(value);
            }

            cursor = terminator + 1;
        }
    }

    private static void ParseOffsetTable(byte[] data, int start, int size, List<int> destination)
    {
        destination.Clear();
        for (int offset = 0; offset + 4 <= size; offset += 4)
        {
            destination.Add(BitConverter.ToInt32(data, start + offset));
        }
    }

    private static void ParseMddfEntries(byte[] data, int start, int size, List<LkMddfPlacement> destination)
    {
        destination.Clear();
        const int entrySize = 36;
        for (int offset = 0; offset + entrySize <= size; offset += entrySize)
        {
            int baseOffset = start + offset;
            int nameIndex = BitConverter.ToInt32(data, baseOffset + 0);
            int uniqueId = BitConverter.ToInt32(data, baseOffset + 4);
            float posX = BitConverter.ToSingle(data, baseOffset + 8);
            float posY = BitConverter.ToSingle(data, baseOffset + 12);
            float posZ = BitConverter.ToSingle(data, baseOffset + 16);
            float rotX = BitConverter.ToSingle(data, baseOffset + 20);
            float rotY = BitConverter.ToSingle(data, baseOffset + 24);
            float rotZ = BitConverter.ToSingle(data, baseOffset + 28);
            ushort scaleRaw = BitConverter.ToUInt16(data, baseOffset + 32);
            ushort flags = BitConverter.ToUInt16(data, baseOffset + 34);

            float scale = scaleRaw / 1024.0f;
            destination.Add(new LkMddfPlacement(nameIndex, uniqueId, posX, posY, posZ, rotX, rotY, rotZ, scale, flags));
        }
    }

    private static void ParseModfEntries(byte[] data, int start, int size, List<LkModfPlacement> destination)
    {
        destination.Clear();
        const int entrySize = 64;
        for (int offset = 0; offset + entrySize <= size; offset += entrySize)
        {
            int baseOffset = start + offset;
            int nameIndex = BitConverter.ToInt32(data, baseOffset + 0);
            int uniqueId = BitConverter.ToInt32(data, baseOffset + 4);
            float posX = BitConverter.ToSingle(data, baseOffset + 8);
            float posY = BitConverter.ToSingle(data, baseOffset + 12);
            float posZ = BitConverter.ToSingle(data, baseOffset + 16);
            float rotX = BitConverter.ToSingle(data, baseOffset + 20);
            float rotY = BitConverter.ToSingle(data, baseOffset + 24);
            float rotZ = BitConverter.ToSingle(data, baseOffset + 28);
            float extentsX = BitConverter.ToSingle(data, baseOffset + 32);
            float extentsY = BitConverter.ToSingle(data, baseOffset + 36);
            float extentsZ = BitConverter.ToSingle(data, baseOffset + 40);
            ushort flags = BitConverter.ToUInt16(data, baseOffset + 44);
            ushort doodadSet = BitConverter.ToUInt16(data, baseOffset + 46);
            ushort nameSet = BitConverter.ToUInt16(data, baseOffset + 48);
            ushort scale = BitConverter.ToUInt16(data, baseOffset + 50);

            destination.Add(new LkModfPlacement(nameIndex, uniqueId, posX, posY, posZ, rotX, rotY, rotZ,
                extentsX, extentsY, extentsZ, flags, doodadSet, nameSet, scale));
        }
    }

    private static int AlignPosition(int position)
    {
        if ((position & 1) != 0)
        {
            position++;
        }
        return position;
    }
}
