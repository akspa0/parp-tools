using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class LkAdtReader
{
    private const int LkMddfEntrySize = 36;
    private const int LkModfEntrySize = 64;
    private const float MapOrigin = 17066.666f;
    private const int RootMcnkHeaderSize = 128;
    private const int RootMcnkMcalSizeOffset = 0x28;
    private const int McnrConsumedSize = 0x1C0;

    public static LkAdtData Read(byte[] adtBytes, byte[]? tex0Bytes, byte[]? obj0Bytes, int tileX, int tileY)
    {
        var rootChunks = new Dictionary<int, LkMcnkData>();
        var textureNames = new List<string>();
        var modelNames = new List<string>();
        var worldModelNames = new List<string>();
        var modelPlacements = new List<LkMddfEntry>();
        var worldModelPlacements = new List<LkModfEntry>();
        uint mhdrFlags = 0;
        int[,,]? mfboFlightBounds = null;

        ParseAdtStream(adtBytes, rootChunks, textureNames, modelNames, worldModelNames, modelPlacements, worldModelPlacements, ref mhdrFlags, ref mfboFlightBounds);

        if (tex0Bytes != null)
            MergeSplitTextureChunks(tex0Bytes, textureNames, rootChunks);

        if (obj0Bytes != null)
            MergeSplitObjectData(obj0Bytes, modelNames, worldModelNames, modelPlacements, worldModelPlacements, rootChunks);

        var chunksList = new List<LkMcnkData>(256);
        for (int i = 0; i < 256; i++)
        {
            if (rootChunks.TryGetValue(i, out var chunk))
                chunksList.Add(chunk);
        }

        return new LkAdtData
        {
            MapName = "",
            TileX = tileX,
            TileY = tileY,
            TextureNames = textureNames,
            ModelNames = modelNames,
            WorldModelNames = worldModelNames,
            ModelPlacements = modelPlacements,
            WorldModelPlacements = worldModelPlacements,
            Chunks = AttachLiquidData(adtBytes, chunksList, tileX, tileY),
            MhdrFlags = mhdrFlags,
            MfboFlightBounds = mfboFlightBounds
        };
    }

    private static void MergeSplitTextureChunks(byte[] tex0Bytes, List<string> textureNames, Dictionary<int, LkMcnkData> rootChunks)
    {
        using var stream = new MemoryStream(tex0Bytes, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "synthetic_tex0.adt");
        if (fileSummary.Kind != MapFileKind.AdtTex)
            return;

        byte[]? mtexData = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mtex);
        if (textureNames.Count == 0 && mtexData is { Length: > 0 })
            textureNames.AddRange(MapSummaryReaderCommon.ReadStringEntries(mtexData));

        int chunkIndex = 0;
        foreach (MapChunkLocation mcnkChunk in fileSummary.Chunks.Where(static chunk => chunk.Id == MapChunkIds.Mcnk))
        {
            if (chunkIndex >= 256)
                break;

            byte[] payload = MapSummaryReaderCommon.ReadChunkPayload(stream, mcnkChunk);
            LkMcnkData splitChunk = ReadSplitTextureChunk(payload, chunkIndex);
            MergeChunks(rootChunks, new Dictionary<int, LkMcnkData>
            {
                [splitChunk.IndexY * 16 + splitChunk.IndexX] = splitChunk,
            });

            chunkIndex++;
        }
    }

    private static void MergeSplitObjectData(
        byte[] obj0Bytes,
        List<string> modelNames,
        List<string> worldModelNames,
        List<LkMddfEntry> modelPlacements,
        List<LkModfEntry> worldModelPlacements,
        Dictionary<int, LkMcnkData> rootChunks)
    {
        using var stream = new MemoryStream(obj0Bytes, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "synthetic_obj0.adt");
        if (fileSummary.Kind != MapFileKind.AdtObj)
            return;

        int modelNameBase = modelNames.Count;
        byte[]? mmdxData = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mmdx);
        if (mmdxData is { Length: > 0 })
            modelNames.AddRange(MapSummaryReaderCommon.ReadStringEntries(mmdxData));

        int worldModelNameBase = worldModelNames.Count;
        byte[]? mwmoData = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mwmo);
        if (mwmoData is { Length: > 0 })
            worldModelNames.AddRange(MapSummaryReaderCommon.ReadStringEntries(mwmoData));

        int modelPlacementBase = modelPlacements.Count;
        byte[]? mddfData = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mddf);
        if (mddfData is { Length: > 0 })
        {
            using var mddfStream = new MemoryStream(mddfData, writable: false);
            using var mddfReader = new BinaryReader(mddfStream, Encoding.ASCII, leaveOpen: true);
            foreach (LkMddfEntry placement in ReadMddfEntries(mddfReader, mddfData.Length))
                modelPlacements.Add(placement with { NameId = placement.NameId + modelNameBase });
        }

        int worldPlacementBase = worldModelPlacements.Count;
        byte[]? modfData = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Modf);
        if (modfData is { Length: > 0 })
        {
            using var modfStream = new MemoryStream(modfData, writable: false);
            using var modfReader = new BinaryReader(modfStream, Encoding.ASCII, leaveOpen: true);
            foreach (LkModfEntry placement in ReadModfEntries(modfReader, modfData.Length))
                worldModelPlacements.Add(placement with { NameId = placement.NameId + worldModelNameBase });
        }

        int chunkIndex = 0;
        foreach (MapChunkLocation mcnkChunk in fileSummary.Chunks.Where(static chunk => chunk.Id == MapChunkIds.Mcnk))
        {
            if (chunkIndex >= 256)
                break;

            byte[] payload = MapSummaryReaderCommon.ReadChunkPayload(stream, mcnkChunk);
            byte[]? mcrdPayload = TryReadSplitMcnkSubchunkPayload(payload, AdtChunkIds.Mcrd);
            byte[]? mcrwPayload = TryReadSplitMcnkSubchunkPayload(payload, AdtChunkIds.Mcrw);
            if (mcrdPayload is null && mcrwPayload is null)
            {
                chunkIndex++;
                continue;
            }

            int chunkX = chunkIndex % 16;
            int chunkY = chunkIndex / 16;
            MergeChunks(rootChunks, new Dictionary<int, LkMcnkData>
            {
                [chunkIndex] = new LkMcnkData
                {
                    IndexX = chunkX,
                    IndexY = chunkY,
                    DoodadRefs = ReadSplitChunkReferences(mcrdPayload, modelPlacementBase),
                    WorldModelRefs = ReadSplitChunkReferences(mcrwPayload, worldPlacementBase),
                },
            });

            chunkIndex++;
        }
    }

    private static IReadOnlyList<int> ReadSplitChunkReferences(byte[]? payload, int baseIndex)
    {
        if (payload is not { Length: >= 4 })
            return [];

        int count = payload.Length / sizeof(int);
        List<int> references = new(count);
        for (int index = 0; index < count; index++)
            references.Add(baseIndex + BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(index * sizeof(int), sizeof(int))));

        return references;
    }

    private static LkMcnkData ReadSplitTextureChunk(byte[] payload, int chunkIndex)
    {
        uint flags = 0;
        int? headerMcalPayloadSize = null;
        int chunkX = chunkIndex % 16;
        int chunkY = chunkIndex / 16;
        int startOffset = 0;
        if (TryReadEmbeddedSplitMcnkHeader(payload, out flags, out headerMcalPayloadSize, out int? headerChunkX, out int? headerChunkY))
        {
            startOffset = RootMcnkHeaderSize;
            chunkX = headerChunkX ?? chunkX;
            chunkY = headerChunkY ?? chunkY;
        }

        List<LkMclyEntry> layers = [];
        byte[]? alphaData = null;
        byte[]? shadowMap = null;
        byte[]? mccvData = null;
        byte[]? mclvData = null;

        int position = startOffset;
        while (position <= payload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(payload.AsSpan(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            int declaredSize = unchecked((int)header.Size);
            int consumedSize = header.Id == AdtChunkIds.Mcnr
                ? Math.Max(declaredSize, McnrConsumedSize)
                : declaredSize;
            long nextOffset = (long)position + ChunkHeader.SizeInBytes + consumedSize;
            if (declaredSize < 0 || nextOffset > payload.Length)
                break;

            int dataOffset = position + ChunkHeader.SizeInBytes;
            if (header.Id == AdtChunkIds.Mcly)
            {
                byte[] layerPayload = new byte[declaredSize];
                Buffer.BlockCopy(payload, dataOffset, layerPayload, 0, declaredSize);
                foreach (AdtTextureLayerDescriptor layer in AdtMcalDecoder.ReadTextureLayers(layerPayload))
                    layers.Add(new LkMclyEntry(layer.TextureId, layer.Flags, layer.AlphaOffset, layer.EffectId));
            }
            else if (header.Id == AdtChunkIds.Mcal)
            {
                int mcalPayloadSize = declaredSize;
                if (headerMcalPayloadSize.HasValue && headerMcalPayloadSize.Value > mcalPayloadSize && dataOffset + headerMcalPayloadSize.Value <= payload.Length)
                    mcalPayloadSize = headerMcalPayloadSize.Value;

                alphaData = new byte[mcalPayloadSize];
                Buffer.BlockCopy(payload, dataOffset, alphaData, 0, mcalPayloadSize);
            }
            else if (header.Id == AdtChunkIds.Mcsh)
            {
                shadowMap = new byte[declaredSize];
                Buffer.BlockCopy(payload, dataOffset, shadowMap, 0, declaredSize);
            }
            else if (header.Id == AdtChunkIds.Mccv)
            {
                mccvData = new byte[declaredSize];
                Buffer.BlockCopy(payload, dataOffset, mccvData, 0, declaredSize);
            }
            else if (header.Id == AdtChunkIds.Mclv)
            {
                mclvData = new byte[declaredSize];
                Buffer.BlockCopy(payload, dataOffset, mclvData, 0, declaredSize);
            }

            position = checked((int)nextOffset);
        }

        return new LkMcnkData
        {
            IndexX = chunkX,
            IndexY = chunkY,
            Flags = unchecked((int)flags),
            NLayers = layers.Count,
            Layers = layers,
            AlphaMapData = alphaData,
            AlphaMapSize = alphaData?.Length ?? 0,
            ShadowMap = shadowMap,
            MccvColors = mccvData,
            MclvLighting = mclvData,
        };
    }

    private static bool TryReadEmbeddedSplitMcnkHeader(
        byte[] payload,
        out uint flags,
        out int? headerMcalPayloadSize,
        out int? chunkX,
        out int? chunkY)
    {
        flags = 0;
        headerMcalPayloadSize = null;
        chunkX = null;
        chunkY = null;

        if (payload.Length < RootMcnkHeaderSize + ChunkHeader.SizeInBytes)
            return false;

        if (!ChunkHeaderReader.TryRead(payload.AsSpan(RootMcnkHeaderSize, ChunkHeader.SizeInBytes), out ChunkHeader firstSubchunk))
            return false;

        if (!IsKnownSplitSubchunk(firstSubchunk.Id))
            return false;

        int declaredSize = checked((int)firstSubchunk.Size);
        int consumedSize = firstSubchunk.Id == AdtChunkIds.Mcnr
            ? Math.Max(declaredSize, McnrConsumedSize)
            : declaredSize;
        long firstSubchunkEnd = (long)RootMcnkHeaderSize + ChunkHeader.SizeInBytes + consumedSize;
        if (declaredSize < 0 || firstSubchunkEnd > payload.Length)
            return false;

        flags = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x00, 4));
        chunkX = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x04, 4)));
        chunkY = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x08, 4)));

        uint sizeMcal = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(RootMcnkMcalSizeOffset, 4));
        if (sizeMcal >= ChunkHeader.SizeInBytes)
            headerMcalPayloadSize = checked((int)(sizeMcal - ChunkHeader.SizeInBytes));

        return true;
    }

    private static bool IsKnownSplitSubchunk(FourCC id)
    {
        return id == AdtChunkIds.Mcvt
            || id == AdtChunkIds.Mcnr
            || id == AdtChunkIds.Mcly
            || id == AdtChunkIds.Mcal
            || id == AdtChunkIds.Mcsh
            || id == AdtChunkIds.Mcse
            || id == AdtChunkIds.Mccv
            || id == AdtChunkIds.Mclv
            || id == AdtChunkIds.Mclq
            || id == AdtChunkIds.Mcrd
            || id == AdtChunkIds.Mcrw
            || id == AdtChunkIds.Mcmt;
    }

    private static byte[]? TryReadSplitMcnkSubchunkPayload(ReadOnlySpan<byte> payload, FourCC chunkId)
    {
        int position = 0;
        while (position <= payload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(payload.Slice(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            int declaredSize = checked((int)header.Size);
            int consumedSize = header.Id == AdtChunkIds.Mcnr
                ? Math.Max(declaredSize, McnrConsumedSize)
                : declaredSize;
            long nextOffset = (long)position + ChunkHeader.SizeInBytes + consumedSize;
            if (declaredSize < 0 || nextOffset > payload.Length)
                break;

            if (header.Id == chunkId)
                return payload.Slice(position + ChunkHeader.SizeInBytes, declaredSize).ToArray();

            position = checked((int)nextOffset);
        }

        return null;
    }

    private static void MergeChunks(Dictionary<int, LkMcnkData> root, Dictionary<int, LkMcnkData> other)
    {
        foreach (var kvp in other)
        {
            if (root.TryGetValue(kvp.Key, out var rootChunk))
            {
                root[kvp.Key] = new LkMcnkData
                {
                    IndexX = rootChunk.IndexX,
                    IndexY = rootChunk.IndexY,
                    Flags = rootChunk.Flags,
                    AreaId = rootChunk.AreaId,
                    NLayers = rootChunk.NLayers,
                    HoleMask = rootChunk.HoleMask,
                    BaseHeight = rootChunk.BaseHeight,
                    Heights = rootChunk.Heights,
                    Normals = rootChunk.Normals,
                    ShadowMap = kvp.Value.ShadowMap ?? rootChunk.ShadowMap,
                    AlphaMapData = kvp.Value.AlphaMapData ?? rootChunk.AlphaMapData,
                    AlphaMapSize = kvp.Value.AlphaMapData != null ? kvp.Value.AlphaMapSize : rootChunk.AlphaMapSize,
                    Layers = (kvp.Value.Layers != null && kvp.Value.Layers.Count > 0) ? kvp.Value.Layers : rootChunk.Layers,
                    DoodadRefs = (kvp.Value.DoodadRefs != null && kvp.Value.DoodadRefs.Count > 0) ? kvp.Value.DoodadRefs : rootChunk.DoodadRefs,
                    WorldModelRefs = (kvp.Value.WorldModelRefs != null && kvp.Value.WorldModelRefs.Count > 0) ? kvp.Value.WorldModelRefs : rootChunk.WorldModelRefs,
                    LiquidData = rootChunk.LiquidData,
                    MccvColors = rootChunk.MccvColors,
                    MclvLighting = rootChunk.MclvLighting,
                    PosX = rootChunk.PosX,
                    PosY = rootChunk.PosY,
                    PosZ = rootChunk.PosZ
                };
            }
            else
            {
                root[kvp.Key] = kvp.Value;
            }
        }
    }

    private static void ParseAdtStream(byte[] bytes, Dictionary<int, LkMcnkData> chunks,
        List<string> textureNames, List<string> modelNames, List<string> worldModelNames,
        List<LkMddfEntry> modelPlacements, List<LkModfEntry> worldModelPlacements,
        ref uint mhdrFlags, ref int[,,]? mfboFlightBounds)
    {
        using var ms = new MemoryStream(bytes, writable: false);
        using var br = new BinaryReader(ms, Encoding.ASCII, leaveOpen: true);

        while (ms.Position + 8 <= ms.Length)
        {
            long headerOffset = ms.Position;
            byte[] tagBytes = br.ReadBytes(4);
            uint size = br.ReadUInt32();
            long chunkEnd = Math.Min(ms.Position + size, ms.Length);
            string tag = Encoding.ASCII.GetString(tagBytes);

            // In split ADTs, KNCM is the tag for MCNK chunks
            if (tag == "MCNK" || tag == "KNCM")
            {
                var chunk = ReadMcnkChunk(br, bytes, (int)size);
                int index = chunk.IndexY * 16 + chunk.IndexX;
                chunks[index] = chunk;
            }
            else if (tag == "MHDR" || tag == "RDHM")
            {
                mhdrFlags = br.ReadUInt32();
                ms.Position = chunkEnd;
            }
            else if (tag == "MTEX" || tag == "XETM")
            {
                if (textureNames.Count == 0)
                    textureNames.AddRange(ReadStringBlock(br, (int)size));
                else ms.Position = chunkEnd;
            }
            else if (tag == "MMDX" || tag == "XDMM")
            {
                if (modelNames.Count == 0)
                    modelNames.AddRange(ReadStringBlock(br, (int)size));
                else ms.Position = chunkEnd;
            }
            else if (tag == "MWMO" || tag == "OMWM")
            {
                if (worldModelNames.Count == 0)
                    worldModelNames.AddRange(ReadStringBlock(br, (int)size));
                else ms.Position = chunkEnd;
            }
            else if (tag == "MDDF" || tag == "FDDM")
            {
                if (modelPlacements.Count == 0)
                    modelPlacements.AddRange(ReadMddfEntries(br, (int)size));
                else ms.Position = chunkEnd;
            }
            else if (tag == "MODF" || tag == "FDOM")
            {
                if (worldModelPlacements.Count == 0)
                    worldModelPlacements.AddRange(ReadModfEntries(br, (int)size));
                else ms.Position = chunkEnd;
            }
            else if (tag == "MFBO")
            {
                if (size >= 36)
                {
                    mfboFlightBounds = new int[2, 3, 3];
                    for (int plane = 0; plane < 2; plane++)
                        for (int row = 0; row < 3; row++)
                            for (int col = 0; col < 3; col++)
                                mfboFlightBounds[plane, row, col] = br.ReadInt16();
                }
                else ms.Position = chunkEnd;
            }
            else
            {
                ms.Position = chunkEnd;
            }

            long nextOffset = AdvanceChunkPosition(bytes, (int)headerOffset, size);
            if (nextOffset <= headerOffset || nextOffset > ms.Length)
                break;

            ms.Position = nextOffset;
        }
    }

    private static int AdvanceChunkPosition(byte[] bytes, int headerOffset, uint size)
    {
        long unpadded = (long)headerOffset + 8 + size;
        if (unpadded <= headerOffset)
            return -1;

        if ((size & 1) == 0)
            return unpadded <= int.MaxValue ? (int)unpadded : -1;

        long padded = unpadded + 1;
        bool unpaddedLooksValid = LooksLikeChunkHeader(bytes, unpadded);
        bool paddedLooksValid = LooksLikeChunkHeader(bytes, padded);

        if (unpaddedLooksValid && !paddedLooksValid)
            return (int)unpadded;

        if (paddedLooksValid && !unpaddedLooksValid)
            return (int)padded;

        if (unpaddedLooksValid)
            return (int)unpadded;

        return padded <= bytes.Length && padded <= int.MaxValue
            ? (int)padded
            : (unpadded <= bytes.Length && unpadded <= int.MaxValue ? (int)unpadded : -1);
    }

    private static bool LooksLikeChunkHeader(byte[] bytes, long offset)
    {
        if (offset < 0 || offset + 8 > bytes.Length || offset > int.MaxValue)
            return false;

        int pos = (int)offset;
        for (int i = 0; i < 4; i++)
        {
            byte value = bytes[pos + i];
            bool isUppercase = value >= (byte)'A' && value <= (byte)'Z';
            bool isDigit = value >= (byte)'0' && value <= (byte)'9';
            if (!isUppercase && !isDigit && value != (byte)'_')
                return false;
        }

        uint size = BitConverter.ToUInt32(bytes, pos + 4);
        return offset + 8 + size <= bytes.Length;
    }

    private static IReadOnlyList<LkMcnkData> AttachLiquidData(byte[] adtBytes, List<LkMcnkData> chunks, int tileX, int tileY)
    {
        try
        {
            using var stream = new MemoryStream(adtBytes, writable: false);
            MapFileSummary summary = MapFileSummaryReader.Read(stream, $"tile_{tileX}_{tileY}.adt");
            AdtLiquidFile liquidFile = AdtLiquidReader.Read(stream, summary);
            if (liquidFile.Chunks.Count == 0)
                return chunks;

            var liquidByChunkIndex = liquidFile.Chunks
                .Where(static chunk => chunk.Layers.Count > 0)
                .ToDictionary(static chunk => chunk.ChunkIndex);

            var result = new List<LkMcnkData>(chunks.Count);
            for (int index = 0; index < chunks.Count; index++)
            {
                LkMcnkData chunk = chunks[index];
                liquidByChunkIndex.TryGetValue(index, out AdtLiquidChunk? liquidData);
                result.Add(new LkMcnkData
                {
                    IndexX = chunk.IndexX,
                    IndexY = chunk.IndexY,
                    Flags = chunk.Flags,
                    AreaId = chunk.AreaId,
                    NLayers = chunk.NLayers,
                    HoleMask = chunk.HoleMask,
                    BaseHeight = chunk.BaseHeight,
                    Heights = chunk.Heights,
                    Normals = chunk.Normals,
                    ShadowMap = chunk.ShadowMap,
                    AlphaMapData = chunk.AlphaMapData,
                    AlphaMapSize = chunk.AlphaMapSize,
                    Layers = chunk.Layers,
                    DoodadRefs = chunk.DoodadRefs,
                    WorldModelRefs = chunk.WorldModelRefs,
                    LiquidData = liquidData,
                    MccvColors = chunk.MccvColors,
                    MclvLighting = chunk.MclvLighting,
                    PosX = chunk.PosX,
                    PosY = chunk.PosY,
                    PosZ = chunk.PosZ
                });
            }

            return result;
        }
        catch
        {
            return chunks;
        }
    }

    private static List<string> ReadStringBlock(BinaryReader br, int size)
    {
        var names = new List<string>();
        byte[] data = br.ReadBytes(size);
        int start = 0;
        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] == 0)
            {
                if (i > start)
                    names.Add(Encoding.UTF8.GetString(data, start, i - start));
                start = i + 1;
            }
        }
        if (start < data.Length)
            names.Add(Encoding.UTF8.GetString(data, start, data.Length - start));
        return names;
    }

    private static List<LkMddfEntry> ReadMddfEntries(BinaryReader br, int size)
    {
        var entries = new List<LkMddfEntry>();
        int count = size / LkMddfEntrySize;
        for (int i = 0; i < count; i++)
        {
            int nameId = br.ReadInt32();
            int uniqueId = br.ReadInt32();
            float rawX = br.ReadSingle();
            float rawZ = br.ReadSingle();
            float rawY = br.ReadSingle();
            float rotX = br.ReadSingle();
            float rotZ = br.ReadSingle();
            float rotY = br.ReadSingle();
            ushort scale = br.ReadUInt16();
            br.ReadUInt16();
            entries.Add(new LkMddfEntry(nameId, uniqueId,
                new System.Numerics.Vector3(MapOrigin - rawY, MapOrigin - rawX, rawZ),
                new System.Numerics.Vector3(rotX, rotY, rotZ),
                scale / 1024f));
        }
        return entries;
    }

    private static List<LkModfEntry> ReadModfEntries(BinaryReader br, int size)
    {
        var entries = new List<LkModfEntry>();
        int count = size / LkModfEntrySize;
        for (int i = 0; i < count; i++)
        {
            int nameId = br.ReadInt32();
            int uniqueId = br.ReadInt32();
            float rawX = br.ReadSingle();
            float rawZ = br.ReadSingle();
            float rawY = br.ReadSingle();
            float rotX = br.ReadSingle();
            float rotZ = br.ReadSingle();
            float rotY = br.ReadSingle();
            float bbMinX = br.ReadSingle();
            float bbMinZ = br.ReadSingle();
            float bbMinY = br.ReadSingle();
            float bbMaxX = br.ReadSingle();
            float bbMaxZ = br.ReadSingle();
            float bbMaxY = br.ReadSingle();
            ushort modfFlags = br.ReadUInt16();
            ushort doodadSet = br.ReadUInt16();
            ushort nameSet = br.ReadUInt16();
            ushort modfScale = br.ReadUInt16();
            entries.Add(new LkModfEntry(nameId, uniqueId,
                new System.Numerics.Vector3(MapOrigin - rawY, MapOrigin - rawX, rawZ),
                new System.Numerics.Vector3(rotX, rotY, rotZ),
                new System.Numerics.Vector3(MapOrigin - bbMaxY, MapOrigin - bbMaxX, bbMinZ),
                new System.Numerics.Vector3(MapOrigin - bbMinY, MapOrigin - bbMinX, bbMaxZ),
                modfFlags, doodadSet, nameSet, modfScale / 1024f));
        }
        return entries;
    }

    private static LkMcnkData ReadMcnkChunk(BinaryReader br, byte[] adtBytes, int declaredSize)
    {
        long mcnkStart = br.BaseStream.Position - 8;
        int headerSize = 128;
        if (declaredSize < headerSize)
        {
            br.BaseStream.Position = mcnkStart + 8 + declaredSize;
            return new LkMcnkData { IndexX = 0, IndexY = 0, Heights = [], Normals = [], Layers = [] };
        }

        byte[] header = br.ReadBytes(headerSize);

        int mcnkFlags = BitConverter.ToInt32(header, 0x00);
        int indexX = BitConverter.ToInt32(header, 0x04);
        int indexY = BitConverter.ToInt32(header, 0x08);
        int nLayers = BitConverter.ToInt32(header, 0x0C);
        int ofsMcvt = BitConverter.ToInt32(header, 0x14);
        int ofsMcnr = BitConverter.ToInt32(header, 0x18);
        int ofsMcly = BitConverter.ToInt32(header, 0x1C);
        int ofsMcrf = BitConverter.ToInt32(header, 0x20);
        int ofsMcal = BitConverter.ToInt32(header, 0x24);
        int sizeMcal = BitConverter.ToInt32(header, 0x28);
        int ofsMcsh = BitConverter.ToInt32(header, 0x2C);
        int sizeMcsh = BitConverter.ToInt32(header, 0x30);
        int areaId = BitConverter.ToInt32(header, 0x34);
        int nDoodadRefs = BitConverter.ToInt32(header, 0x38);
        int holeMask = BitConverter.ToInt32(header, 0x3C);
        int nMapObjRefs = BitConverter.ToInt32(header, 0x3C + 0x14);
        int sizeMclq = BitConverter.ToInt32(header, 0x64);
        int ofsLiquid = BitConverter.ToInt32(header, 0x68);
        float baseHeight = BitConverter.ToSingle(header, 0x70);
        float posX = BitConverter.ToSingle(header, 0x68);
        float posY = BitConverter.ToSingle(header, 0x6C);

        int mcnkPayloadOffset = (int)mcnkStart + 8;
        int mcnkPayloadEnd = mcnkPayloadOffset + declaredSize;
        if (mcnkPayloadEnd > adtBytes.Length)
            mcnkPayloadEnd = adtBytes.Length;

        byte[] heightData = [];
        byte[] normalData = [];
        byte[]? shadowData = null;
        var layers = new List<LkMclyEntry>();
        byte[]? alphaData = null;
        int alphaTotalSize = 0;
        var doodadRefs = new List<int>();
        var worldModelRefs = new List<int>();

        if (ofsMcvt >= headerSize && ofsMcvt + 145 * 4 <= declaredSize)
        {
            int srcOffset = mcnkPayloadOffset + ofsMcvt;
            if (srcOffset + 145 * 4 <= adtBytes.Length)
            {
                heightData = new byte[145 * 4];
                Buffer.BlockCopy(adtBytes, srcOffset, heightData, 0, 145 * 4);
            }
        }

        if (ofsMcnr >= headerSize && ofsMcnr + 448 <= declaredSize)
        {
            int srcOffset = mcnkPayloadOffset + ofsMcnr;
            if (srcOffset + 448 <= adtBytes.Length)
            {
                normalData = new byte[448];
                Buffer.BlockCopy(adtBytes, srcOffset, normalData, 0, 448);
            }
        }

        if (ofsMcly >= headerSize)
        {
            int srcOffset = mcnkPayloadOffset + ofsMcly;
            if (srcOffset + 8 <= adtBytes.Length)
            {
                int mclyReadEnd = Math.Min(srcOffset + declaredSize - ofsMcly, adtBytes.Length) - srcOffset;
                int mclyAvail = Math.Max(0, mclyReadEnd);
                int layerCount = Math.Min(Math.Max(nLayers, 0), mclyAvail / 16);
                for (int i = 0; i < layerCount; i++)
                {
                    int off = srcOffset + i * 16;
                    if (off + 16 > adtBytes.Length) break;
                    uint texId = BitConverter.ToUInt32(adtBytes, off);
                    uint layerMclyFlags = BitConverter.ToUInt32(adtBytes, off + 4);
                    uint alphaOff = BitConverter.ToUInt32(adtBytes, off + 8);
                    uint effectId = BitConverter.ToUInt32(adtBytes, off + 12);
                    layers.Add(new LkMclyEntry(texId, layerMclyFlags, alphaOff, effectId));
                }
            }
        }

        if (ofsMcrf >= headerSize)
        {
            int srcOffset = mcnkPayloadOffset + ofsMcrf;
            int available = Math.Min(declaredSize - ofsMcrf, adtBytes.Length - srcOffset);
            if (available > 8)
            {
                using var mcrfMs = new MemoryStream(adtBytes, srcOffset, available, writable: false);
                using var mcrfBr = new BinaryReader(mcrfMs, Encoding.ASCII, leaveOpen: true);
                int nRefs = mcrfBr.ReadInt32();
                for (int i = 0; i < nRefs && mcrfMs.Position + 4 <= mcrfMs.Length; i++)
                    doodadRefs.Add(mcrfBr.ReadInt32());
                if (mcrfMs.Position + 4 <= mcrfMs.Length)
                {
                    int nWmoRefs = mcrfBr.ReadInt32();
                    for (int i = 0; i < nWmoRefs && mcrfMs.Position + 4 <= mcrfMs.Length; i++)
                        worldModelRefs.Add(mcrfBr.ReadInt32());
                }
            }
        }

        if (ofsMcal >= headerSize && sizeMcal > 0)
        {
            int srcOffset = mcnkPayloadOffset + ofsMcal;
            if (srcOffset + sizeMcal <= adtBytes.Length)
            {
                alphaData = new byte[sizeMcal];
                Buffer.BlockCopy(adtBytes, srcOffset, alphaData, 0, sizeMcal);
                alphaTotalSize = sizeMcal;
            }
        }

        if (ofsMcsh >= headerSize && sizeMcsh > 0)
        {
            int srcOffset = mcnkPayloadOffset + ofsMcsh;
            if (srcOffset + sizeMcsh <= adtBytes.Length)
            {
                shadowData = new byte[sizeMcsh];
                Buffer.BlockCopy(adtBytes, srcOffset, shadowData, 0, sizeMcsh);
            }
        }

        byte[]? mccvData = null;
        byte[]? mclvData = null;
        int scanStart = mcnkPayloadOffset + headerSize;
        int scanEnd = mcnkPayloadEnd;
        int pos = scanStart;
        while (pos + 8 <= scanEnd)
        {
            if (pos + 4 <= adtBytes.Length)
            {
                string subTag = Encoding.ASCII.GetString(adtBytes, pos, 4);
                if (pos + 8 <= adtBytes.Length)
                {
                    int subSize = BitConverter.ToInt32(adtBytes, pos + 4);
                    if (subSize < 0 || pos + 8 + subSize > scanEnd)
                        break;

                    if (subTag == "MCCV" && subSize >= 580)
                    {
                        int dataOffset = pos + 8;
                        if (dataOffset + 580 <= adtBytes.Length)
                        {
                            mccvData = new byte[580];
                            Buffer.BlockCopy(adtBytes, dataOffset, mccvData, 0, 580);
                        }
                    }
                    else if (subTag == "MCLV" && subSize >= 580)
                    {
                        int dataOffset = pos + 8;
                        if (dataOffset + 580 <= adtBytes.Length)
                        {
                            mclvData = new byte[580];
                            Buffer.BlockCopy(adtBytes, dataOffset, mclvData, 0, 580);
                        }
                    }

                    int skip = 8 + subSize;
                    pos += (skip + 3) & ~3;
                }
                else break;
            }
            else break;
        }

        br.BaseStream.Position = mcnkStart + 8 + declaredSize;

        float[] heights = new float[145];
        for (int i = 0; i < 145 && i * 4 + 4 <= heightData.Length; i++)
            heights[i] = BitConverter.ToSingle(heightData, i * 4);

        return new LkMcnkData
        {
            IndexX = indexX,
            IndexY = indexY,
            Flags = mcnkFlags,
            AreaId = areaId,
            NLayers = nLayers,
            HoleMask = holeMask,
            BaseHeight = baseHeight,
            Heights = heights,
            Normals = normalData,
            ShadowMap = shadowData,
            AlphaMapData = alphaData,
            AlphaMapSize = alphaTotalSize,
            Layers = layers,
            DoodadRefs = doodadRefs,
            WorldModelRefs = worldModelRefs,
            MccvColors = mccvData,
            MclvLighting = mclvData,
            PosX = posX,
            PosY = posY,
            PosZ = baseHeight
        };
    }
}
