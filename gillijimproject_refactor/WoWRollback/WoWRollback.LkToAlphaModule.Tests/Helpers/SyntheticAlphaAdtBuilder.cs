using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace WoWRollback.LkToAlphaModule.Tests.Helpers;

/// <summary>
/// Builder for creating synthetic Alpha ADT binary data for testing.
/// Produces valid Alpha ADT file format with controllable chunk data.
/// </summary>
public class SyntheticAlphaAdtBuilder
{
    private readonly List<McnkBuilder> _mcnks = new();
    private string _mapName = "TestMap";
    private int _tileX = 0;
    private int _tileY = 0;

    public SyntheticAlphaAdtBuilder WithMapInfo(string mapName, int tileX, int tileY)
    {
        _mapName = mapName;
        _tileX = tileX;
        _tileY = tileY;
        return this;
    }

    public McnkBuilder AddMcnk(int indexX, int indexY)
    {
        var builder = new McnkBuilder(this, indexX, indexY);
        _mcnks.Add(builder);
        return builder;
    }

    /// <summary>
    /// Builds a complete Alpha ADT file as byte array.
    /// Format: MHDR chunk + MCIN chunk + 256 MCNK chunks
    /// </summary>
    public byte[] Build()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Write placeholder MHDR (we'll update offsets later)
        long mhdrPos = ms.Position;
        WriteMhdr(writer);

        // Write MCIN chunk (256 MCNK offset/size pairs)
        long mcinPos = ms.Position;
        var mcnkOffsets = new List<(int offset, int size)>();
        WriteMcinPlaceholder(writer);

        // Write all MCNK chunks and record their offsets
        for (int i = 0; i < 256; i++)
        {
            long mcnkStart = ms.Position;
            
            // Find matching builder or create default
            var builder = _mcnks.Find(m => m.IndexX == i % 16 && m.IndexY == i / 16);
            if (builder == null)
            {
                builder = new McnkBuilder(this, i % 16, i / 16);
            }

            byte[] mcnkData = builder.BuildMcnk();
            writer.Write(mcnkData);

            mcnkOffsets.Add(((int)mcnkStart, mcnkData.Length));
        }

        // Go back and write actual MCIN data
        ms.Seek(mcinPos, SeekOrigin.Begin);
        WriteMcin(writer, mcnkOffsets);

        return ms.ToArray();
    }

    private void WriteMhdr(BinaryWriter writer)
    {
        // MHDR chunk header
        writer.Write(Encoding.ASCII.GetBytes("RDHM")); // "MHDR" reversed
        writer.Write(64); // Size

        // MHDR content (all zeros for now - minimal valid header)
        writer.Write(new byte[64]);
    }

    private void WriteMcinPlaceholder(BinaryWriter writer)
    {
        // MCIN chunk header
        writer.Write(Encoding.ASCII.GetBytes("NICM")); // "MCIN" reversed
        writer.Write(256 * 16); // Size: 256 entries * 16 bytes each

        // Placeholder data (will be overwritten)
        writer.Write(new byte[256 * 16]);
    }

    private void WriteMcin(BinaryWriter writer, List<(int offset, int size)> mcnkOffsets)
    {
        // MCIN chunk header
        writer.Write(Encoding.ASCII.GetBytes("NICM")); // "MCIN" reversed
        writer.Write(256 * 16); // Size

        // Write offset/size pairs
        foreach (var (offset, size) in mcnkOffsets)
        {
            writer.Write(offset);
            writer.Write(size);
            writer.Write(0); // flags
            writer.Write(0); // asyncId
        }
    }

    /// <summary>
    /// Builder for individual MCNK chunks.
    /// </summary>
    public class McnkBuilder
    {
        private readonly SyntheticAlphaAdtBuilder _parent;
        private readonly int _indexX;
        private readonly int _indexY;
        private int _layerCount = 1;
        private byte[]? _mclyData;
        private byte[]? _mcalData;
        private byte[]? _mcvtData;
        private byte[]? _mcnrData;
        private int _areaId = 0;

        public int IndexX => _indexX;
        public int IndexY => _indexY;

        internal McnkBuilder(SyntheticAlphaAdtBuilder parent, int indexX, int indexY)
        {
            _parent = parent;
            _indexX = indexX;
            _indexY = indexY;
        }

        public McnkBuilder WithLayers(int count)
        {
            _layerCount = count;
            return this;
        }

        public McnkBuilder WithAreaId(int areaId)
        {
            _areaId = areaId;
            return this;
        }

        public McnkBuilder WithMclyData(params byte[][] layerEntries)
        {
            using var ms = new MemoryStream();
            foreach (var entry in layerEntries)
            {
                if (entry.Length != 16)
                    throw new ArgumentException("Each MCLY layer entry must be 16 bytes");
                ms.Write(entry, 0, 16);
            }
            _mclyData = ms.ToArray();
            _layerCount = layerEntries.Length;
            return this;
        }

        public McnkBuilder WithMcalData(byte[] alphaData)
        {
            _mcalData = alphaData;
            return this;
        }

        public McnkBuilder WithMcvtData(byte[] vertexData)
        {
            _mcvtData = vertexData;
            return this;
        }

        public McnkBuilder WithMcnrData(byte[] normalData)
        {
            _mcnrData = normalData;
            return this;
        }

        public SyntheticAlphaAdtBuilder And()
        {
            return _parent;
        }

        internal byte[] BuildMcnk()
        {
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            // In Alpha ADT format:
            // - Chunk header: "KNCM" (4 bytes) + size (4 bytes) = 8 bytes
            // - MCNK header: 128 bytes with offsets
            // - Subchunk data follows
            //
            // The extractor treats offsets as relative to dataStart (after 8+128=136 bytes)
            // So offsets in the header should just be the position within the data section
            
            const int mcnkHeaderSize = 128;
            
            // Calculate subchunk offsets (relative to start of data section, NOT including headers)
            int currentDataOffset = 0;
            
            // MCVT (no chunk header in Alpha)
            int mcvtSize = _mcvtData?.Length ?? 580;
            int mcvtOffset = currentDataOffset;
            currentDataOffset += mcvtSize;

            // MCNR (no chunk header in Alpha)
            int mcnrSize = _mcnrData?.Length ?? 448;
            int mcnrOffset = currentDataOffset;
            currentDataOffset += mcnrSize;

            // MCLY (HAS chunk header in Alpha: "YLCM" + size + data)
            int mclyDataSize = _mclyData?.Length ?? (_layerCount * 16);
            int mclyTotalSize = 8 + mclyDataSize;
            int mclyOffset = currentDataOffset;
            currentDataOffset += mclyTotalSize;

            // MCRF (skip for now)
            int mcrfOffset = -1;

            // MCAL (no chunk header, raw data)
            int mcalSize = _mcalData?.Length ?? 0;
            int mcalOffset = mcalSize > 0 ? currentDataOffset : 0;
            currentDataOffset += mcalSize;

            // MCSH (skip for now)
            int mcshOffset = -1;
            int mcshSize = 0;

            // Write MCNK chunk header ("KNCM" + size)
            writer.Write(Encoding.ASCII.GetBytes("KNCM"));
            // payloadSize = MCNK header (128) + all subchunk data
            int payloadSize = mcnkHeaderSize + currentDataOffset;
            writer.Write(payloadSize);

            // Write MCNK header (128 bytes)
            writer.Write(0u);         // 0x00: flags
            writer.Write(_indexX);    // 0x04: ix
            writer.Write(_indexY);    // 0x08: iy
            writer.Write(0);          // 0x0C: radius (float, write as 0)
            writer.Write(_layerCount); // 0x10: nLayers
            writer.Write(0);          // 0x14: nDoodadRefs
            writer.Write(mcvtOffset); // 0x18: ofsHeight
            writer.Write(mcnrOffset); // 0x1C: ofsNormal
            writer.Write(mclyOffset); // 0x20: ofsLayer
            writer.Write(mcrfOffset); // 0x24: ofsRefs
            writer.Write(mcalOffset); // 0x28: ofsAlpha
            writer.Write(mcalSize);   // 0x2C: sizeAlpha
            writer.Write(mcshOffset); // 0x30: ofsShadow
            writer.Write(mcshSize);   // 0x34: sizeShadow
            writer.Write(_areaId);    // 0x38: areaId
            writer.Write(0); // mapObjRefs
            writer.Write((ushort)0); // holes
            writer.Write((ushort)0); // padding
            writer.Write(new byte[16]); // low quality texture map
            writer.Write(0); // predTex
            writer.Write(0); // noEffectDoodad
            writer.Write(0); // offsSndEmitters
            writer.Write(0); // nSndEmitters
            writer.Write(0); // offsLiquid
            writer.Write(0); // sizeLiquid
            writer.Write(new byte[12]); // position
            writer.Write(new byte[21]); // padding to exactly 128 bytes (8 + 13 = 21)

            // Write MCVT (raw data, no header in Alpha)
            if (_mcvtData != null)
            {
                writer.Write(_mcvtData);
            }
            else
            {
                writer.Write(CreateDefaultMcvt());
            }

            // Write MCNR (raw data, no header in Alpha)
            if (_mcnrData != null)
            {
                writer.Write(_mcnrData);
            }
            else
            {
                writer.Write(CreateDefaultMcnr());
            }

            // Write MCLY chunk (HAS header: "YLCM" + size + data)
            writer.Write(Encoding.ASCII.GetBytes("YLCM")); // "MCLY" reversed
            writer.Write(mclyDataSize);
            if (_mclyData != null)
            {
                writer.Write(_mclyData);
            }
            else
            {
                writer.Write(CreateDefaultMcly(_layerCount));
            }

            // Write MCAL (NO header, raw data)
            if (_mcalData != null && _mcalData.Length > 0)
            {
                writer.Write(_mcalData);
            }

            return ms.ToArray();
        }

        private byte[] CreateDefaultMcvt()
        {
            // 145 floats (zero height)
            return new byte[145 * 4];
        }

        private byte[] CreateDefaultMcnr()
        {
            // 145 normals (3 bytes each, pointing up)
            var data = new byte[145 * 3];
            for (int i = 0; i < 145; i++)
            {
                data[i * 3 + 1] = 127; // Y component (up)
            }
            return data;
        }

        private byte[] CreateDefaultMcly(int layerCount)
        {
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            uint alphaOffset = 0;
            for (int i = 0; i < layerCount; i++)
            {
                writer.Write((uint)(100 + i)); // textureId
                writer.Write((uint)(i > 0 ? 0x4 : 0x0)); // flags (use_alpha_map for layers > 0)
                writer.Write(alphaOffset); // offsetInMCAL
                writer.Write((ushort)0); // effectId
                writer.Write((ushort)0); // padding

                if (i > 0)
                {
                    alphaOffset += 2048; // Compressed alpha size estimate
                }
            }

            return ms.ToArray();
        }
    }
}
