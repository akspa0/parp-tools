using System.Text;
using System.Numerics;
using WoWRollback.Core.Models.ADT;

namespace WoWRollback.Core.Services.Parsers;

public class AdtParser
{
    private readonly byte[] _data;
    private int _pos;

    public AdtParser(byte[] data)
    {
        _data = data;
        _pos = 0;
    }

    public static AdtData Parse(byte[] data, string mapName, int tileX, int tileY)
    {
        var parser = new AdtParser(data);
        return parser.ParseInternal(mapName, tileX, tileY);
    }

    private AdtData ParseInternal(string mapName, int tileX, int tileY)
    {
        var adtData = new AdtData
        {
            MapName = mapName,
            TileX = tileX,
            TileY = tileY
        };

        var m2Names = new List<string>();
        var wmoNames = new List<string>();

        // Read all root chunks
        while (_pos + 8 <= _data.Length)
        {
            var chunkId = ReadChunkId();
            var size = ReadInt32();
            var nextPos = _pos + size;

            switch (chunkId)
            {
                case "MVER":
                case "MHDR":
                case "MCIN": // Pointers to MCNKs, but we just scan linearly usually or use offsets
                case "MH2O":
                case "MFBO":
                    // Skip header/irrelevant chunks for now or parse if needed
                    break;

                case "MTEX":
                    adtData.Textures.AddRange(ParseStringBlock(size));
                    break;

                case "MMDX":
                    m2Names = ParseStringBlock(size);
                    break;

                case "MMID": // Offsets for MMDX, usually skipped if we just read strings
                case "MWID": // Offsets for MWMO
                    break;

                case "MWMO":
                    wmoNames = ParseStringBlock(size);
                    break;

                case "MDDF":
                    ParseMddf(size, m2Names, adtData.M2Objects);
                    break;

                case "MODF":
                    ParseModf(size, wmoNames, adtData.WmoObjects);
                    break;

                case "MCNK":
                    // MCNK is special, it contains sub-chunks. 
                    // But root MCNKs appear sequentially in the file (usually).
                    // However, we need to treat MCNK content carefully.
                    // The MCNK header is 128 bytes.
                    var mcnkStart = _pos - 8; // Backtrack to include header if we were passing the whole block
                    // Actually, we are INSIDE the chunk data now (after size).
                    // We need to parse the MCNK content.
                    var chunk = ParseMcnk(size, adtData.Chunks.Count);
                    if (chunk != null)
                    {
                        adtData.Chunks.Add(chunk);
                    }
                    break;
                
                default:
                    // Unknown chunk
                    break;

            }

            _pos = nextPos;
        }

        return adtData;
    }

    private AdtChunk? ParseMcnk(int size, int index)
    {
        // MCNK Header is 128 bytes
        if (size < 128) return null;

        var headerStart = _pos;
        
        // Read critical header info
        // +0x00 Flags (uint)
        // +0x04 IndexX (uint)
        // +0x08 IndexY (uint)
        // +0x0C nLayers (uint)
        // +0x10 nDoodadRefs
        // +0x14 offsetH2O
        // +0x18 offsetMVT
        // ...
        // +0x68 Position (x,y,z) (3 floats)
        
        _pos += 4; // Flags
        var indexX = ReadInt32();
        var indexY = ReadInt32();
        _pos += 4 + 4; // nLayers, nDoodadRefs
        _pos += 4; // offsetH2O or offsetMCVT depending on version? WotLK uses strict offsets from MCNK start
        
        // Let's jump to Position at 0x68 (104)
        _pos = headerStart + 104;
        var z = ReadFloat(); // Z (Height) is first in strict ADT coords? No, Position is X,Y,Z
        // Wait, standard ADT header struct:
        // u32 flags, u32 ix, u32 iy, u32 nLayers, u32 nDoodadRefs, u32 ofsMCVT...
        // ... u32 predTex, u32 noEffectDoodad, u32 ofsMCSE, u32 nSndEmitters, u32 ofsMCLQ, u32 sizeMCLQ, float z, float x, float y
        // Z is base height. X, Y are coords.
        // Let's verify MCNK header layout.
        // 0x68 in decimal is 104.
        
        var posZ = ReadFloat(); // Height (Z in WoW)
        var posX = ReadFloat();
        var posY = ReadFloat();
        
        // Holes at +0x3C? Or part of flags?
        // High res holes are 0x00 Flags & 0x10000? 
        // Holes low-res are usually in MCVT or MCNK header + 0x14?
        // standard 3.3.5: holes is u32 holes at 0x3C (60)
         _pos = headerStart + 60;
        var holes = ReadInt32();
        
        // Liquid size check?
        _pos = headerStart + 8 * 4; // Skip to offsetMCLQ
        // offsets usually relative to MCNK start.
        
        // Now parse sub-chunks. They start after the 128 bytes header.
        var subChunkStart = headerStart + 128;
        var mcnkEnd = headerStart + size;
        
        var chunk = new AdtChunk
        {
            IndexX = indexX,
            IndexY = indexY,
            PositionX = posX,
            PositionY = posY,
            PositionZ = posZ, // Base height
            Holes = holes,
            StartIndex = index * 145, // Assumption for simple grid
        };
        
        // Scan subchunks
        int scPos = subChunkStart;
        while (scPos + 8 <= mcnkEnd)
        {
            // Read tag/size manually without advancing global _pos
            var tag = Encoding.ASCII.GetString(_data, scPos, 4);
            var scSize = BitConverter.ToInt32(_data, scPos + 4);
            var scDataStart = scPos + 8;
            var tagRev = new string(tag.Reverse().ToArray()); // WoW chunks often read reversed
            
            if (scSize < 0 || scPos + 8 + scSize > mcnkEnd) break;
            
            switch (tagRev)
            {
                case "MCVT":
                     chunk = chunk with { Heights = ParseMcvt(scDataStart, scSize, posZ) };
                     break;
                case "MCLY":
                     // Requires texture IDs
                     break;
                case "MCAL":
                     var alpha = new byte[scSize];
                     Array.Copy(_data, scDataStart, alpha, 0, scSize);
                     chunk = chunk with { AlphaMap = alpha };
                     break;
                case "MCSH":
                     var shadow = new byte[scSize];
                     Array.Copy(_data, scDataStart, shadow, 0, scSize);
                     chunk = chunk with { ShadowMap = shadow };
                     break;
            }
            
            // For MCLY we need to append. 'with' syntax replaces list? No, lists are ref types.
            if (tagRev == "MCLY")
            {
                var layers = ParseMcly(scDataStart, scSize);
                chunk.Layers.AddRange(layers);
            }
            
            scPos += 8 + scSize;
        }

        return chunk;
    }

    private float[] ParseMcvt(int offset, int size, float baseHeight)
    {
        // 145 floats
        var count = size / 4;
        if (count > 145) count = 145;
        var heights = new float[145];
        for (int i = 0; i < count; i++)
        {
            heights[i] = baseHeight + BitConverter.ToSingle(_data, offset + i * 4);
        }
        return heights;
    }

    private List<AdtTextureLayer> ParseMcly(int offset, int size)
    {
        var list = new List<AdtTextureLayer>();
        var count = size / 16;
        for (int i = 0; i < count; i++)
        {
            var p = offset + i * 16;
            var texId = BitConverter.ToInt32(_data, p);
            var flags = BitConverter.ToUInt32(_data, p + 4);
            var alphaOff = BitConverter.ToInt32(_data, p + 8);
            var effId = BitConverter.ToInt32(_data, p + 12);
            
            list.Add(new AdtTextureLayer
            {
                TextureId = texId,
                Flags = flags,
                AlphaOffset = alphaOff,
                EffectId = effId
                // TextureName resolved later using MTEX list
            });
        }
        return list;
    }

    private void ParseMddf(int size, List<string> names, List<AdtM2Placement> list)
    {
        // 36 bytes per entry
        var count = size / 36;
        for (int i = 0; i < count; i++)
        {
            // We need to use _pos directly if we want to use helper methods, but we are in a switch.
            // Let's assume _pos points to data start.
            // Wait, inside Scan loop `_pos` is updated.
            // But here I'm using `_pos` as a global cursor. 
            // The helper `ParseInternal` sets `_pos = nextPos` after switch.
            // So inside switch, `_pos` is at start of data.
            // So we need to read from `_pos + offset`.
            
            var p = _pos + i * 36;
            var nameId = BitConverter.ToUInt32(_data, p);
            var uniqId = BitConverter.ToUInt32(_data, p + 4);
            var pos = new Vector3(
                BitConverter.ToSingle(_data, p + 8),
                BitConverter.ToSingle(_data, p + 12),
                BitConverter.ToSingle(_data, p + 16)
            );
            var rot = new Vector3(
                BitConverter.ToSingle(_data, p + 20),
                BitConverter.ToSingle(_data, p + 24),
                BitConverter.ToSingle(_data, p + 28)
            );
            var scale = BitConverter.ToUInt16(_data, p + 32) / 1024f;
            // flags at +34
            
            var name = nameId < names.Count ? names[(int)nameId] : "";
            
            list.Add(new AdtM2Placement {
                ModelName = name,
                UniqueId = uniqId,
                Position = pos,
                Rotation = rot,
                Scale = scale
            });
        }
    }

    private void ParseModf(int size, List<string> names, List<AdtWmoPlacement> list)
    {
        // 64 bytes per entry
        var count = size / 64;
        for (int i = 0; i < count; i++)
        {
            var p = _pos + i * 64;
            var nameId = BitConverter.ToUInt32(_data, p);
            var uniqId = BitConverter.ToUInt32(_data, p + 4);
            var pos = new Vector3(
                BitConverter.ToSingle(_data, p + 8),
                BitConverter.ToSingle(_data, p + 12),
                BitConverter.ToSingle(_data, p + 16)
            );
            var rot = new Vector3(
                BitConverter.ToSingle(_data, p + 20),
                BitConverter.ToSingle(_data, p + 24),
                BitConverter.ToSingle(_data, p + 28)
            );
            var min = new Vector3(
                BitConverter.ToSingle(_data, p + 32),
                BitConverter.ToSingle(_data, p + 36),
                BitConverter.ToSingle(_data, p + 40)
            );
            var max = new Vector3(
                BitConverter.ToSingle(_data, p + 44),
                BitConverter.ToSingle(_data, p + 48),
                BitConverter.ToSingle(_data, p + 52)
            );
            // flags +56, doodad +58, name +60, scale +62
            var doodad = BitConverter.ToUInt16(_data, p + 58);
            var nameSet = BitConverter.ToUInt16(_data, p + 60);
            var scale = BitConverter.ToUInt16(_data, p + 62); 
            // If scale is 0? standard says 0 means 1 in some versions, but usually initialized?
            // Default 1024.
            float scaleF = scale == 0 ? 1.0f : scale / 1024f; // Handle potentially uninitialized scale in older ADTs?

            var name = nameId < names.Count ? names[(int)nameId] : "";

            list.Add(new AdtWmoPlacement {
                WmoName = name,
                UniqueId = uniqId,
                Position = pos,
                Rotation = rot,
                ExtentsMin = min,
                ExtentsMax = max,
                DoodadSet = doodad,
                NameSet = nameSet,
                Scale = scaleF
            });
        }
    }

    private List<string> ParseStringBlock(int size)
    {
        var list = new List<string>();
        // Read block from _pos
        var bytes = new byte[size];
        Array.Copy(_data, _pos, bytes, 0, size);
        
        int sStart = 0;
        for (int i = 0; i < size; i++)
        {
            if (bytes[i] == 0)
            {
                if (i > sStart)
                {
                    var str = Encoding.UTF8.GetString(bytes, sStart, i - sStart);
                    list.Add(str);
                }
                sStart = i + 1;
            }
        }
        return list;
    }

    private string ReadChunkId()
    {
        var s = Encoding.ASCII.GetString(_data, _pos, 4);
        _pos += 4;
        return new string(s.Reverse().ToArray());
    }

    private int ReadInt32()
    {
        var v = BitConverter.ToInt32(_data, _pos);
        _pos += 4;
        return v;
    }

    private float ReadFloat()
    {
        var v = BitConverter.ToSingle(_data, _pos);
        _pos += 4;
        return v;
    }
}
