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

    public static AdtData Parse(byte[] data, string mapName, int tileX, int tileY, Func<uint, string?>? nameResolver = null)
    {
        var parser = new AdtParser(data);
        return parser.ParseInternal(mapName, tileX, tileY, nameResolver);
    }

    private AdtData ParseInternal(string mapName, int tileX, int tileY, Func<uint, string?>? nameResolver)
    {
        var adtData = new AdtData
        {
            MapName = mapName,
            TileX = tileX,
            TileY = tileY
        };

        // Raw string data (offset -> string)
        var m2StringBlock = new byte[0];
        var wmoStringBlock = new byte[0];
        var m2Offsets = new List<uint>();  // MMID: offsets into MMDX
        var wmoOffsets = new List<uint>(); // MWID: offsets into MWMO
        uint version = 18; // Default to 18 (Classic/WotLK)

        // Read all root chunks
        while (_pos + 8 <= _data.Length)
        {
            var chunkId = ReadChunkId();
            var size = ReadInt32();
            var nextPos = _pos + size;

            switch (chunkId)
            {
                case "MVER":
                    if (size == 4) version = BitConverter.ToUInt32(_data, _pos);
                    break;
                case "MHDR":
                case "MCIN":
                case "MH2O":
                case "MFBO":
                    break;

                case "MTEX":
                    adtData.Textures.AddRange(ParseStringBlock(size));
                    break;

                case "MMDX":
                    m2StringBlock = new byte[size];
                    Array.Copy(_data, _pos, m2StringBlock, 0, size);
                    break;

                case "MMID":
                    m2Offsets = ParseOffsetArray(size);
                    break;

                case "MWID":
                    wmoOffsets = ParseOffsetArray(size);
                    break;

                case "MWMO":
                    wmoStringBlock = new byte[size];
                    Array.Copy(_data, _pos, wmoStringBlock, 0, size);
                    break;

                case "MDDF":
                    ParseMddf(size, m2StringBlock, m2Offsets, adtData.M2Objects, nameResolver);
                    break;

                case "MODF":
                    ParseModf(size, wmoStringBlock, wmoOffsets, adtData.WmoObjects, nameResolver);
                    break;

                case "MCNK":
                    // Pass version and tile info for coordinate calculation if needed
                    var chunk = ParseMcnk(size, adtData.Chunks.Count, version, tileX, tileY);
                    if (chunk != null)
                    {
                        adtData.Chunks.Add(chunk);
                    }
                    break;
                
                default:
                    break;

            }

            _pos = nextPos;
        }

        return adtData;
    }

    private AdtChunk? ParseMcnk(int size, int index, uint version, int tileX, int tileY)
    {
        // MCNK Header is 128 bytes
        if (size < 128) return null;

        var headerStart = _pos;
        
        _pos += 4; // Flags
        var indexX = ReadInt32();
        var indexY = ReadInt32();
        _pos += 4 + 4; // nLayers, nDoodadRefs
        _pos += 4; // offsetH2O

         float posZ, posX, posY;

        // Version check for Position fields
        // Alpha (ver < 18 or implicit) often lacks these in header or has different structure.
        // Standard v18 MCNK header has Position at 0x68 (104).
        // If version is < 18 (e.g. 0 for Alpha/pre-release without MVER?), we calculate.
        // Note: Some Alpha files MIGHT have MVER 18 but structure differs? 
        // Safer: Check if 0x68 is within header and looks valid, OR just calculate for known Alpha.
        // For now, let's assume version < 18 means calculate.
        
        if (version < 18)
        {
            // Calculate coords based on TileX/Y and IndexX/Y (internal chunk index 0-15)
            // World size = 64 * 533.3333
            const float TileSize = 1600.0f / 3.0f; // 533.3333
            const float ChunkSize = TileSize / 16.0f; // 33.3333
            const float Origin = 32.0f * TileSize; // Center

            // In WoW:
            // X (North/South) decreases from top (Positive) to bottom (Negative).
            // Y (East/West) decreases from left (Positive) to right (Negative).
            // Wait, WoW coords: X is North (+), Y is West (+). 
            // 0,0 is center.
            
            // Tile 32,32 is center?
            // TileX=0 is Top (Max X), TileY=0 is Left (Max Y).
            
            // Chunk index is usually row/col inside tile. 
            // indexX = row (inverted?), indexY = col?
            // Actually MCNK IndexX/Y are coordinates in the tile (0-15).
            
            // Base pos of tile:
            float tileMaxX = Origin - (tileX * TileSize);
            float tileMaxY = Origin - (tileY * TileSize);
            
            // Chunk pos (Top-Left of chunk):
            // MCNK indexX is usually 'row' (X axis related?) or 'col'?
            // Standard: IndexX is column (Y axis?), IndexY is row (X axis?)
            // Let's verify standard calc:
            // X = tileMaxX - (indexY * ChunkSize)
            // Y = tileMaxY - (indexX * ChunkSize)
            // Warning: IndexX/Y meanings swap in some versions.
            
            // For now use standard logic assumption:
            // pos will be calculated. Z set to 0 initially? 
            // Or read from MCVT later if needed.
            
            posX = tileMaxX - (indexY * ChunkSize);
            posY = tileMaxY - (indexX * ChunkSize);
            posZ = 0; // Alpha header doesn't have Z base?
        }
        else
        {
            // Standard reading
            _pos = headerStart + 104;
            posZ = ReadFloat(); 
            posX = ReadFloat();
            posY = ReadFloat();
        }
        
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

    private void ParseMddf(int size, byte[] stringBlock, List<uint> offsets, List<AdtM2Placement> list, Func<uint, string?>? resolver)
    {
        // 36 bytes per entry
        var count = size / 36;
        for (int i = 0; i < count; i++)
        {
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
            
            string name = "";
            // Use MMID offset indirection
            if (nameId < offsets.Count)
            {
                var offset = offsets[(int)nameId];
                name = ReadNullTerminatedString(stringBlock, (int)offset);
            }
            
            // Fallback to resolver (for modern FDID-based maps)
            if (string.IsNullOrEmpty(name) && resolver != null)
            {
                var resolved = resolver(nameId);
                if (!string.IsNullOrEmpty(resolved)) name = resolved;
            }
            
            list.Add(new AdtM2Placement {
                ModelName = name,
                UniqueId = uniqId,
                Position = pos,
                Rotation = rot,
                Scale = scale
            });
        }
    }

    private void ParseModf(int size, byte[] stringBlock, List<uint> offsets, List<AdtWmoPlacement> list, Func<uint, string?>? resolver)
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
            float scaleF = scale == 0 ? 1.0f : scale / 1024f;

            string name = "";
            // Use MWID offset indirection
            if (nameId < offsets.Count)
            {
                var offset = offsets[(int)nameId];
                name = ReadNullTerminatedString(stringBlock, (int)offset);
            }
            
            // Fallback to resolver
            if (string.IsNullOrEmpty(name) && resolver != null)
            {
                 var resolved = resolver(nameId);
                 if (!string.IsNullOrEmpty(resolved)) name = resolved;
            }

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

    private List<uint> ParseOffsetArray(int size)
    {
        var list = new List<uint>();
        var count = size / 4;
        for (int i = 0; i < count; i++)
        {
            list.Add(BitConverter.ToUInt32(_data, _pos + i * 4));
        }
        return list;
    }

    private static string ReadNullTerminatedString(byte[] data, int offset)
    {
        if (offset < 0 || offset >= data.Length) return "";
        int end = offset;
        while (end < data.Length && data[end] != 0) end++;
        if (end == offset) return "";
        return Encoding.UTF8.GetString(data, offset, end - offset);
    }
}
