# C002: MHDR

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
Header chunk for ADT files containing offsets to other chunks and global flags for the ADT map tile. This chunk provides a directory of all other chunks in the ADT file and includes metadata about special features present in the tile.

## Original Structure (C++)
```cpp
struct MHDR 
{
    enum MHDRFlags 
    { 
        mhdr_MFBO = 0x01,      // contains a MFBO chunk
        mhdr_northrend = 0x02, // is set for some northrend ones
    };
    
    /*0x00*/ uint32_t flags;         // Flags for this map tile
    /*0x04*/ uint32_t mcin;          // Offset to MCIN chunk (Cata+: removed)
    /*0x08*/ uint32_t mtex;          // Offset to MTEX chunk
    /*0x0C*/ uint32_t mmdx;          // Offset to MMDX chunk
    /*0x10*/ uint32_t mmid;          // Offset to MMID chunk
    /*0x14*/ uint32_t mwmo;          // Offset to MWMO chunk
    /*0x18*/ uint32_t mwid;          // Offset to MWID chunk
    /*0x1C*/ uint32_t mddf;          // Offset to MDDF chunk
    /*0x20*/ uint32_t modf;          // Offset to MODF chunk
    /*0x24*/ uint32_t mfbo;          // Offset to MFBO chunk (only if flags & mhdr_MFBO)
    /*0x28*/ uint32_t mh2o;          // Offset to MH2O chunk
    /*0x2C*/ uint32_t mtxf;          // Offset to MTXF chunk
    /*0x30*/ uint8_t mamp_value;     // Global MAMP value (Cata+, explicit MAMP chunk overrides data)
    /*0x31*/ uint8_t padding[3];     // Padding bytes
    /*0x34*/ uint32_t unused[3];     // Unused data
};
```

## Properties
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | flags | uint32_t | Flags indicating presence of certain chunks |
| 0x04 | mcin | uint32_t | Offset to MCIN chunk (not used in Cata+) |
| 0x08 | mtex | uint32_t | Offset to MTEX chunk |
| 0x0C | mmdx | uint32_t | Offset to MMDX chunk |
| 0x10 | mmid | uint32_t | Offset to MMID chunk |
| 0x14 | mwmo | uint32_t | Offset to MWMO chunk |
| 0x18 | mwid | uint32_t | Offset to MWID chunk |
| 0x1C | mddf | uint32_t | Offset to MDDF chunk |
| 0x20 | modf | uint32_t | Offset to MODF chunk |
| 0x24 | mfbo | uint32_t | Offset to MFBO chunk (only if flags & mhdr_MFBO) |
| 0x28 | mh2o | uint32_t | Offset to MH2O chunk |
| 0x2C | mtxf | uint32_t | Offset to MTXF chunk |
| 0x30 | mamp_value | uint8_t | Global MAMP value (Cata+, explicit MAMP chunk overrides data) |
| 0x31 | padding | uint8_t[3] | Padding bytes |
| 0x34 | unused | uint32_t[3] | Unused data |

## Flags
| Name | Value | Description |
|------|-------|-------------|
| mhdr_MFBO | 0x01 | Contains a MFBO chunk |
| mhdr_northrend | 0x02 | Set for some Northrend ADTs |

## Dependencies
- MVER (C001) - Must be read first to confirm file version

## Implementation Notes
- Split files: only appears in root file
- Contains offsets relative to the start of the MHDR data (after chunk header) in the file
- WoW only takes this for parsing the ADT file
- All offsets are relative to the start of the MHDR data
- Some offsets may be 0 if the corresponding chunk doesn't exist
- For Cataclysm and later, several chunks may be moved to split files

## C# Implementation
```csharp
[Flags]
public enum MHDRFlags
{
    None = 0,
    HasMFBO = 0x01,
    Northrend = 0x02
}

public class MHDR : IChunk
{
    public MHDRFlags Flags { get; set; }
    public uint MCINOffset { get; set; }
    public uint MTEXOffset { get; set; }
    public uint MMDXOffset { get; set; }
    public uint MMIDOffset { get; set; }
    public uint MWMOOffset { get; set; }
    public uint MWIDOffset { get; set; }
    public uint MDDFOffset { get; set; }
    public uint MODFOffset { get; set; }
    public uint MFBOOffset { get; set; }
    public uint MH2OOffset { get; set; }
    public uint MTXFOffset { get; set; }
    public byte MAMPValue { get; set; }
    public byte[] Padding { get; private set; }
    public uint[] Unused { get; private set; }

    public MHDR()
    {
        Flags = MHDRFlags.None;
        MCINOffset = 0;
        MTEXOffset = 0;
        MMDXOffset = 0;
        MMIDOffset = 0;
        MWMOOffset = 0;
        MWIDOffset = 0;
        MDDFOffset = 0;
        MODFOffset = 0;
        MFBOOffset = 0;
        MH2OOffset = 0;
        MTXFOffset = 0;
        MAMPValue = 0;
        Padding = new byte[3];
        Unused = new uint[3];
    }

    public void Parse(BinaryReader reader, long size)
    {
        if (size != 64)
            throw new InvalidDataException($"MHDR chunk has invalid size: {size} bytes (expected 64)");

        Flags = (MHDRFlags)reader.ReadUInt32();
        MCINOffset = reader.ReadUInt32();
        MTEXOffset = reader.ReadUInt32();
        MMDXOffset = reader.ReadUInt32();
        MMIDOffset = reader.ReadUInt32();
        MWMOOffset = reader.ReadUInt32();
        MWIDOffset = reader.ReadUInt32();
        MDDFOffset = reader.ReadUInt32();
        MODFOffset = reader.ReadUInt32();
        MFBOOffset = reader.ReadUInt32();
        MH2OOffset = reader.ReadUInt32();
        MTXFOffset = reader.ReadUInt32();
        MAMPValue = reader.ReadByte();
        
        Padding = reader.ReadBytes(3);
        
        Unused = new uint[3];
        for (int i = 0; i < 3; i++)
            Unused[i] = reader.ReadUInt32();
    }

    public void Write(BinaryWriter writer)
    {
        writer.Write((uint)Flags);
        writer.Write(MCINOffset);
        writer.Write(MTEXOffset);
        writer.Write(MMDXOffset);
        writer.Write(MMIDOffset);
        writer.Write(MWMOOffset);
        writer.Write(MWIDOffset);
        writer.Write(MDDFOffset);
        writer.Write(MODFOffset);
        writer.Write(MFBOOffset);
        writer.Write(MH2OOffset);
        writer.Write(MTXFOffset);
        writer.Write(MAMPValue);
        writer.Write(Padding);
        
        foreach (uint unused in Unused)
            writer.Write(unused);
    }
    
    public bool HasChunk(ChunkType type)
    {
        switch (type)
        {
            case ChunkType.MCIN: return MCINOffset != 0;
            case ChunkType.MTEX: return MTEXOffset != 0;
            case ChunkType.MMDX: return MMDXOffset != 0;
            case ChunkType.MMID: return MMIDOffset != 0;
            case ChunkType.MWMO: return MWMOOffset != 0;
            case ChunkType.MWID: return MWIDOffset != 0;
            case ChunkType.MDDF: return MDDFOffset != 0;
            case ChunkType.MODF: return MODFOffset != 0;
            case ChunkType.MFBO: return MFBOOffset != 0 && Flags.HasFlag(MHDRFlags.HasMFBO);
            case ChunkType.MH2O: return MH2OOffset != 0;
            case ChunkType.MTXF: return MTXFOffset != 0;
            default: return false;
        }
    }
    
    public uint GetChunkOffset(ChunkType type)
    {
        switch (type)
        {
            case ChunkType.MCIN: return MCINOffset;
            case ChunkType.MTEX: return MTEXOffset;
            case ChunkType.MMDX: return MMDXOffset;
            case ChunkType.MMID: return MMIDOffset;
            case ChunkType.MWMO: return MWMOOffset;
            case ChunkType.MWID: return MWIDOffset;
            case ChunkType.MDDF: return MDDFOffset;
            case ChunkType.MODF: return MODFOffset;
            case ChunkType.MFBO: return MFBOOffset;
            case ChunkType.MH2O: return MH2OOffset;
            case ChunkType.MTXF: return MTXFOffset;
            default: return 0;
        }
    }
}

public enum ChunkType
{
    MCIN,
    MTEX,
    MMDX,
    MMID,
    MWMO,
    MWID,
    MDDF,
    MODF,
    MFBO,
    MH2O,
    MTXF
}
```

## Usage Context
The MHDR chunk serves as a directory for locating other chunks within the ADT file. The offsets are relative to the start of the MHDR data section (after the chunk header) and point to the corresponding chunks. This allows the client to efficiently find and parse only the chunks it needs without scanning the entire file. The flags provide quick information about special features in this map tile, such as whether it contains flight boundaries (MFBO). 