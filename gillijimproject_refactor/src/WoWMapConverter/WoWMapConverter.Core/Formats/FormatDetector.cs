using System.Text;

namespace WoWMapConverter.Core.Formats;

/// <summary>
/// Detects WoW file format versions and types.
/// </summary>
public static class FormatDetector
{
    public enum FileType
    {
        Unknown,
        Wdt,          // World Definition Table
        Adt,          // ADT terrain (any version)
        AdtTex,       // Split ADT texture file (_tex0)
        AdtObj,       // Split ADT object file (_obj0)
        AdtLod,       // Split ADT LOD file (_lod)
        Wmo,          // World Map Object
        WmoGroup,     // WMO group file
        M2,           // Model (modern)
        Mdx,          // Model (Alpha/WC3)
        M3,           // Model (Legion+)
        Blp,          // Texture
        Dbc,          // Database Client (pre-Cata)
        Db2,          // Database 2 (Cata+)
        Pm4           // Pathfinding mesh
    }

    public enum WowVersion
    {
        Unknown,
        Alpha053,     // 0.5.3 (Alpha)
        Classic,      // 1.x
        Tbc,          // 2.x
        Wotlk,        // 3.x
        Cataclysm,    // 4.x (split ADT introduced)
        Mop,          // 5.x
        Wod,          // 6.x
        Legion,       // 7.x (_lod.adt introduced)
        Bfa,          // 8.x (MAID in WDT)
        Shadowlands,  // 9.x
        Dragonflight, // 10.x
        War,          // 11.x
        Midnight      // 12.x
    }

    /// <summary>
    /// Detect file type from magic bytes.
    /// </summary>
    public static FileType DetectType(BinaryReader reader)
    {
        var startPos = reader.BaseStream.Position;
        
        if (reader.BaseStream.Length < 8)
            return FileType.Unknown;

        var magic = reader.ReadUInt32();
        var size = reader.ReadUInt32();

        // Check for reversed FourCC (WoW files store reversed)
        var magicStr = Encoding.ASCII.GetString(BitConverter.GetBytes(magic));
        var reversedStr = new string(magicStr.Reverse().ToArray());

        reader.BaseStream.Position = startPos;

        // MDX (Alpha models)
        if (magicStr == "MDLX" || reversedStr == "MDLX")
            return FileType.Mdx;

        // M2 (modern models) - look for MD20 or MD21
        if (magic == 0x3032444D || magic == 0x3132444D) // MD20, MD21
            return FileType.M2;

        // M3 (Legion+ models)
        if (magicStr == "M3DT" || reversedStr == "M3DT" || 
            magicStr == "33DM" || reversedStr == "33DM")
            return FileType.M3;

        // BLP texture
        if (magicStr == "BLP2" || magicStr == "BLP1" || 
            reversedStr == "BLP2" || reversedStr == "BLP1")
            return FileType.Blp;

        // DBC
        if (magic == 0x43424457) // WDBC
            return FileType.Dbc;

        // DB2
        if (magic == 0x32424457 || magic == 0x35424457 || 
            magic == 0x36424457 || magic == 0x37424457) // WDB2-7
            return FileType.Db2;

        // PM4
        if (magicStr == "PM4\0" || reversedStr == "PM4\0" ||
            magicStr == "4MP\0" || reversedStr == "4MP\0")
            return FileType.Pm4;

        // MVER-based files (WDT, ADT, WMO)
        if (magic == 0x5245564D || reversedStr == "MVER") // MVER
        {
            // Read version
            var version = reader.ReadUInt32();
            reader.BaseStream.Position = startPos + 8 + size;

            if (reader.BaseStream.Position >= reader.BaseStream.Length - 8)
            {
                reader.BaseStream.Position = startPos;
                return FileType.Unknown;
            }

            // Read next chunk to determine type
            var nextMagic = reader.ReadUInt32();
            var nextMagicStr = Encoding.ASCII.GetString(BitConverter.GetBytes(nextMagic));
            var nextReversed = new string(nextMagicStr.Reverse().ToArray());

            reader.BaseStream.Position = startPos;

            // WDT has MPHD after MVER
            if (nextMagic == 0x4448504D || nextReversed == "MPHD")
                return FileType.Wdt;

            // WMO root has MOHD after MVER
            if (nextMagic == 0x44484F4D || nextReversed == "MOHD")
                return FileType.Wmo;

            // WMO group has MOGP after MVER
            if (nextMagic == 0x50474F4D || nextReversed == "MOGP")
                return FileType.WmoGroup;

            // ADT has MHDR or MCNK after MVER
            if (nextMagic == 0x5244484D || nextReversed == "MHDR" ||
                nextMagic == 0x4B4E434D || nextReversed == "MCNK")
                return FileType.Adt;

            // Tex ADT has MTEX or MDID after MVER
            if (nextMagic == 0x58455444 || nextReversed == "MTEX" ||
                nextMagic == 0x4449444D || nextReversed == "MDID")
                return FileType.AdtTex;

            // Obj ADT has MMDX or MWMO after MVER
            if (nextMagic == 0x58444D4D || nextReversed == "MMDX" ||
                nextMagic == 0x4F4D574D || nextReversed == "MWMO")
                return FileType.AdtObj;
        }

        // Alpha WDT has different structure (MVER with version 18 for WDT)
        reader.BaseStream.Position = startPos;
        return FileType.Unknown;
    }

    /// <summary>
    /// Detect WoW version from ADT features.
    /// </summary>
    public static WowVersion DetectAdtVersion(BinaryReader reader)
    {
        var startPos = reader.BaseStream.Position;
        bool hasMh2o = false;
        bool hasMccv = false;
        bool hasMcbb = false;
        uint mverVersion = 0;

        while (reader.BaseStream.Position < reader.BaseStream.Length - 8)
        {
            var chunkId = reader.ReadUInt32();
            var chunkSize = reader.ReadUInt32();
            var nextPos = reader.BaseStream.Position + chunkSize;

            switch (chunkId)
            {
                case 0x5245564D: // MVER
                    mverVersion = reader.ReadUInt32();
                    break;
                case 0x4F32484D: // MH2O (WotLK+)
                    hasMh2o = true;
                    break;
                case 0x5643434D: // MCCV (WotLK+)
                    hasMccv = true;
                    break;
                case 0x4242434D: // MCBB (MoP+)
                    hasMcbb = true;
                    break;
            }

            if (reader.BaseStream.Position < nextPos)
                reader.BaseStream.Position = nextPos;
        }

        reader.BaseStream.Position = startPos;

        // Alpha uses version 18 with monolithic WDT
        if (mverVersion == 18)
        {
            if (hasMcbb)
                return WowVersion.Mop;
            if (hasMh2o || hasMccv)
                return WowVersion.Wotlk;
            // Need more context to distinguish Classic/TBC
            return WowVersion.Classic;
        }

        return WowVersion.Unknown;
    }

    /// <summary>
    /// Detect WMO version from header.
    /// </summary>
    public static int DetectWmoVersion(BinaryReader reader)
    {
        var startPos = reader.BaseStream.Position;

        // Read MVER
        var magic = reader.ReadUInt32();
        if (magic != 0x5245564D) // MVER
        {
            reader.BaseStream.Position = startPos;
            return -1;
        }

        var size = reader.ReadUInt32();
        var version = reader.ReadUInt32();

        reader.BaseStream.Position = startPos;
        return (int)version;
    }

    /// <summary>
    /// Check if this is a split ADT (Cata+) by looking for companion files.
    /// </summary>
    public static bool IsSplitAdt(string adtPath)
    {
        var basePath = adtPath;
        
        // Remove extension
        if (adtPath.EndsWith(".adt", StringComparison.OrdinalIgnoreCase))
            basePath = adtPath[..^4];

        // Check for companion files
        var texPath = basePath + "_tex0.adt";
        var objPath = basePath + "_obj0.adt";

        return File.Exists(texPath) || File.Exists(objPath);
    }

    /// <summary>
    /// Check if this is an Alpha monolithic WDT.
    /// </summary>
    public static bool IsAlphaWdt(BinaryReader reader)
    {
        var startPos = reader.BaseStream.Position;

        // Read through looking for MAIN chunk with Alpha structure
        while (reader.BaseStream.Position < reader.BaseStream.Length - 8)
        {
            var magic = reader.ReadUInt32();
            var size = reader.ReadUInt32();

            // Alpha MAIN has 16-byte entries with MHDR offsets
            if (magic == 0x4E49414D) // MAIN
            {
                reader.BaseStream.Position = startPos;
                // Alpha MAIN size is 64*64*16 = 65536
                return size == 0x10000;
            }

            reader.BaseStream.Position += size;
        }

        reader.BaseStream.Position = startPos;
        return false;
    }
}
