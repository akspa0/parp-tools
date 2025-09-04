namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] Minimal equivalent of ChunkHeaders.h - common FourCC constants. Extend as needed.
/// </summary>
public static class ChunkHeaders
{
    public const string MVER = "MVER";
    public const string MPHD = "MPHD";
    public const string MAIN = "MAIN";
    public const string MDNM = "MDNM";
    public const string MONM = "MONM";
    public const string MWMO = "MWMO";
    public const string MODF = "MODF";

    public const string MHDR = "MHDR";
    public const string MCIN = "MCIN";
    public const string MCAL = "MCAL";
    public const string MCRF = "MCRF";
    public const string MDDF = "MDDF";
    public const string MH2O = "MH2O";
    public const string MMDX = "MMDX";
    public const string MMID = "MMID";

    // ADT top-level (LK)
    public const string MTEX = "MTEX";
    public const string MFBO = "MFBO";
    public const string MTXF = "MTXF";

    // MCNK subchunks
    public const string MCVT = "MCVT";
    public const string MCNR = "MCNR";
    public const string MCLY = "MCLY";
    public const string MCSH = "MCSH";
    public const string MCSE = "MCSE";
    public const string MCCV = "MCCV";
    public const string MCLV = "MCLV";
    public const string MCLQ = "MCLQ";
}

/// <summary>
/// [PORT] C# port of McnkHeader from ChunkHeaders.h
/// Header structure for Lich King MCNK chunks
/// </summary>
public struct McnkHeader
{
    public int Flags;
    public int IndexX;
    public int IndexY;
    public int NLayers;
    public int M2Number;
    public int McvtOffset;
    public int McnrOffset;
    public int MclyOffset;
    public int McrfOffset;
    public int McalOffset;
    public int McalSize;
    public int McshOffset;
    public int McshSize;
    public int AreaId;
    public int WmoNumber;
    public int Holes;
    public int GroundEffectsMap1;
    public int GroundEffectsMap2;
    public int GroundEffectsMap3;
    public int GroundEffectsMap4;
    public int PredTex;
    public int NEffectDoodad;
    public int McseOffset;
    public int NSndEmitters;
    public int MclqOffset;
    public int MclqSize;
    public float PosX;
    public float PosY;
    public float PosZ;
    public int MccvOffset;
    public int MclvOffset;
    public int Unused;
}

/// <summary>
/// [PORT] C# port of McnkAlphaHeader from ChunkHeaders.h
/// Header structure for Alpha MCNK chunks
/// </summary>
public struct McnkAlphaHeader
{
    public int Flags;
    public int IndexX;
    public int IndexY;
    public float Unknown1;
    public int NLayers;
    public int M2Number;
    public int McvtOffset;
    public int McnrOffset;
    public int MclyOffset;
    public int McrfOffset;
    public int McalOffset;
    public int McalSize;
    public int McshOffset;
    public int McshSize;
    public int Unknown3;
    public int WmoNumber;
    public int Holes;
    public int GroundEffectsMap1;
    public int GroundEffectsMap2;
    public int GroundEffectsMap3;
    public int GroundEffectsMap4;
    public int Unknown6;
    public int Unknown7;
    public int McnkChunksSize;
    public int Unknown8;
    public int MclqOffset;
    public int Unused1;
    public int Unused2;
    public int Unused3;
    public int Unused4;
    public int Unused5;
    public int Unused6;
}
