namespace GillijimProject.WowFiles;

public static class Tags
{
    public const uint MVER = 0x4D564552; // 'MVER' (FourCcToDisplay output)
    public const uint MPHD = 0x4D504844; // 'MPHD' (FourCcToDisplay output)  
    public const uint MAIN = 0x4D41494E; // 'MAIN' (FourCcToDisplay output)
    public const uint MHDR = 0x4D484452; // 'MHDR' (FourCcToDisplay output)
    public const uint MCIN = 0x4D43494E; // 'MCIN' (FourCcToDisplay output)
    public const uint MCNK = 0x4D434E4B; // 'MCNK' (FourCcToDisplay output)
    public const uint MDNM = 0x4D444E4D; // 'MDNM' (FourCcToDisplay output)
    public const uint MONM = 0x4D4F4E4D; // 'MONM' (FourCcToDisplay output)
    public const uint MTEX = 0x4D544558; // 'MTEX' (FourCcToDisplay output)
    public const uint MCLY = 0x4D434C59; // 'MCLY' (FourCcToDisplay output)
    public const uint MCRF = 0x4D435246; // 'MCRF' (FourCcToDisplay output)
    public const uint MCSH = 0x4D435348; // 'MCSH' (FourCcToDisplay output)
    public const uint MCAL = 0x4D43414C; // 'MCAL' (FourCcToDisplay output)
    
    // Terrain subchunks
    public const uint MCVT = 0x4D435654; // 'MCVT' (FourCcToDisplay output)
    public const uint MCNR = 0x4D434E52; // 'MCNR' (FourCcToDisplay output)
    public const uint MCCV = 0x4D434356; // 'MCCV' (FourCcToDisplay output)
    public const uint MCLQ = 0x4D434C51; // 'MCLQ' (FourCcToDisplay output)
    public const uint MCSE = 0x4D435345; // 'MCSE' (FourCcToDisplay output)
    public const uint MCBB = 0x4D434242; // 'MCBB' (FourCcToDisplay output)
    
    // World object chunks
    public const uint MMDX = 0x4D4D4458; // 'MMDX' (FourCcToDisplay output)
    public const uint MMID = 0x4D4D4944; // 'MMID' (FourCcToDisplay output)
    public const uint MWMO = 0x4D574D4F; // 'MWMO' (FourCcToDisplay output)
    public const uint MWID = 0x4D574944; // 'MWID' (FourCcToDisplay output)
    public const uint MDDF = 0x4D444446; // 'MDDF' (FourCcToDisplay output)
    public const uint MODF = 0x4D4F4446; // 'MODF' (FourCcToDisplay output)
}

public readonly record struct ChunkHeader(uint Tag, uint Size, long PayloadOffset);
