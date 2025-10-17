using System;
using System.Collections.Generic;

namespace WoWRollback.LkToAlphaModule.Models;

public sealed class AlphaWdt
{
    public required string Path { get; init; }
    public AlphaMphd Mphd { get; set; } = new();
    public List<AlphaTile> Tiles { get; } = new();
}

public sealed class AlphaMphd
{
    public int NDoodadNames { get; set; }
    public int OffsDoodadNames { get; set; }
    public int NMapObjNames { get; set; }
    public int OffsMapObjNames { get; set; }
}

public sealed class AlphaTile
{
    public int Index { get; set; }           // 0..4095
    public int MhdrOffset { get; set; }      // absolute file offset to MHDR letters
    public int SizeToFirstMcnk { get; set; }
    public int DataEndOffset { get; set; }
    public AlphaMhdr Mhdr { get; set; } = new();
    public AlphaMcnkHeader? FirstMcnk { get; set; }
    public List<AlphaMcseEntry> Mcse { get; } = new();
}

public sealed class AlphaMhdr
{
    public int OffsInfo { get; set; }
    public int OffsTex { get; set; }
    public int SizeTex { get; set; }
    public int OffsDoo { get; set; }
    public int SizeDoo { get; set; }
    public int OffsMob { get; set; }
    public int SizeMob { get; set; }
}

public sealed class AlphaMcnkHeader
{
    public int Flags { get; set; }
    public int IndexX { get; set; }
    public int IndexY { get; set; }
    public float Radius { get; set; }
    public int NLayers { get; set; }
    public int NDoodadRefs { get; set; }
    public int OffsHeight { get; set; }
    public int OffsNormal { get; set; }
    public int OffsLayer { get; set; }
    public int OffsRefs { get; set; }
    public int OffsAlpha { get; set; }
    public int SizeAlpha { get; set; }
    public int OffsShadow { get; set; }
    public int SizeShadow { get; set; }
    public int AreaId { get; set; }
    public int NMapObjRefs { get; set; }
    public ushort Holes { get; set; }
    public int OffsSndEmitters { get; set; }
    public int NSndEmitters { get; set; }
    public int OffsLiquid { get; set; }
}

public sealed class AlphaMcseEntry
{
    // 0.5.3 76-byte structure mapping (best effort)
    public uint SoundPointId { get; set; }
    public uint SoundNameId { get; set; }
    public float PosX { get; set; }
    public float PosY { get; set; }
    public float PosZ { get; set; }
    public float MinDistance { get; set; }
    public float MaxDistance { get; set; }
    public float CutoffDistance { get; set; }
    public uint StartTime { get; set; }
    public uint EndTime { get; set; }
    public uint Mode { get; set; }
    public uint GroupSilenceMin { get; set; }
    public uint GroupSilenceMax { get; set; }
    public uint PlayInstancesMin { get; set; }
    public uint PlayInstancesMax { get; set; }
    public uint LoopCountMin { get; set; }
    public uint LoopCountMax { get; set; }
    public uint InterSoundGapMin { get; set; }
    public uint InterSoundGapMax { get; set; }
}
