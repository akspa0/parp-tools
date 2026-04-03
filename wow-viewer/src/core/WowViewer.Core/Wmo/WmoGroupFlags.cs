namespace WowViewer.Core.Wmo;

[Flags]
public enum WmoGroupFlags : uint
{
    None = 0,
    HasBspChunks = 0x00000001,
    IsExterior = 0x00000008,
    HasVertexColorChunk = 0x00000004,
    UsesExteriorLighting = 0x00000040,
    HasLightRefChunk = 0x00000200,
    HasMpbChunks = 0x00000400,
    HasDoodadRefChunk = 0x00000800,
    HasLiquidChunk = 0x00001000,
    HasMoriMorbChunks = 0x00020000,
    HasSecondaryVertexColorChunk = 0x01000000,
    HasSecondaryUvSet = 0x02000000,
    HasTertiaryUvSet = 0x40000000,
    AllKnown = HasBspChunks
        | IsExterior
        | HasVertexColorChunk
        | UsesExteriorLighting
        | HasLightRefChunk
        | HasMpbChunks
        | HasDoodadRefChunk
        | HasLiquidChunk
        | HasMoriMorbChunks
        | HasSecondaryVertexColorChunk
        | HasSecondaryUvSet
        | HasTertiaryUvSet,
}