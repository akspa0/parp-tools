namespace WowViewer.Core.Wmo;

public sealed record WmoDetailDoodadEntry(uint DoodadId, byte Weight);

public sealed record WmoDetailDoodadLayer(byte Density, IReadOnlyList<WmoDetailDoodadEntry> Doodads);

public sealed record WmoDetailDoodadDecodedCommand(
    ushort LayerIndex,
    ushort BatchIndex,
    bool RollAllLocations,
    int LocRangeIndex,
    bool SingleLocation,
    IReadOnlyList<int> Locations);

public sealed record WmoGroupDetailDoodadData(
    ushort GroupIndex,
    byte[] RawData,
    IReadOnlyList<WmoDetailDoodadDecodedCommand> Commands);

public sealed record WmoDetailDoodadDocument(
    float MinTriangleArea,
    IReadOnlyList<WmoDetailDoodadLayer> Layers,
    IReadOnlyList<WmoGroupDetailDoodadData> Groups);
