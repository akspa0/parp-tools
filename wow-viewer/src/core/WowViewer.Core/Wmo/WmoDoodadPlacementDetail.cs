using System.Numerics;

namespace WowViewer.Core.Wmo;

public sealed class WmoDoodadPlacementDetail
{
    public WmoDoodadPlacementDetail(int placementIndex, uint nameIndex, string modelPath, Vector3 position, Quaternion rotation, float scale, uint colorBgra)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(placementIndex);
        ArgumentNullException.ThrowIfNull(modelPath);

        PlacementIndex = placementIndex;
        NameIndex = nameIndex;
        ModelPath = modelPath;
        Position = position;
        Rotation = rotation;
        Scale = scale;
        ColorBgra = colorBgra;
    }

    public int PlacementIndex { get; }

    public uint NameIndex { get; }

    public string ModelPath { get; }

    public Vector3 Position { get; }

    public Quaternion Rotation { get; }

    public float Scale { get; }

    public uint ColorBgra { get; }

    public byte Alpha => (byte)((ColorBgra >> 24) & 0xFF);
}