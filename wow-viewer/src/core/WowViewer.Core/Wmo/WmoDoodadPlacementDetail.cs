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
        ModelKind = ClassifyModelKind(modelPath);
    }

    public int PlacementIndex { get; }

    public uint NameIndex { get; }

    public string ModelPath { get; }

    public Vector3 Position { get; }

    public Quaternion Rotation { get; }

    public float Scale { get; }

    public uint ColorBgra { get; }

    public byte Alpha => (byte)((ColorBgra >> 24) & 0xFF);

    public WmoDoodadModelKind ModelKind { get; }

    private static WmoDoodadModelKind ClassifyModelKind(string modelPath)
    {
        string extension = Path.GetExtension(modelPath);
        if (extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase))
            return WmoDoodadModelKind.Mdx;

        if (extension.Equals(".m2", StringComparison.OrdinalIgnoreCase))
            return WmoDoodadModelKind.M2;

        return WmoDoodadModelKind.Unknown;
    }
}