namespace WowViewer.Core.Blp;

public enum BlpPixelFormat : uint
{
    Dxt1 = 0,
    Dxt3 = 1,
    Argb8888 = 2,
    PalArgb1555DitherFloydSteinberg = 3,
    PalArgb4444DitherFloydSteinberg = 4,
    PalArgb565DitherFloydSteinberg = 5,
    Dxt5 = 7,
    Palettized = 8,
    PalArgb2565DitherFloydSteinberg = 9,
}