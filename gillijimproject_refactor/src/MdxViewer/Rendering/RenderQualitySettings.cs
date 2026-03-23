using Silk.NET.OpenGL;

namespace MdxViewer.Rendering;

public enum TextureFilteringMode
{
    Nearest = 0,
    Bilinear = 1,
    Trilinear = 2,
}

public static class RenderQualitySettings
{
    public static TextureFilteringMode TextureFilteringMode { get; set; } = TextureFilteringMode.Trilinear;

    public static TextureMinFilter GetMinFilter(bool hasMipmaps)
        => TextureFilteringMode switch
        {
            TextureFilteringMode.Nearest => TextureMinFilter.Nearest,
            TextureFilteringMode.Bilinear => hasMipmaps ? TextureMinFilter.LinearMipmapNearest : TextureMinFilter.Linear,
            _ => hasMipmaps ? TextureMinFilter.LinearMipmapLinear : TextureMinFilter.Linear,
        };

    public static TextureMagFilter GetMagFilter()
        => TextureFilteringMode == TextureFilteringMode.Nearest
            ? TextureMagFilter.Nearest
            : TextureMagFilter.Linear;

    public static void ApplySampling(GL gl, TextureTarget target, bool hasMipmaps, TextureWrapMode wrapS, TextureWrapMode wrapT)
    {
        gl.TexParameter(target, TextureParameterName.TextureMinFilter, (int)GetMinFilter(hasMipmaps));
        gl.TexParameter(target, TextureParameterName.TextureMagFilter, (int)GetMagFilter());
        gl.TexParameter(target, TextureParameterName.TextureWrapS, (int)wrapS);
        gl.TexParameter(target, TextureParameterName.TextureWrapT, (int)wrapT);
    }

    public static string GetLabel(TextureFilteringMode mode)
        => mode switch
        {
            TextureFilteringMode.Nearest => "Nearest",
            TextureFilteringMode.Bilinear => "Bilinear",
            _ => "Trilinear",
        };
}