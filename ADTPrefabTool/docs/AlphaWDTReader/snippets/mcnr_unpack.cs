// docs/AlphaWDTReader/snippets/mcnr_unpack.cs
// Purpose: Unpack MCNR signed-byte normals to floats with scale; optional normalization.

using System;

namespace Snippets
{
    public static class McnrUnpack
    {
        // scale typically ~1/127f; verify per format variant
        public static void Unpack(ReadOnlySpan<sbyte> src, Span<float> dst, float scale = 1.0f/127.0f, bool normalize = false)
        {
            if (src.Length % 3 != 0) throw new ArgumentException("src len not multiple of 3");
            if (dst.Length < src.Length) throw new ArgumentException("dst too small");

            for (int i = 0; i < src.Length; i+=3)
            {
                float x = src[i+0] * scale;
                float y = src[i+1] * scale;
                float z = src[i+2] * scale;
                if (normalize)
                {
                    float len = MathF.Sqrt(x*x+y*y+z*z);
                    if (len > 1e-5f) { x/=len; y/=len; z/=len; }
                }
                dst[i+0]=x; dst[i+1]=y; dst[i+2]=z;
            }
        }
    }
}
