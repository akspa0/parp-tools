using Silk.NET.OpenGL;

namespace WowViewer.Core.Renderer.Headless;

public sealed class FrameCapture
{
    private readonly GL _gl;
    private readonly int _width;
    private readonly int _height;

    public FrameCapture(HeadlessContext context, int width, int height)
    {
        _gl = context.GL;
        _width = width;
        _height = height;
    }

    public unsafe byte[] CaptureRgba()
    {
        byte[] pixels = new byte[_width * _height * 4];
        fixed (byte* ptr = pixels)
            _gl.ReadPixels(0, 0, (uint)_width, (uint)_height, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
        FlipY(pixels);
        return pixels;
    }

    private void FlipY(byte[] pixels)
    {
        int rowSize = _width * 4;
        byte[] row = new byte[rowSize];
        for (int y = 0; y < _height / 2; y++)
        {
            int top = y * rowSize;
            int bottom = (_height - 1 - y) * rowSize;
            System.Buffer.BlockCopy(pixels, top, row, 0, rowSize);
            System.Buffer.BlockCopy(pixels, bottom, pixels, top, rowSize);
            System.Buffer.BlockCopy(row, 0, pixels, bottom, rowSize);
        }
    }
}
