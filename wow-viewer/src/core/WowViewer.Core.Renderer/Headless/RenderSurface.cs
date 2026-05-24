using Silk.NET.OpenGL;

namespace WowViewer.Core.Renderer.Headless;

public sealed class RenderSurface : IDisposable
{
    private readonly GL _gl;
    private uint _framebuffer;
    private uint _colorTexture;
    private uint _depthRenderbuffer;
    private int _width;
    private int _height;
    private bool _disposed;

    public RenderSurface(GL gl, int width, int height)
    {
        _gl = gl ?? throw new ArgumentNullException(nameof(gl));
        _width = width;
        _height = height;
        CreateFramebuffer();
    }

    public uint Framebuffer => _framebuffer;
    public uint ColorTexture => _colorTexture;
    public int Width => _width;
    public int Height => _height;

    private unsafe void CreateFramebuffer()
    {
        _framebuffer = _gl.GenFramebuffer();
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _framebuffer);

        _colorTexture = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, _colorTexture);
        _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba8, (uint)_width, (uint)_height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, null);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, TextureTarget.Texture2D, _colorTexture, 0);

        _depthRenderbuffer = _gl.GenRenderbuffer();
        _gl.BindRenderbuffer(RenderbufferTarget.Renderbuffer, _depthRenderbuffer);
        _gl.RenderbufferStorage(RenderbufferTarget.Renderbuffer, InternalFormat.DepthComponent24, (uint)_width, (uint)_height);
        _gl.FramebufferRenderbuffer(FramebufferTarget.Framebuffer, FramebufferAttachment.DepthAttachment, RenderbufferTarget.Renderbuffer, _depthRenderbuffer);

        if (_gl.CheckFramebufferStatus(FramebufferTarget.Framebuffer) != GLEnum.FramebufferComplete)
            throw new InvalidOperationException("Framebuffer is not complete");

        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
    }

    public unsafe void Resize(int width, int height)
    {
        _width = width;
        _height = height;
        DestroyFramebuffer();
        CreateFramebuffer();
    }

    public void Bind()
    {
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _framebuffer);
        _gl.Viewport(0, 0, (uint)_width, (uint)_height);
    }

    public void Unbind()
    {
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
    }

    public void Clear(float r, float g, float b, float a)
    {
        Bind();
        _gl.ClearColor(r, g, b, a);
        _gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
    }

    private void DestroyFramebuffer()
    {
        if (_colorTexture != 0)
        {
            _gl.DeleteTexture(_colorTexture);
            _colorTexture = 0;
        }
        if (_depthRenderbuffer != 0)
        {
            _gl.DeleteRenderbuffer(_depthRenderbuffer);
            _depthRenderbuffer = 0;
        }
        if (_framebuffer != 0)
        {
            _gl.DeleteFramebuffer(_framebuffer);
            _framebuffer = 0;
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        DestroyFramebuffer();
    }
}
