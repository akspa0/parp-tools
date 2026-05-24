using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.Windowing;

namespace WowViewer.Core.Renderer.Headless;

public sealed class HeadlessContext : IDisposable
{
    private readonly IWindow _window;
    private GL _gl = null!;
    private bool _disposed;

    public HeadlessContext(int width = 1024, int height = 1024)
    {
        WindowOptions options = WindowOptions.Default with
        {
            Size = new Vector2D<int>(width, height),
            IsVisible = false,
            IsEventDriven = false,
            API = new GraphicsAPI(ContextAPI.OpenGL, ContextProfile.Core, ContextFlags.ForwardCompatible, new APIVersion(4, 1)),
            WindowBorder = WindowBorder.Hidden,
        };

        _window = Window.Create(options);
        _window.Load += OnLoad;
        _window.Initialize();

        Width = width;
        Height = height;
    }

    public GL GL => _gl ?? throw new InvalidOperationException("GL context not initialized");
    public int Width { get; private set; }
    public int Height { get; private set; }

    private void OnLoad()
    {
        _gl = _window.CreateOpenGL();
    }

    public void Resize(int width, int height)
    {
        Width = width;
        Height = height;
        _window.Size = new Vector2D<int>(width, height);
    }

    public void RenderSingleFrame()
    {
        _window.DoUpdate();
        _window.DoRender();
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _window.Close();
    }
}
