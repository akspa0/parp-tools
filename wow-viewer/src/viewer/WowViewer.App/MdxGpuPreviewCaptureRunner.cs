using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.Windowing;

namespace WowViewer.App;

internal static class MdxGpuPreviewCaptureRunner
{
    public static void Capture(MdxPreviewLoadRequest request, string outputPath)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);

        MdxPreviewLoadResult preview = MdxPreviewLoader.Load(request);
        Exception? failure = null;

        WindowOptions options = WindowOptions.Default;
        options.Title = "WoWAlphaViewer MDX GPU Preview Capture";
        options.Size = new Vector2D<int>(request.VisualWidth, request.VisualHeight);
        options.IsVisible = false;
        options.VSync = false;
        options.API = new GraphicsAPI(ContextAPI.OpenGL, ContextProfile.Core, ContextFlags.ForwardCompatible, new APIVersion(3, 3));

        IWindow window = Window.Create(options);
        GL? gl = null;
        MdxGpuPreviewRenderer? renderer = null;
        bool capturePending = true;

        window.Load += () =>
        {
            gl = window.CreateOpenGL();
            renderer = new MdxGpuPreviewRenderer(gl);
            renderer.LoadPreview(preview);
        };

        window.Render += _ =>
        {
            if (!capturePending || renderer == null)
                return;

            try
            {
                renderer.CaptureBmp(outputPath, request.VisualWidth, request.VisualHeight);
            }
            catch (Exception ex)
            {
                failure = ex;
            }
            finally
            {
                capturePending = false;
                window.Close();
            }
        };

        window.Closing += () =>
        {
            renderer?.Dispose();
            gl?.Dispose();
        };

        window.Run();

        if (failure != null)
            throw new InvalidOperationException($"MDX GPU preview capture failed: {failure.Message}", failure);
    }
}
