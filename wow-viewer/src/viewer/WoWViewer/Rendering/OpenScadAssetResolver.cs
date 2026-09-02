using System;
using System.IO;
using Silk.NET.OpenGL;
using WoWViewer.Logging;

namespace WoWViewer.Rendering;

/// <summary>
/// Resolves and loads authored OpenSCAD assets (.off / .scad) stored in the project source tree or build output.
/// </summary>
public static class OpenScadAssetResolver
{
    /// <summary>
    /// Searches for an authored OpenSCAD asset file across standard build output and source directories.
    /// </summary>
    public static string? FindAssetPath(string filename)
    {
        string[] searchDirs = {
            Path.Combine(AppContext.BaseDirectory, "Assets", "OpenScad"),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Assets", "OpenScad")),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "src", "viewer", "WoWViewer", "Assets", "OpenScad")),
            Path.Combine(Directory.GetCurrentDirectory(), "Assets", "OpenScad")
        };

        foreach (var dir in searchDirs)
        {
            if (Directory.Exists(dir))
            {
                string candidate = Path.Combine(dir, filename);
                if (File.Exists(candidate))
                    return candidate;
            }
        }
        return null;
    }

    /// <summary>
    /// Attempts to load an authored OpenSCAD .off asset into a GPU mesh.
    /// </summary>
    public static ProceduralMesh? TryLoadMesh(GL gl, string filename)
    {
        string? path = FindAssetPath(filename);
        if (path != null && File.Exists(path))
        {
            try
            {
                using var reader = new StreamReader(path);
                var mesh = ProceduralMeshLoader.LoadFromOff(gl, reader);
                ViewerLog.Info(ViewerLog.Category.General, $"Loaded authored OpenSCAD 3D asset: {path} ({mesh.IndexCount / 3} triangles)");
                return mesh;
            }
            catch (Exception ex)
            {
                ViewerLog.Important(ViewerLog.Category.General, $"Failed to load OpenSCAD mesh from {path}: {ex.Message}");
            }
        }
        return null;
    }
}
