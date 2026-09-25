using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Catalog;
using WoWViewer.Capture;
using WoWViewer.Population;
using WoWViewer.Terrain;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// Pure scene/viewport math: ray-AABB intersection, point-segment distance, world-to-viewport/screen projection.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01); stateless, no host access.
/// </summary>
internal static class SceneViewportMath
{

    internal static float DistanceSquaredPointToSegment(Vector2 point, Vector2 start, Vector2 end)
    {
        Vector2 segment = end - start;
        float segmentLengthSq = segment.LengthSquared();
        if (segmentLengthSq <= 0.0001f)
            return Vector2.DistanceSquared(point, start);

        float t = Vector2.Dot(point - start, segment) / segmentLengthSq;
        t = Math.Clamp(t, 0f, 1f);
        Vector2 closest = start + segment * t;
        return Vector2.DistanceSquared(point, closest);
    }

    internal static float RayAabbIntersect(Vector3 origin, Vector3 dir, Vector3 boundsMin, Vector3 boundsMax)
    {
        float tmin = 0f;
        float tmax = float.MaxValue;

        if (!UpdateRayAabbInterval(origin.X, dir.X, boundsMin.X, boundsMax.X, ref tmin, ref tmax)
            || !UpdateRayAabbInterval(origin.Y, dir.Y, boundsMin.Y, boundsMax.Y, ref tmin, ref tmax)
            || !UpdateRayAabbInterval(origin.Z, dir.Z, boundsMin.Z, boundsMax.Z, ref tmin, ref tmax))
        {
            return -1f;
        }

        return tmin >= 0f ? tmin : tmax >= 0f ? tmax : -1f;
    }

    private static bool UpdateRayAabbInterval(float origin, float direction, float min, float max, ref float tmin, ref float tmax)
    {
        if (MathF.Abs(direction) < 0.0001f)
            return origin >= min && origin <= max;

        float invDir = 1f / direction;
        float t1 = (min - origin) * invDir;
        float t2 = (max - origin) * invDir;
        if (t1 > t2)
            (t1, t2) = (t2, t1);

        tmin = MathF.Max(tmin, t1);
        tmax = MathF.Min(tmax, t2);
        return tmax >= tmin;
    }

    internal static bool TryProjectWorldToViewport(Vector3 worldPosition, Matrix4x4 view, Matrix4x4 proj, float viewportWidth, float viewportHeight, out Vector2 projected)
    {
        Vector4 clip = Vector4.Transform(Vector4.Transform(new Vector4(worldPosition, 1f), view), proj);
        if (clip.W <= 0.0001f)
        {
            projected = Vector2.Zero;
            return false;
        }

        Vector3 ndc = new Vector3(clip.X, clip.Y, clip.Z) / clip.W;
        if (ndc.Z < -1f || ndc.Z > 1f)
        {
            projected = Vector2.Zero;
            return false;
        }

        projected = new Vector2(
            (ndc.X * 0.5f + 0.5f) * viewportWidth,
            (1f - (ndc.Y * 0.5f + 0.5f)) * viewportHeight);
        return true;
    }

    private static bool TryProjectToScreen(Vector3 worldPos, Matrix4x4 viewProj, int screenW, int screenH, out float sx, out float sy)
    {
        var clip = Vector4.Transform(new Vector4(worldPos, 1f), viewProj);
        if (clip.W <= 0) { sx = sy = 0; return false; }
        float ndcX = clip.X / clip.W;
        float ndcY = clip.Y / clip.W;
        sx = (ndcX * 0.5f + 0.5f) * screenW;
        sy = (1f - (ndcY * 0.5f + 0.5f)) * screenH;
        return true;
    }
}
