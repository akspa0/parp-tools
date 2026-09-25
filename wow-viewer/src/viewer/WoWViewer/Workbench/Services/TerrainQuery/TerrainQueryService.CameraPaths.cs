using System.Numerics;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;
using WowViewer.Core.M2;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using Silk.NET.Input;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// TerrainQueryService: members moved from ViewerApp_CameraPaths.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class TerrainQueryService
{

    private void DrawCameraPathOverlay(Terrain.BoundingBoxRenderer overlay)
    {
        if (!_showCameraPathOverlay || _cameraPath.Keyframes.Count == 0)
            return;

        M2CameraPathEvaluator.NormalizeAndValidate(_cameraPath);
        const int sampleCount = 96;
        const float pathHeight = 0.35f;
        Vector3 pathColor = new(1f, 0.65f, 0.1f);
        Vector3 targetColor = new(0.35f, 0.85f, 1f);
        Vector3 keyColor = new(1f, 0.25f, 0.1f);

        if (_cameraPath.Keyframes.Count > 1)
        {
            Vector3 previous = M2CameraPathEvaluator.Sample(_cameraPath, 0).Position;
            for (int index = 1; index <= sampleCount; index++)
            {
                int time = (int)MathF.Round(_cameraPath.DurationMs * (index / (float)sampleCount));
                Vector3 current = M2CameraPathEvaluator.Sample(_cameraPath, time).Position;
                overlay.BatchLine(previous, current, pathColor);
                previous = current;
            }
        }

        foreach (M2CameraPathKeyframe key in _cameraPath.Keyframes)
        {
            overlay.BatchOctahedron(key.Position, pathHeight * 3f, keyColor);
            overlay.BatchLine(key.Position, key.Target, targetColor);
        }
    }
}
