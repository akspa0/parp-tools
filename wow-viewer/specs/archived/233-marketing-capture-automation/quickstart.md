# Operator Quickstart: Renderer Marketing Capture

This is the intended real-client gate after implementation; it is not proof that the current checkout has produced a promotional video.

1. Supply the licensed encoder and notices required by the existing release-hardening gate, then launch a Debug viewer with a configured client and world.
2. Open **Utilities > Capture > Camera Path**, import `FlybyUndead.mdx` or `.m2` from the loaded client, and confirm the map/build binding.
3. Select **Warm Path** and wait for the bounded preload status to become `ready`.
4. Select the named feature-tour recipe and start **Feature Tour + Video**. Do not manually click the options bar after the command starts.
5. Verify a video and its adjacent `*.tour-receipt.json` under the viewer's managed `output/captures/<map>/<build>/` tree. Read the receipt before treating the run as a benchmark; check its outcome, CPU frame-time distribution, and hitches.
6. If the receipt is completed/degraded and its artifact paths are contained by that output root, create an authoring handoff for the future MCP/ComfyUI workflow. A transport-unavailable response is expected until a concrete MCP schema is registered.
7. Review the video and at least one real still with their receipt before linking them from the GitHub README.

Focused source checks after each implementation task:

```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~MarketingCapture"
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

The operator owns live playback, visual timing, encoder output/playback, renderer performance interpretation, ComfyUI execution, and README publication review.

