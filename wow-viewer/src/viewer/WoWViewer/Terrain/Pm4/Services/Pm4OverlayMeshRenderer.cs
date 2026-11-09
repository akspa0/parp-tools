using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using Silk.NET.OpenGL;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using static WoWViewer.Terrain.Pm4OverlayColors;
using WowViewer.Core.PM4.Footprint;

namespace WoWViewer.Terrain.Pm4.Services;

/// <summary>
/// Telemetry metrics returned from each PM4 overlay render pass.
/// </summary>
internal readonly record struct Pm4RenderTimingStats(
    double BoundsMs,
    int BoundsPreparedCount,
    double GeometryPrepareMs,
    double GeometrySubmitMs,
    int GeometryPreparedCount,
    int GeometrySubmittedCount,
    double NodesMs,
    int NodesPreparedCount);

/// <summary>
/// Decomposed mesh rendering service for PM4 overlays (Spec 288 Phase 3, AGENTS.md §10).
/// Manages dedicated GPU VAOs, VBOs, GL shader programs, and per-tile cached buffers
/// to eliminate per-frame CPU vertex matrix transforms for static terrain tiles.
/// </summary>
internal sealed class Pm4OverlayMeshRenderer : IDisposable
{
    private ShaderProgram? _lineShader;
    private ShaderProgram? _solidShader;
    private bool _shadersFailed;

    private readonly Dictionary<(int tileX, int tileY), TileMeshGpuBuffer> _tileBuffers = new();
    private readonly ConcurrentQueue<uint> _vaosToDelete = new();
    private readonly ConcurrentQueue<uint> _vbosToDelete = new();

    private bool _disposed;

    /// <summary>
    /// Renders all visible PM4 overlay geometry, bounds, nodes, and position references.
    /// Replaces inline WorldScene.Render PM4 block.
    /// </summary>
    public void Render(
        GL gl,
        BoundingBoxRenderer bbRenderer,
        Pm4OverlayScene overlay,
        in Matrix4x4 view,
        in Matrix4x4 proj,
        in Vector3 cameraPos,
        out Pm4RenderTimingStats stats)
    {
        // Per-frame visible tallies: reset before accumulating. These were only cleared on
        // load/clear, so the sidebar counters grew without bound every frame.
        overlay._pm4VisibleObjectCount = 0;
        overlay._pm4VisibleLineCount = 0;
        overlay._pm4VisibleTriangleCount = 0;
        overlay._pm4VisiblePositionRefCount = 0;

        if (_disposed || !overlay._showPm4Overlay || overlay._pm4TileObjects.Count == 0)
        {
            stats = default;
            return;
        }

        ProcessPendingDeletions(gl);
        EnsureShaders(gl);

        double boundsMs = 0;
        int boundsPrepared = 0;
        double geomPrepareMs = 0;
        double geomSubmitMs = 0;
        int geomPrepared = 0;
        int geomSubmitted = 0;
        double nodesMs = 0;
        int nodesPrepared = 0;

        var sw = Stopwatch.StartNew();

        // ── 1. Bounds & Placement Markers ─────────────────────────────────────
        if (overlay._showPm4ObjectBounds || overlay._showPm4PlacementZPlane || overlay._showPm4Ck24Bounds || overlay._showPm4ExtrudedWalls)
        {
            sw.Restart();
            RenderBoundsAndMarkers(bbRenderer, overlay, cameraPos, ref boundsPrepared);
            boundsMs = sw.Elapsed.TotalMilliseconds;
        }

        // ── 2. Mesh Geometry Preparation (Tile Buffers) ──────────────────────
        sw.Restart();
        var visibleTiles = new List<((int tileX, int tileY) key, TileMeshGpuBuffer buffer)>();
        Matrix4x4 pm4Transform = overlay.BuildPm4OverlayTransformMatrix();
        bool applyPm4Transform = overlay._pm4OverlayTranslation != Vector3.Zero
            || overlay._pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
            || overlay._pm4OverlayScale != Vector3.One;

        foreach (var (tileKey, objects) in overlay._pm4TileObjects)
        {
            if (!overlay.ShouldRenderPm4Tile(tileKey.tileX, tileKey.tileY))
                continue;

            ulong fingerprint = ComputeTileFingerprint(overlay, tileKey, applyPm4Transform, pm4Transform);

            if (!_tileBuffers.TryGetValue(tileKey, out var tileBuffer))
            {
                tileBuffer = new TileMeshGpuBuffer();
                _tileBuffers[tileKey] = tileBuffer;
            }

            if (!tileBuffer.IsValid || tileBuffer.StateFingerprint != fingerprint)
            {
                RebuildTileMeshGpuBuffer(gl, tileBuffer, tileKey, objects, overlay, applyPm4Transform, pm4Transform, fingerprint);
            }

            if (tileBuffer.IsValid && (tileBuffer.LineVertexCount > 0 || tileBuffer.SolidVertexCount > 0))
            {
                visibleTiles.Add((tileKey, tileBuffer));
                overlay._pm4VisibleObjectCount += tileBuffer.ObjectCount;
                overlay._pm4VisibleLineCount += tileBuffer.LineVertexCount / 2;
                overlay._pm4VisibleTriangleCount += tileBuffer.SolidVertexCount / 3;
            }

            // MPRL position refs (interactive pins)
            if (overlay._showPm4PositionRefs && overlay._pm4TilePositionRefs.TryGetValue(tileKey, out var positionRefs) && positionRefs.Count > 0)
            {
                for (int i = 0; i < positionRefs.Count; i++)
                {
                    Vector3 marker = applyPm4Transform ? Pm4OverlayScene.ApplyPm4OverlayTransform(positionRefs[i], pm4Transform) : positionRefs[i];
                    if (!overlay.IsWithinPm4RenderDistance(cameraPos, marker))
                        continue;
                    bbRenderer.BatchPin(marker, 16f, 3f, Pm4ColorMprl);
                    overlay._pm4VisiblePositionRefCount++;
                }
            }
        }

        geomPrepared = overlay._pm4VisibleLineCount + overlay._pm4VisibleTriangleCount;
        geomPrepareMs = sw.Elapsed.TotalMilliseconds;

        // ── 3. Mesh Geometry Draw Submission ──────────────────────────────────
        sw.Restart();
        Matrix4x4 viewProj = view * proj;
        bool ignoreDepth = overlay._pm4OverlayIgnoreDepth;

        // A. Solid Fill Pass
        if (overlay._showPm4SolidOverlay && overlay._pm4VisibleTriangleCount > 0 && _solidShader != null)
        {
            gl.Enable(EnableCap.Blend);
            gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

            if (ignoreDepth)
            {
                gl.Disable(EnableCap.DepthTest);
            }
            else
            {
                gl.Enable(EnableCap.DepthTest);
                gl.DepthFunc(DepthFunction.Lequal);
            }

            gl.DepthMask(false);
            gl.Disable(EnableCap.CullFace);

            _solidShader.Use();
            _solidShader.SetMat4("uVP", viewProj);

            for (int i = 0; i < visibleTiles.Count; i++)
            {
                var buffer = visibleTiles[i].buffer;
                if (buffer.SolidVertexCount > 0 && buffer.SolidVao != 0)
                {
                    gl.BindVertexArray(buffer.SolidVao);
                    gl.DrawArrays(PrimitiveType.Triangles, 0, (uint)buffer.SolidVertexCount);
                }
            }

            gl.BindVertexArray(0);
            gl.Enable(EnableCap.CullFace);
            gl.Disable(EnableCap.Blend);
        }

        // B. Node markers (MSCN / MSPV solid cubes)
        if (overlay._pm4RenderNodesAsCubes && (overlay._showPm4MscnNodes || overlay._showPm4MspvNodes))
        {
            sw.Stop();
            var nodeSw = Stopwatch.StartNew();
            RenderNodeCubes(gl, bbRenderer, overlay, view, proj, cameraPos, ignoreDepth, ref nodesPrepared);
            nodesMs = nodeSw.Elapsed.TotalMilliseconds;
            sw.Start();
        }

        // C. Wireframe Lines Pass
        bool hasLineGeometry = overlay._pm4VisibleLineCount > 0
            || overlay._pm4VisiblePositionRefCount > 0
            || (overlay._showPm4ObjectCentroids && overlay._pm4VisibleObjectCount > 0);

        if (hasLineGeometry)
        {
            gl.LineWidth(overlay._pm4WireframeLineWidth);

            if (ignoreDepth)
            {
                gl.Disable(EnableCap.DepthTest);
            }
            else
            {
                gl.Enable(EnableCap.DepthTest);
                gl.DepthFunc(DepthFunction.Lequal);
            }

            // Draw cached wireframes
            if (_lineShader != null)
            {
                _lineShader.Use();
                _lineShader.SetMat4("uVP", viewProj);

                for (int i = 0; i < visibleTiles.Count; i++)
                {
                    var buffer = visibleTiles[i].buffer;
                    if (buffer.LineVertexCount > 0 && buffer.LineVao != 0)
                    {
                        gl.BindVertexArray(buffer.LineVao);
                        gl.DrawArrays(PrimitiveType.Lines, 0, (uint)buffer.LineVertexCount);
                    }
                }
                gl.BindVertexArray(0);
            }

            // Centroid pins (if enabled)
            if (overlay._showPm4ObjectCentroids && overlay._pm4VisibleObjectCount > 0)
            {
                for (int i = 0; i < visibleTiles.Count; i++)
                {
                    var (tileKey, buffer) = visibleTiles[i];
                    if (overlay._pm4TileObjects.TryGetValue(tileKey, out var objects))
                    {
                        for (int o = 0; o < objects.Count; o++)
                        {
                            var obj = objects[o];
                            if (overlay.ShouldRenderPm4Object(obj))
                                bbRenderer.BatchPin(obj.Center, 22f, 4f, Pm4ColorCentroid);
                        }
                    }
                }
            }

            // Flush dynamic wireframe pins (MPRL refs / centroids)
            gl.DepthMask(false);
            bbRenderer.FlushBatch(view, proj);
            gl.LineWidth(1.0f);
        }

        // Reset OpenGL state
        gl.Enable(EnableCap.DepthTest);
        gl.DepthFunc(DepthFunction.Lequal);
        gl.DepthMask(true);
        gl.Disable(EnableCap.Blend);

        bbRenderer.BeginBatch();
        bbRenderer.BeginSolidBatch();

        geomSubmitted = geomPrepared;
        geomSubmitMs = sw.Elapsed.TotalMilliseconds;

        stats = new Pm4RenderTimingStats(
            boundsMs,
            boundsPrepared,
            geomPrepareMs,
            geomSubmitMs,
            geomPrepared,
            geomSubmitted,
            nodesMs,
            nodesPrepared);
    }

    private void RenderBoundsAndMarkers(
        BoundingBoxRenderer bbRenderer,
        Pm4OverlayScene overlay,
        in Vector3 cameraPos,
        ref int boundsPrepared)
    {
        Matrix4x4 pm4Transform = overlay.BuildPm4OverlayTransformMatrix();
        bool applyPm4Transform = overlay._pm4OverlayTranslation != Vector3.Zero
            || overlay._pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
            || overlay._pm4OverlayScale != Vector3.One;

        if (overlay._showPm4ObjectBounds)
        {
            foreach (var (tileKey, objects) in overlay._pm4TileObjects)
            {
                if (!overlay.ShouldRenderPm4Tile(tileKey.tileX, tileKey.tileY))
                    continue;

                foreach (Pm4OverlayObject obj in objects)
                {
                    if (!overlay.ShouldRenderPm4Object(obj))
                        continue;

                    var objectKey = (tileKey.tileX, tileKey.tileY, obj.Ck24, obj.ObjectPartId);
                    Matrix4x4 objectTransform = overlay.BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);
                    if (!overlay.ShouldRenderPm4Object(obj, objectTransform, applyObjectTransform, cameraPos, out _))
                        continue;

                    WorldScene.TransformBounds(obj.BoundsMin, obj.BoundsMax, Pm4OverlayScene.BuildPm4GeometryTransform(obj, objectTransform, applyObjectTransform), out Vector3 boundsMin, out Vector3 boundsMax);

                    Vector3 boxColor = Pm4ColorObjectBounds;
                    if (overlay._highlightedPm4ObjectKeys.Contains(objectKey))
                        boxColor = Pm4ColorHighlight;
                    if (overlay.IsHullInSelectedCompound(objectKey))
                        boxColor = Pm4ColorSelection;
                    else if (overlay._selectedPm4ObjectKey.HasValue && overlay._selectedPm4ObjectKey.Value == objectKey)
                        boxColor = Pm4ColorSelection;

                    bbRenderer.BatchBoxMinMax(boundsMin, boundsMax, boxColor);
                    boundsPrepared++;
                }
            }
        }


        if (overlay._showPm4PlacementZPlane)
        {
            foreach (var tileEntry in overlay._pm4TileObjects)
            {
                if (!overlay.ShouldRenderPm4Tile(tileEntry.Key.tileX, tileEntry.Key.tileY))
                    continue;

                foreach (Pm4OverlayObject obj in tileEntry.Value)
                {
                    if (!overlay.ShouldRenderPm4Object(obj) || obj.Ck24 == 0)
                        continue;

                    var thisKey = (tileEntry.Key.tileX, tileEntry.Key.tileY, obj.Ck24, obj.ObjectPartId);
                    if (!overlay._showPm4PlacementZForAllObjects)
                    {
                        if (!overlay._selectedPm4ObjectKey.HasValue || overlay._selectedPm4ObjectKey.Value != thisKey)
                            continue;
                    }

                    Matrix4x4 objTransform = overlay.BuildPm4ObjectTransform(thisKey, applyPm4Transform, pm4Transform, out bool applyObj);
                    if (!overlay.ShouldRenderPm4Object(obj, objTransform, applyObj, cameraPos, out _))
                        continue;

                    WorldScene.TransformBounds(obj.BoundsMin, obj.BoundsMax, Pm4OverlayScene.BuildPm4GeometryTransform(obj, objTransform, applyObj), out Vector3 bMin, out Vector3 bMax);

                    float z = BitConverter.UInt32BitsToSingle(obj.Ck24 << 8);
                    Vector3 centre = (bMin + bMax) * 0.5f;
                    const float MarkerHalfWidth = 1.25f;
                    Vector3 min = new(centre.X - MarkerHalfWidth, centre.Y - MarkerHalfWidth, z - 0.1f);
                    Vector3 max = new(centre.X + MarkerHalfWidth, centre.Y + MarkerHalfWidth, z + 0.1f);
                    bbRenderer.BatchBoxMinMax(min, max, new Vector3(0.35f, 0.95f, 0.45f));
                    boundsPrepared++;
                }
            }
        }

        if (overlay._showPm4Ck24Bounds)
        {
            var ck24Groups = new Dictionary<(int tileX, int tileY, uint ck24), (Vector3 min, Vector3 max, byte ck24Type)>();
            foreach (var (tileKey, objects) in overlay._pm4TileObjects)
            {
                if (!overlay.ShouldRenderPm4Tile(tileKey.tileX, tileKey.tileY))
                    continue;

                foreach (Pm4OverlayObject obj in objects)
                {
                    if (!overlay.ShouldRenderPm4ObjectType(obj.Ck24Type))
                        continue;

                    var ck24Key = (tileKey.tileX, tileKey.tileY, obj.Ck24);
                    var objectKey = (tileKey.tileX, tileKey.tileY, obj.Ck24, obj.ObjectPartId);
                    Matrix4x4 objectTransform = overlay.BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);
                    if (!overlay.ShouldRenderPm4Object(obj, objectTransform, applyObjectTransform, cameraPos, out _))
                        continue;

                    WorldScene.TransformBounds(obj.BoundsMin, obj.BoundsMax, Pm4OverlayScene.BuildPm4GeometryTransform(obj, objectTransform, applyObjectTransform), out Vector3 boundsMin, out Vector3 boundsMax);

                    // When Ck24 == 0 (M2 doodads), render individual object bounds.
                    // Grouping by Ck24 is only valid for non-zero Ck24 (WMOs), where Ck24 is a unique building identity.
                    if (obj.Ck24 == 0)
                    {
                        bbRenderer.BatchBoxMinMax(boundsMin, boundsMax, new Vector3(0.65f, 0.65f, 0.75f));
                        boundsPrepared++;
                        continue;
                    }

                    if (ck24Groups.TryGetValue(ck24Key, out var existing))
                    {
                        ck24Groups[ck24Key] = (Vector3.Min(existing.min, boundsMin), Vector3.Max(existing.max, boundsMax), existing.ck24Type);
                    }
                    else
                    {
                        ck24Groups[ck24Key] = (boundsMin, boundsMax, obj.Ck24Type);
                    }
                }
            }

            foreach (var (_, (boundsMin, boundsMax, ck24Type)) in ck24Groups)
            {
                Vector3 boxColor = ck24Type switch
                {
                    0x00 => new Vector3(0.65f, 0.65f, 0.75f),
                    0x40 or 0x41 => new Vector3(1.00f, 0.95f, 0.65f),
                    0x42 or 0x43 => new Vector3(0.65f, 0.95f, 0.95f),
                    0xC0 or 0xC1 or 0xC2 or 0xC3 => new Vector3(1.00f, 0.75f, 0.60f),
                    _ => new Vector3(0.80f, 0.80f, 0.80f)
                };
                bbRenderer.BatchBoxMinMax(boundsMin, boundsMax, boxColor);
                boundsPrepared++;
            }
        }

        if (overlay._showPm4ExtrudedWalls)
        {
            foreach (var (tileKey, objects) in overlay._pm4TileObjects)
            {
                if (!overlay.ShouldRenderPm4Tile(tileKey.tileX, tileKey.tileY))
                    continue;

                // Group by discrete object: non-zero Ck24 groups all parts of the same WMO,
                // while zero Ck24 groups by dominant group or part id.
                var objectGroups = new Dictionary<string, List<Pm4OverlayObject>>();
                foreach (Pm4OverlayObject obj in objects)
                {
                    if (!overlay.ShouldRenderPm4ObjectType(obj.Ck24Type))
                        continue;

                    string groupKey = obj.Ck24 != 0
                        ? $"ck24_{obj.Ck24:X6}"
                        : $"part_{obj.ObjectPartId}";

                    if (!objectGroups.TryGetValue(groupKey, out var groupList))
                    {
                        groupList = new List<Pm4OverlayObject>();
                        objectGroups[groupKey] = groupList;
                    }
                    groupList.Add(obj);
                }

                foreach (var (_, groupParts) in objectGroups)
                {
                    Pm4OverlayObject rep = groupParts[0];
                    var repKey = (tileKey.tileX, tileKey.tileY, rep.Ck24, rep.ObjectPartId);
                    Matrix4x4 repObjTransform = overlay.BuildPm4ObjectTransform(repKey, applyPm4Transform, pm4Transform, out bool repApplyObjTransform);
                    if (!overlay.ShouldRenderPm4Object(rep, repObjTransform, repApplyObjTransform, cameraPos, out _))
                        continue;

                    var surfaceVerts = new List<Vector3>();
                    float groupMinZ = float.MaxValue;
                    float groupMaxZ = float.MinValue;

                    for (int p = 0; p < groupParts.Count; p++)
                    {
                        Pm4OverlayObject part = groupParts[p];
                        var partKey = (tileKey.tileX, tileKey.tileY, part.Ck24, part.ObjectPartId);
                        Matrix4x4 partObjTransform = overlay.BuildPm4ObjectTransform(partKey, applyPm4Transform, pm4Transform, out bool partApplyObjTransform);
                        Matrix4x4 geomTransform = Pm4OverlayScene.BuildPm4GeometryTransform(part, partObjTransform, partApplyObjTransform);

                        WorldScene.TransformBounds(part.BoundsMin, part.BoundsMax, geomTransform, out Vector3 partMin, out Vector3 partMax);
                        if (partMin.Z < groupMinZ) groupMinZ = partMin.Z;
                        if (partMax.Z > groupMaxZ) groupMaxZ = partMax.Z;

                        for (int t = 0; t < part.Triangles.Count; t++)
                        {
                            surfaceVerts.Add(Vector3.Transform(part.Triangles[t].A, geomTransform));
                            surfaceVerts.Add(Vector3.Transform(part.Triangles[t].B, geomTransform));
                            surfaceVerts.Add(Vector3.Transform(part.Triangles[t].C, geomTransform));
                        }
                    }

                                        float spanX = MathF.Abs(groupParts.Max(p => p.BoundsMax.X) - groupParts.Min(p => p.BoundsMin.X));
                    float spanY = MathF.Abs(groupParts.Max(p => p.BoundsMax.Y) - groupParts.Min(p => p.BoundsMin.Y));
                    if (surfaceVerts.Count < 3 || groupMaxZ - groupMinZ < 0.05f || spanX > 120f || spanY > 120f)
                        continue;

                    Vector3 wallColor = overlay.GetPm4ObjectColor(tileKey, rep);
                    Pm4FootprintPrism prism = Pm4FootprintExtruder.ExtrudeFootprint(surfaceVerts, null, null, groupMinZ);
                    for (int w = 0; w < prism.Walls.Count; w++)
                    {
                        Pm4ExtrudedWall wall = prism.Walls[w];
                        bbRenderer.BatchLine(wall.TopV0, wall.TopV1, wallColor);
                        bbRenderer.BatchLine(wall.TopV0, wall.BaseV0, wallColor);
                        bbRenderer.BatchLine(wall.BaseV0, wall.BaseV1, wallColor);
                        bbRenderer.BatchLine(wall.TopV1, wall.BaseV1, wallColor);
                        boundsPrepared += 4;
                    }
                }
            }
        }
    }

    private void RenderNodeCubes(
        GL gl,
        BoundingBoxRenderer bbRenderer,
        Pm4OverlayScene overlay,
        in Matrix4x4 view,
        in Matrix4x4 proj,
        in Vector3 cameraPos,
        bool ignoreDepth,
        ref int nodesPrepared)
    {
        if (overlay._showPm4MscnNodes)
            overlay.EnsurePm4MscnData();
        if (overlay._showPm4MspvNodes)
            overlay.EnsurePm4MspvData();

        gl.Enable(EnableCap.Blend);
        gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
        if (ignoreDepth)
            gl.Disable(EnableCap.DepthTest);
        else
        {
            gl.Enable(EnableCap.DepthTest);
            gl.DepthFunc(DepthFunction.Lequal);
        }
        gl.DepthMask(false);
        gl.Disable(EnableCap.CullFace);

        int mscnDrawn = 0;
        if (overlay._showPm4MscnNodes && overlay._pm4TileMscnPoints.Count > 0)
        {
            int limit = 15000;
            foreach (var kv in overlay._pm4TileMscnPoints)
            {
                var pts = kv.Value;
                if (pts == null || !overlay.ShouldRenderPm4Tile(kv.Key.tileX, kv.Key.tileY)) continue;
                for (int i = 0; i < pts.Count && mscnDrawn < limit; i++)
                {
                    if (!overlay.IsWithinPm4RenderDistance(cameraPos, pts[i])) continue;
                    bbRenderer.BatchSolidCube(pts[i], overlay._pm4MscnCubeSize, Pm4ColorMscn, overlay._pm4MscnCubeAlpha);
                    mscnDrawn++;
                }
            }
        }

        int mspvDrawn = 0;
        if (overlay._showPm4MspvNodes && overlay._pm4TileMspvPoints.Count > 0)
        {
            int limit = 8000;
            foreach (var kv in overlay._pm4TileMspvPoints)
            {
                var pts = kv.Value;
                if (pts == null || !overlay.ShouldRenderPm4Tile(kv.Key.tileX, kv.Key.tileY)) continue;
                for (int i = 0; i < pts.Count && mspvDrawn < limit; i++)
                {
                    if (!overlay.IsWithinPm4RenderDistance(cameraPos, pts[i])) continue;
                    bbRenderer.BatchSolidCube(pts[i], overlay._pm4MspvCubeSize, Pm4ColorMspv, overlay._pm4MspvCubeAlpha);
                    mspvDrawn++;
                }
            }
        }

        if (mscnDrawn > 0 || mspvDrawn > 0)
        {
            bbRenderer.FlushSolidBatch(view, proj);
            overlay._pm4VisiblePositionRefCount += mscnDrawn + mspvDrawn;
            nodesPrepared += mscnDrawn + mspvDrawn;
        }

        gl.Enable(EnableCap.CullFace);
        gl.Disable(EnableCap.Blend);
    }

    private static ulong ComputeTileFingerprint(
        Pm4OverlayScene overlay,
        (int tileX, int tileY) tileKey,
        bool applyPm4Transform,
        in Matrix4x4 pm4Transform)
    {
        unchecked
        {
            ulong hash = 14695981039346656037UL;
            hash = (hash ^ (ulong)overlay.Pm4ColorMode) * 1099511628211UL;
            hash = (hash ^ (ulong)tileKey.tileX) * 1099511628211UL;
            hash = (hash ^ (ulong)tileKey.tileY) * 1099511628211UL;
            hash = (hash ^ (ulong)(applyPm4Transform ? 1 : 0)) * 1099511628211UL;
            if (applyPm4Transform)
            {
                hash = (hash ^ (ulong)pm4Transform.M11.GetHashCode()) * 1099511628211UL;
                hash = (hash ^ (ulong)pm4Transform.M41.GetHashCode()) * 1099511628211UL;
            }
            if (overlay.SelectedCompoundObject != null)
            {
                var comp = overlay.SelectedCompoundObject;
                if (comp.SpanningTiles.Contains((tileKey.tileX, tileKey.tileY)) || (comp.TileX == tileKey.tileX && comp.TileY == tileKey.tileY))
                    hash = (hash ^ (ulong)(comp.CompoundId + 1) * 31337UL) * 1099511628211UL;
            }
            if (overlay._selectedPm4ObjectKey.HasValue)
            {
                var sel = overlay._selectedPm4ObjectKey.Value;
                if (sel.tileX == tileKey.tileX && sel.tileY == tileKey.tileY)
                    hash = (hash ^ ((ulong)sel.ck24 * 104729UL ^ (ulong)sel.objectPart)) * 1099511628211UL;
            }
            if (overlay._selectedPm4ObjectGroupKey.HasValue)
            {
                var sel = overlay._selectedPm4ObjectGroupKey.Value;
                if (sel.tileX == tileKey.tileX && sel.tileY == tileKey.tileY)
                    hash = (hash ^ (ulong)sel.ck24) * 1099511628211UL;
            }
            hash = (hash ^ (ulong)overlay._highlightedPm4ObjectKeys.Count) * 1099511628211UL;
            hash = (hash ^ (ulong)(overlay.Pm4ShowPathWalls ? 1 : 0)) * 1099511628211UL;
            return hash;
        }
    }

    private unsafe void RebuildTileMeshGpuBuffer(
        GL gl,
        TileMeshGpuBuffer buffer,
        (int tileX, int tileY) tileKey,
        List<Pm4OverlayObject> objects,
        Pm4OverlayScene overlay,
        bool applyPm4Transform,
        in Matrix4x4 pm4Transform,
        ulong fingerprint)
    {
        var lineFloats = new List<float>(objects.Count * 24);
        var solidFloats = new List<float>(objects.Count * 28);
        int validObjects = 0;

        for (int o = 0; o < objects.Count; o++)
        {
            Pm4OverlayObject obj = objects[o];
            if (!overlay.ShouldRenderPm4ObjectType(obj.Ck24Type) || !overlay.ShouldRenderPm4Object(obj))
                continue;

            var objectKey = (tileKey.tileX, tileKey.tileY, obj.Ck24, obj.ObjectPartId);
            Matrix4x4 objectTransform = overlay.BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);
            Matrix4x4 geometryTransform = Pm4OverlayScene.BuildPm4GeometryTransform(obj, objectTransform, applyObjectTransform);

            validObjects++;
            Vector3 color = overlay.GetPm4ObjectColor(tileKey, obj);
            bool isSelected = overlay.IsHullInSelectedCompound(objectKey)
                || (overlay._selectedPm4ObjectKey.HasValue && overlay._selectedPm4ObjectKey.Value == objectKey);
            bool isGrouped = !isSelected && overlay._selectedPm4ObjectGroupKey.HasValue
                && overlay.IsPm4ObjectInGroup(overlay._selectedPm4ObjectGroupKey.Value, objectKey);
            bool isHighlighted = overlay._highlightedPm4ObjectKeys.Contains(objectKey);

            if (isHighlighted)
            {
                color = Pm4ColorHighlight;
            }
            else if (isSelected)
            {
                // In SemanticCategory mode, preserve the semantic surface color identity!
                // Blend 25% with selection yellow to indicate selection while keeping orange doodads, cyan floors, green lowest floors readable.
                // The selection's 3D boundary is prominently outlined by the yellow BoundingBoxRenderer.
                if (overlay.Pm4ColorMode == Pm4OverlayColorMode.SemanticCategory)
                {
                    color = Vector3.Lerp(color, Pm4ColorSelection, 0.25f);
                }
                else
                {
                    color = Pm4ColorSelection;
                }
            }
            else if (isGrouped)
            {
                color = Vector3.Lerp(color, Pm4ColorSelection, 0.15f);
            }

            // Triangles (pos vec3 + col vec4 = 7 floats)
            if (obj.Triangles.Count > 0)
            {
                for (int i = 0; i < obj.Triangles.Count; i++)
                {
                    Pm4Triangle tri = obj.Triangles[i];
                    Vector3 a = Pm4OverlayScene.ApplyPm4OverlayTransform(tri.A, geometryTransform);
                    Vector3 b = Pm4OverlayScene.ApplyPm4OverlayTransform(tri.B, geometryTransform);
                    Vector3 c = Pm4OverlayScene.ApplyPm4OverlayTransform(tri.C, geometryTransform);

                    solidFloats.Add(a.X); solidFloats.Add(a.Y); solidFloats.Add(a.Z);
                    solidFloats.Add(color.X); solidFloats.Add(color.Y); solidFloats.Add(color.Z); solidFloats.Add(0.20f);

                    solidFloats.Add(b.X); solidFloats.Add(b.Y); solidFloats.Add(b.Z);
                    solidFloats.Add(color.X); solidFloats.Add(color.Y); solidFloats.Add(color.Z); solidFloats.Add(0.20f);

                    solidFloats.Add(c.X); solidFloats.Add(c.Y); solidFloats.Add(c.Z);
                    solidFloats.Add(color.X); solidFloats.Add(color.Y); solidFloats.Add(color.Z); solidFloats.Add(0.20f);
                }
            }

            // Lines (pos vec3 + col vec3 = 6 floats)
            if (obj.Lines.Count > 0)
            {
                for (int i = 0; i < obj.Lines.Count; i++)
                {
                    Pm4LineSegment line = obj.Lines[i];
                    Vector3 from = Pm4OverlayScene.ApplyPm4OverlayTransform(line.From, geometryTransform);
                    Vector3 to = Pm4OverlayScene.ApplyPm4OverlayTransform(line.To, geometryTransform);

                    lineFloats.Add(from.X); lineFloats.Add(from.Y); lineFloats.Add(from.Z);
                    lineFloats.Add(color.X); lineFloats.Add(color.Y); lineFloats.Add(color.Z);

                    lineFloats.Add(to.X); lineFloats.Add(to.Y); lineFloats.Add(to.Z);
                    lineFloats.Add(color.X); lineFloats.Add(color.Y); lineFloats.Add(color.Z);
                }
            }
        }

        // Upload Solid Triangles VBO
        if (solidFloats.Count > 0)
        {
            if (buffer.SolidVao == 0)
            {
                buffer.SolidVao = gl.GenVertexArray();
                buffer.SolidVbo = gl.GenBuffer();
            }

            gl.BindVertexArray(buffer.SolidVao);
            gl.BindBuffer(BufferTargetARB.ArrayBuffer, buffer.SolidVbo);
            fixed (float* p = solidFloats.ToArray())
                gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(solidFloats.Count * sizeof(float)), p, BufferUsageARB.DynamicDraw);

            // layout 0: pos (vec3, 12 bytes), layout 1: color (vec4, 16 bytes), stride = 28 bytes
            gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 7 * sizeof(float), (void*)0);
            gl.EnableVertexAttribArray(0);
            gl.VertexAttribPointer(1, 4, VertexAttribPointerType.Float, false, 7 * sizeof(float), (void*)(3 * sizeof(float)));
            gl.EnableVertexAttribArray(1);
            gl.BindVertexArray(0);

            buffer.SolidVertexCount = solidFloats.Count / 7;
        }
        else
        {
            buffer.SolidVertexCount = 0;
        }

        // Upload Lines VBO
        if (lineFloats.Count > 0)
        {
            if (buffer.LineVao == 0)
            {
                buffer.LineVao = gl.GenVertexArray();
                buffer.LineVbo = gl.GenBuffer();
            }

            gl.BindVertexArray(buffer.LineVao);
            gl.BindBuffer(BufferTargetARB.ArrayBuffer, buffer.LineVbo);
            fixed (float* p = lineFloats.ToArray())
                gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(lineFloats.Count * sizeof(float)), p, BufferUsageARB.DynamicDraw);

            // layout 0: pos (vec3, 12 bytes), layout 1: color (vec3, 12 bytes), stride = 24 bytes
            gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), (void*)0);
            gl.EnableVertexAttribArray(0);
            gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), (void*)(3 * sizeof(float)));
            gl.EnableVertexAttribArray(1);
            gl.BindVertexArray(0);

            buffer.LineVertexCount = lineFloats.Count / 6;
        }
        else
        {
            buffer.LineVertexCount = 0;
        }

        buffer.ObjectCount = validObjects;
        buffer.StateFingerprint = fingerprint;
        buffer.IsValid = true;
    }

    private void EnsureShaders(GL gl)
    {
        if (_shadersFailed || (_lineShader != null && _solidShader != null))
            return;

        try
        {
            string lineVertSrc = @"#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
uniform mat4 uVP;
out vec3 vColor;
void main() {
    gl_Position = uVP * vec4(aPos, 1.0);
    vColor = aColor;
}";
            string lineFragSrc = @"#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() { FragColor = vec4(vColor, 1.0); }";

            string solidVertSrc = @"#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;
uniform mat4 uVP;
out vec4 vColor;
void main() {
    gl_Position = uVP * vec4(aPos, 1.0);
    vColor = aColor;
}";
            string solidFragSrc = @"#version 330 core
in vec4 vColor;
out vec4 FragColor;
void main() { FragColor = vColor; }";

            _lineShader = ShaderProgram.Create(gl, lineVertSrc, lineFragSrc);
            _solidShader = ShaderProgram.Create(gl, solidVertSrc, solidFragSrc);
        }
        catch (Exception ex)
        {
            _shadersFailed = true;
            ViewerLog.Trace($"[Pm4OverlayMeshRenderer] Shader initialization failed: {ex.Message}");
        }
    }

    private void ProcessPendingDeletions(GL gl)
    {
        while (_vaosToDelete.TryDequeue(out uint vao))
        {
            if (vao != 0) gl.DeleteVertexArray(vao);
        }
        while (_vbosToDelete.TryDequeue(out uint vbo))
        {
            if (vbo != 0) gl.DeleteBuffer(vbo);
        }
    }

    /// <summary>Marks a tile's mesh cache as invalid for reload.</summary>
    public void InvalidateTile(int tileX, int tileY)
    {
        if (_tileBuffers.TryGetValue((tileX, tileY), out var buf))
            buf.IsValid = false;
    }

    /// <summary>Marks all cached tile geometry buffers as invalid.</summary>
    public void InvalidateAll()
    {
        foreach (var buf in _tileBuffers.Values)
        {
            if (buf.LineVao != 0) _vaosToDelete.Enqueue(buf.LineVao);
            if (buf.LineVbo != 0) _vbosToDelete.Enqueue(buf.LineVbo);
            if (buf.SolidVao != 0) _vaosToDelete.Enqueue(buf.SolidVao);
            if (buf.SolidVbo != 0) _vbosToDelete.Enqueue(buf.SolidVbo);
            buf.LineVao = 0;
            buf.LineVbo = 0;
            buf.SolidVao = 0;
            buf.SolidVbo = 0;
            buf.IsValid = false;
        }
        _tileBuffers.Clear();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        InvalidateAll();

        _lineShader?.Dispose();
        _solidShader?.Dispose();
        _lineShader = null;
        _solidShader = null;
    }

    private sealed class TileMeshGpuBuffer
    {
        public uint LineVao;
        public uint LineVbo;
        public int LineVertexCount;

        public uint SolidVao;
        public uint SolidVbo;
        public int SolidVertexCount;

        public int ObjectCount;
        public ulong StateFingerprint;
        public bool IsValid;
    }
}

