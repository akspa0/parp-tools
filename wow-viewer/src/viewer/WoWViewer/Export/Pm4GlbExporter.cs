using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using SharpGLTF.Geometry;
using SharpGLTF.Geometry.VertexTypes;
using SharpGLTF.Materials;
using SharpGLTF.Scenes;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using WoWViewer.Terrain.Pm4;
using WowViewer.Core.IO.Files;
using WowViewer.Core.PM4.Matching;

namespace WoWViewer.Export;

using VERTEX = VertexBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>;

/// <summary>
/// Spec 298: Exports PM4 collision meshes and navigation geometry to GLB (glTF 2.0 Binary).
/// Every PM4 collision object is exported as an independent, named scene node with color-coded
/// PBR collision materials, sitting seamlessly atop authentic textured/MCCV terrain geometry.
/// </summary>
internal static class Pm4GlbExporter
{
    private static Vector3 ZupToYup(float x, float y, float z) => new(x, z, -y);
    private static Vector3 ZupToYup(Vector3 v) => new(v.X, v.Z, -v.Y);

    /// <summary>
    /// Exports all PM4 collision objects on a single tile to a standalone GLB file with separate object nodes.
    /// </summary>
    internal static void ExportTilePm4Glb(
        TerrainManager? terrain,
        IDataSource? dataSource,
        Md5TranslateIndex? md5Index,
        int tileX,
        int tileY,
        IReadOnlyList<Pm4OverlayObject> objects,
        string outputPath,
        bool includeTerrain = true,
        IReadOnlyList<TaughtPairing>? taughtPairings = null)
    {
        ArgumentNullException.ThrowIfNull(objects);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");
        var scene = new SceneBuilder();
        var materialCache = new Dictionary<byte, MaterialBuilder>();

        // 1. Add textured/MCCV terrain geometry if available
        if (includeTerrain && terrain != null)
        {
            try
            {
                var result = terrain.GetOrLoadTileLoadResult(tileX, tileY);
                if (result.Chunks.Count > 0)
                {
                    var terrainMesh = MapGlbExporter.BuildTerrainTileMesh(terrain, tileX, tileY, result.Chunks, dataSource, md5Index);
                    if (terrainMesh != null)
                    {
                        scene.AddRigidMesh(terrainMesh, Matrix4x4.Identity);
                    }
                }
            }
            catch (Exception ex)
            {
                ViewerLog.Info(ViewerLog.Category.Terrain, $"[Pm4GlbExporter] Could not attach terrain for tile ({tileX},{tileY}): {ex.Message}");
            }
        }

        // 2. Add each PM4 collision object as an independent, named rigid mesh node
        int exportedCount = 0;
        foreach (var obj in objects)
        {
            string? pairedModel = ResolvePairedModelName(obj, tileX, tileY, taughtPairings);
            if (AddPm4ObjectToScene(scene, obj, tileX, tileY, pairedModel, materialCache))
            {
                exportedCount++;
            }
        }

        var model = scene.ToGltf2();
        model.SaveGLB(outputPath);

        ViewerLog.Important(ViewerLog.Category.General,
            $"[Pm4GlbExporter] Exported {exportedCount} PM4 objects on tile ({tileX},{tileY}) to {outputPath}");
    }

    /// <summary>
    /// Exports all loaded PM4 collision objects across multiple tiles to a combined GLB file.
    /// </summary>
    internal static void ExportAllLoadedPm4Glb(
        TerrainManager? terrain,
        IDataSource? dataSource,
        Md5TranslateIndex? md5Index,
        IReadOnlyDictionary<(int tileX, int tileY), List<Pm4OverlayObject>> tileObjects,
        string outputPath,
        bool includeTerrain = true,
        IReadOnlyList<TaughtPairing>? taughtPairings = null)
    {
        ArgumentNullException.ThrowIfNull(tileObjects);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");
        var scene = new SceneBuilder();
        var materialCache = new Dictionary<byte, MaterialBuilder>();

        int totalTiles = 0;
        int totalObjects = 0;

        foreach (var kvp in tileObjects.OrderBy(e => e.Key.tileX).ThenBy(e => e.Key.tileY))
        {
            int tileX = kvp.Key.tileX;
            int tileY = kvp.Key.tileY;
            var objects = kvp.Value;
            if (objects.Count == 0) continue;

            totalTiles++;

            // Optional terrain mesh for this tile
            if (includeTerrain && terrain != null)
            {
                try
                {
                    var result = terrain.GetOrLoadTileLoadResult(tileX, tileY);
                    if (result.Chunks.Count > 0)
                    {
                        var terrainMesh = MapGlbExporter.BuildTerrainTileMesh(terrain, tileX, tileY, result.Chunks, dataSource, md5Index);
                        if (terrainMesh != null)
                        {
                            scene.AddRigidMesh(terrainMesh, Matrix4x4.Identity);
                        }
                    }
                }
                catch (Exception ex)
                {
                    ViewerLog.Info(ViewerLog.Category.Terrain, $"[Pm4GlbExporter] Could not attach terrain for tile ({tileX},{tileY}): {ex.Message}");
                }
            }

            // PM4 objects
            foreach (var obj in objects)
            {
                string? pairedModel = ResolvePairedModelName(obj, tileX, tileY, taughtPairings);
                if (AddPm4ObjectToScene(scene, obj, tileX, tileY, pairedModel, materialCache))
                {
                    totalObjects++;
                }
            }
        }

        var model = scene.ToGltf2();
        model.SaveGLB(outputPath);

        ViewerLog.Important(ViewerLog.Category.General,
            $"[Pm4GlbExporter] Exported combined PM4 scene ({totalObjects} objects across {totalTiles} tiles) to {outputPath}");
    }

    /// <summary>
    /// Exports a single selected PM4 collision object to a standalone GLB file.
    /// </summary>
    internal static void ExportSingleObjectGlb(
        Pm4OverlayObject obj,
        int tileX,
        int tileY,
        string outputPath,
        string? pairedModelName = null)
    {
        ArgumentNullException.ThrowIfNull(obj);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");
        var scene = new SceneBuilder();
        var materialCache = new Dictionary<byte, MaterialBuilder>();

        AddPm4ObjectToScene(scene, obj, tileX, tileY, pairedModelName, materialCache);

        var model = scene.ToGltf2();
        model.SaveGLB(outputPath);

        ViewerLog.Important(ViewerLog.Category.General,
            $"[Pm4GlbExporter] Exported single PM4 object [{tileX}_{tileY}] Part {obj.ObjectPartId} to {outputPath}");
    }

    /// <summary>
    /// Converts a PM4 overlay object into a distinct rigid mesh and attaches it to the GLB scene builder.
    /// </summary>
    internal static bool AddPm4ObjectToScene(
        SceneBuilder scene,
        Pm4OverlayObject obj,
        int tileX,
        int tileY,
        string? pairedModelName,
        Dictionary<byte, MaterialBuilder> materialCache)
    {
        if (obj.Triangles.Count == 0)
            return false;

        var mat = GetOrCreateCollisionMaterial(obj.DominantGroupKey, materialCache);

        string partLabel = $"Part_{obj.ObjectPartId}";
        string descLabel = !string.IsNullOrWhiteSpace(pairedModelName)
            ? Path.GetFileNameWithoutExtension(pairedModelName)
            : (obj.LinkGroupObjectId != 0 ? $"Lnk_0x{obj.LinkGroupObjectId:X}" : $"Class_0x{obj.DominantGroupKey:X2}");

        string nodeName = $"[{tileX:D2}_{tileY:D2}]_{partLabel}_{descLabel}";

        var mesh = new MeshBuilder<VertexPositionNormal, VertexTexture1, VertexEmpty>(nodeName);
        var prim = mesh.UsePrimitive(mat);

        for (int t = 0; t < obj.Triangles.Count; t++)
        {
            var tri = obj.Triangles[t];
            Vector3 wA = Vector3.Transform(tri.A, obj.BaseTransform);
            Vector3 wB = Vector3.Transform(tri.B, obj.BaseTransform);
            Vector3 wC = Vector3.Transform(tri.C, obj.BaseTransform);

            Vector3 yupA = ZupToYup(wA);
            Vector3 yupB = ZupToYup(wB);
            Vector3 yupC = ZupToYup(wC);

            Vector3 norm = Vector3.Normalize(Vector3.Cross(yupB - yupA, yupC - yupA));
            if (!float.IsFinite(norm.X) || norm == Vector3.Zero)
                norm = Vector3.UnitY;

            prim.AddTriangle(
                new VERTEX(new VertexPositionNormal(yupA, norm), new VertexTexture1(Vector2.Zero)),
                new VERTEX(new VertexPositionNormal(yupB, norm), new VertexTexture1(Vector2.UnitX)),
                new VERTEX(new VertexPositionNormal(yupC, norm), new VertexTexture1(Vector2.UnitY)));
        }

        scene.AddRigidMesh(mesh, Matrix4x4.Identity);
        return true;
    }

    private static MaterialBuilder GetOrCreateCollisionMaterial(
        byte groupKey,
        Dictionary<byte, MaterialBuilder> materialCache)
    {
        if (materialCache.TryGetValue(groupKey, out var existing))
            return existing;

        Vector4 color = groupKey switch
        {
            0x03 => new Vector4(0.00f, 0.90f, 1.00f, 0.85f),         // Doodad: Vibrant Cyan
            0x10 => new Vector4(0.00f, 0.90f, 0.45f, 0.90f),         // Lowest floor: Emerald green
            0x11 or 0x12 => new Vector4(0.30f, 0.85f, 0.39f, 0.90f), // Floor / ext solid: Lime green
            0x13 => new Vector4(0.85f, 0.00f, 0.98f, 0.85f),         // Upper floor / interior: Purple
            0x14 => new Vector4(1.00f, 0.57f, 0.00f, 0.90f),         // Angled / ramp: Orange
            0x15 => new Vector4(1.00f, 0.84f, 0.00f, 0.90f),         // Flat base: Amber
            _ => new Vector4(0.16f, 0.47f, 1.00f, 0.85f)             // Default: Sky blue
        };

        string name = groupKey switch
        {
            0x03 => "pm4_doodad_cyan",
            0x10 => "pm4_lowest_floor_green",
            0x11 or 0x12 => "pm4_floor_lime",
            0x13 => "pm4_upper_floor_purple",
            0x14 => "pm4_ramp_orange",
            0x15 => "pm4_flat_base_amber",
            _ => $"pm4_class_0x{groupKey:X2}"
        };

        var mat = new MaterialBuilder(name)
            .WithDoubleSide(true)
            .WithMetallicRoughnessShader()
            .WithBaseColor(color);

        if (color.W < 1.0f)
        {
            mat.WithAlpha(AlphaMode.BLEND);
        }

        materialCache[groupKey] = mat;
        return mat;
    }

    private static string? ResolvePairedModelName(
        Pm4OverlayObject obj,
        int tileX,
        int tileY,
        IReadOnlyList<TaughtPairing>? taughtPairings)
    {
        if (taughtPairings == null || taughtPairings.Count == 0)
            return null;

        // Check exact component / tile match first
        var exact = taughtPairings.FirstOrDefault(t =>
            t.TileX == tileX && t.TileY == tileY && t.ObjectPartId == obj.ObjectPartId);
        if (exact != null)
            return exact.ModelName;

        // Fallback to geometric bounding dimensions agreement
        float objLen = MathF.Abs(obj.BoundsMax.X - obj.BoundsMin.X);
        float objWid = MathF.Abs(obj.BoundsMax.Y - obj.BoundsMin.Y);
        float objH = MathF.Abs(obj.BoundsMax.Z - obj.BoundsMin.Z);

        var match = taughtPairings.FirstOrDefault(t =>
            MathF.Abs(t.Length - objLen) < 0.75f &&
            MathF.Abs(t.Width - objWid) < 0.75f &&
            MathF.Abs(t.Height - objH) < 1.0f);

        return match?.ModelName;
    }
}
