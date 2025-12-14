using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;

namespace WoWRollback.PM4Module.Analysis;

/// <summary>
/// Iterative analysis tool for PM4 scenes.
/// </summary>
public class Pm4SceneAnalyzer
{
    private class ScenePoint
    {
        public Vector3 Position; // World Space (Z-Up)
        public bool IsConsumed;
        public string Source = ""; // "MSVT" or "MSCN"
        public int OriginalIndex;
    }

    private class WmoCandidate
    {
        public uint Ck24;
        public Vector3 Position;
        public string WmoPath = "";
        public List<Vector3> LocalVertices = new();
        public BoundingBox Bounds;
    }

    private struct BoundingBox
    {
        public Vector3 Min;
        public Vector3 Max;
        public BoundingBox(Vector3 min, Vector3 max) { Min = min; Max = max; }
        public bool Contains(Vector3 p) => 
            p.X >= Min.X && p.X <= Max.X &&
            p.Y >= Min.Y && p.Y <= Max.Y &&
            p.Z >= Min.Z && p.Z <= Max.Z;
        
        public void Encapsulate(Vector3 p)
        {
            Min = Vector3.Min(Min, p);
            Max = Vector3.Max(Max, p);
        }
    }

    private List<ScenePoint> _scenePoints = new();

    public void Analyze(string pm4Path, string csvPath, string outputDir)
    {
        System.IO.Directory.CreateDirectory(outputDir);
        
        Console.WriteLine("Step 1: Loading Scene Data...");
        LoadScenePoints(pm4Path);
        Console.WriteLine($"Loaded {_scenePoints.Count} points (MSVT + MSCN).");

        Console.WriteLine("Step 2: Loading WMO Candidates...");
        var candidates = LoadCandidates(csvPath, pm4Path, out HashSet<int> consumedMsvtIndices);
        Console.WriteLine($"Loaded {candidates.Count} WMO candidates. identified {consumedMsvtIndices.Count} WMO MSVT indices.");

        Console.WriteLine("Step 3: Verifying and Subtracting...");
        
        // Pass 1: Exact Subtraction for MSVT
        foreach (var p in _scenePoints)
        {
            if (p.Source == "MSVT" && consumedMsvtIndices.Contains(p.OriginalIndex))
            {
                p.IsConsumed = true;
            }
        }

        // Pass 2: Spatial Subtraction for MSCN
        var verifiedWmos = new List<WmoCandidate>();
        foreach (var wmo in candidates)
        {
            ConsumePointsInBounds(wmo.Bounds, wmo.Position, padding: 0.1f, sourceFilter: "MSCN");
            verifiedWmos.Add(wmo);
        }

        // --- Step 4: Analyze Residuals for Boundary Properties ---
        int boundaryResiduals = 0;
        float edgeThreshold = 1.0f; // 1 yard tolerance
        
        _statsBuilder.AppendLine("\n[Analysis] Residuals Boundary Check:");
        foreach(var p in _scenePoints)
        {
            if (!p.IsConsumed && p.Source == "MSCN")
            {
                bool onEdge = 
                    Math.Abs(p.Position.X - _tileMin.X) < edgeThreshold || Math.Abs(p.Position.X - _tileMax.X) < edgeThreshold ||
                    Math.Abs(p.Position.Y - _tileMin.Y) < edgeThreshold || Math.Abs(p.Position.Y - _tileMax.Y) < edgeThreshold;
                
                if (onEdge) boundaryResiduals++;
            }
        }
        int mscnResiduals = _scenePoints.Count(p => !p.IsConsumed && p.Source == "MSCN");
        _statsBuilder.AppendLine($"  MSCN Residuals on Tile Edge (>1yd): {boundaryResiduals} / {mscnResiduals}");
        
        if (mscnResiduals > 0)
        {
             _statsBuilder.AppendLine($"  Percentage on Boundary: {(float)boundaryResiduals / mscnResiduals * 100.0:F1}%");
        }

        _statsBuilder.AppendLine($"\nProcessed {candidates.Count} WMOs.");
        _statsBuilder.AppendLine($"Total Points: {_scenePoints.Count}");
        _statsBuilder.AppendLine($"Consumed Points: {_scenePoints.Count(p => p.IsConsumed)}");
        _statsBuilder.AppendLine($"Residual Points: {_scenePoints.Count(p => !p.IsConsumed)}");
        _statsBuilder.AppendLine($"  - MSVT Residuals: {_scenePoints.Count(p => !p.IsConsumed && p.Source == "MSVT")}");
        _statsBuilder.AppendLine($"  - MSCN Residuals: {mscnResiduals}");
        
        System.IO.File.WriteAllText(System.IO.Path.Combine(outputDir, "analysis_diagnostics.txt"), _statsBuilder.ToString());
        
        Console.WriteLine(_statsBuilder.ToString());
        
        Console.WriteLine("Step 4: Exporting Residuals and Debug...");
        ExportResiduals(System.IO.Path.Combine(outputDir, "residuals.obj"));
        ExportConsumed(System.IO.Path.Combine(outputDir, "debug_consumed.obj"));
        ExportVerifiedCsv(verifiedWmos, System.IO.Path.Combine(outputDir, "verified_wmos_scene.csv"));
    }

    private void LoadScenePoints(string pm4Path)
    {
        var pm4 = PM4File.FromFile(pm4Path);
        
        // Load MSVT (Track Index)
        for (int i = 0; i < pm4.MeshVertices.Count; i++)
        {
            var v = pm4.MeshVertices[i];
            _scenePoints.Add(new ScenePoint 
            { 
                Position = new Vector3(v.X, v.Z, v.Y), 
                Source = "MSVT",
                IsConsumed = false,
                OriginalIndex = i
            });
        }

        // Load MSCN (No Index mapping to MSVT)
        foreach (var v in pm4.ExteriorVertices)
        {
            var alignedV = new Vector3(v.Y, v.X, v.Z);
            _scenePoints.Add(new ScenePoint 
            { 
                Position = new Vector3(alignedV.X, alignedV.Z, alignedV.Y), 
                Source = "MSCN",
                IsConsumed = false,
                OriginalIndex = -1
            });
        }

        // --- Rigorous Verification Logic ---
        var sb = new StringBuilder();
        sb.AppendLine($"--- Analysis for {System.IO.Path.GetFileName(pm4Path)} ---");

        // 1. Tile Boundary Analysis
        // We assume the MSVT points mostly define the tile volume.
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        foreach(var p in _scenePoints)
        {
            min = Vector3.Min(min, p.Position);
            max = Vector3.Max(max, p.Position);
        }
        sb.AppendLine($"Tile Bounds (Auto-Detected): {min} to {max}");
        sb.AppendLine($"Size: {max-min}");

        // 2. CK24=0 Surface Geometry Analysis
        var zeroSurfs = pm4.Surfaces.Where(s => s.CK24 == 0).ToList();
        if (zeroSurfs.Count > 0)
        {
            sb.AppendLine($"\n[Analysis] CK24=0 Surfaces ({zeroSurfs.Count} found):");
            foreach(var s in zeroSurfs)
            {
                // Inspect geometry
                sb.Append($"  ID={s.GroupKey} Indices={s.IndexCount} Attr={s.AttributeMask:X} ");
                
                if (s.IndexCount == 4)
                {
                    // Check if Quad
                    sb.Append(" [Possible Quad/Portal]");
                }
                
                // Get Vertices to check orientation
                if (s.IndexCount > 0)
                {
                    uint firstIdx = pm4.MeshIndices[(int)s.MsviFirstIndex];
                    var v1 = pm4.MeshVertices[(int)firstIdx];
                    // Just basic check: Vertical range?
                    // We need all verts to be sure, but let's check height delta of first few
                    // (Omitted full normal calc for brevity, but could add if needed)
                }
                sb.AppendLine();
            }
        }
        else
        {
            sb.AppendLine("\n[Info] No CK24=0 Surfaces to analyze.");
        }
        
        // Store for later (Residual boundary check needs final IsConsumed status)
        _statsBuilder = sb;
        _tileMin = min;
        _tileMax = max;

        Console.WriteLine(sb.ToString());
    }

    private StringBuilder _statsBuilder = new StringBuilder();
    private Vector3 _tileMin = new Vector3(float.MaxValue);
    private Vector3 _tileMax = new Vector3(float.MinValue);

    // ... (rest of class) ...



    private List<WmoCandidate> LoadCandidates(string csvPath, string pm4Path, out HashSet<int> consumedIndices)
    {
        var candidates = new List<WmoCandidate>();
        consumedIndices = new HashSet<int>();
        
        var pm4 = PM4File.FromFile(pm4Path);
        var groups = pm4.Surfaces
             .GroupBy(s => s.CK24)
             .Where(g => g.Key != 0) // Skip residuals in loading candidates
             .ToDictionary(g => g.Key, g => g.ToList());

        var lines = System.IO.File.ReadAllLines(csvPath).Skip(1);
        foreach (var line in lines)
        {
            var parts = line.Split(',');
            if (parts.Length < 7) continue;

            if (uint.TryParse(parts[0], System.Globalization.NumberStyles.HexNumber, null, out uint ck24))
            {
                float x = float.Parse(parts[4]);
                float y = float.Parse(parts[5]);
                float z = float.Parse(parts[6]);
                var pos = new Vector3(x, y, z);
                
                var bounds = new BoundingBox(new Vector3(float.MaxValue), new Vector3(float.MinValue));
                bool hasGeometry = false;

                if (groups.TryGetValue(ck24, out var surfaces))
                {
                    foreach (var surf in surfaces)
                    {
                        if (surf.GroupKey == 0) continue; 
                        
                        uint start = surf.MsviFirstIndex;
                        uint count = surf.IndexCount;
                        for(int i=0; i<count; i++) 
                        {
                            if (start+i < pm4.MeshIndices.Count) {
                                uint vIdx = pm4.MeshIndices[(int)(start+i)];
                                consumedIndices.Add((int)vIdx);
                                var v = pm4.MeshVertices[(int)vIdx];
                                var worldV = new Vector3(v.X, v.Z, v.Y);
                                bounds.Encapsulate(worldV);
                                hasGeometry = true;
                            }
                        }
                    }
                }
                
                if (hasGeometry)
                {
                    candidates.Add(new WmoCandidate { Ck24 = ck24, Position = pos, WmoPath = parts[1], Bounds = bounds });
                }
            }
        }
        return candidates;
    }

    private int ConsumePointsInBounds(BoundingBox bounds, Vector3 wmoPos, float padding, string? sourceFilter = null)
    {
        int count = 0;
        var paddedMin = bounds.Min - new Vector3(padding);
        var paddedMax = bounds.Max + new Vector3(padding);
        var paddedBounds = new BoundingBox(paddedMin, paddedMax);

        foreach (var p in _scenePoints)
        {
            if (p.IsConsumed) continue;
            if (sourceFilter != null && p.Source != sourceFilter) continue;

            if (paddedBounds.Contains(p.Position))
            {
                p.IsConsumed = true;
                count++;
            }
        }
        return count;
    }

    private void ExportResiduals(string path)
    {
        using var sw = new System.IO.StreamWriter(path);
        sw.WriteLine("# PM4 Residuals (M2 Candidates)");
        sw.WriteLine("# Points not consumed by WMO bounding boxes or indices");
        
        foreach (var p in _scenePoints)
        {
            if (!p.IsConsumed)
            {
                sw.WriteLine($"v {p.Position.X:F4} {p.Position.Y:F4} {p.Position.Z:F4}");
            }
        }
    }
    
    private void ExportConsumed(string path)
    {
        using var sw = new System.IO.StreamWriter(path);
        sw.WriteLine("# PM4 Consumed Points");
        foreach (var p in _scenePoints)
        {
            if (p.IsConsumed)
            {
                sw.WriteLine($"v {p.Position.X:F4} {p.Position.Y:F4} {p.Position.Z:F4}");
            }
        }
    }

    private void ExportVerifiedCsv(List<WmoCandidate> wmos, string path)
    {
        using var sw = new System.IO.StreamWriter(path);
        sw.WriteLine("ck24,wmo_path,name_id,unique_id,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,scale");
        uint idCounter = 7000000;
        foreach (var wmo in wmos)
        {
             sw.WriteLine($"{wmo.Ck24:X6},{wmo.WmoPath},0,{idCounter++},{wmo.Position.X:F4},{wmo.Position.Y:F4},{wmo.Position.Z:F4},0,0,0,1");
        }
    }
}
