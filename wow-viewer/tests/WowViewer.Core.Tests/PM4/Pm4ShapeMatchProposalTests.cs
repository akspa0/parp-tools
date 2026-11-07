using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Footprint;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;
using Xunit;

namespace WowViewer.Core.Tests.PM4;

public sealed class Pm4ShapeMatchProposalTests
{
    private static string GetWowViewerRoot()
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "WowViewer.slnx")))
                return current.FullName;

            current = current.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate the wow-viewer repository root from the test output directory.");
    }

    private static string GetValidationMapDirectory()
    {
        string root = GetWowViewerRoot();
        string originalDev = Path.Combine(root, "test_data", "original_development", "World", "Maps", "development");
        if (Directory.Exists(originalDev))
            return originalDev;

        return Path.Combine(root, "test_data", "development", "World", "Maps", "development");
    }

    [Fact]
    public void ValidationTile_MeasuresCentroidCalibration_AndRanksAuthoredProposalsWithinOneYard()
    {
        string mapDir = GetValidationMapDirectory();
        string pm4Path = Path.Combine(mapDir, "development_00_00.pm4");
        string adtPath = Path.Combine(mapDir, "development_0_0_obj0.adt");

        Assert.True(File.Exists(pm4Path), $"Missing validation PM4 tile: {pm4Path}");
        Assert.True(File.Exists(adtPath), $"Missing validation ADT object tile: {adtPath}");

        Pm4ResearchDocument doc = Pm4ResearchReader.ReadFile(pm4Path);
        AdtPlacementCatalog catalog = AdtPlacementReader.Read(adtPath);

        Assert.NotEmpty(catalog.WorldModelPlacements);

        // Group PM4 surface vertices by CK24 (top 24 bits of MSUR.0x1C Placement-Z)
        var verticesByCk24 = new Dictionary<uint, List<Vector3>>();
        IReadOnlyList<Vector3> msvt = doc.KnownChunks.Msvt;
        IReadOnlyList<uint> msvi = doc.KnownChunks.Msvi;
        IReadOnlyList<Pm4MsurEntry> msur = doc.KnownChunks.Msur;

        for (int i = 0; i < msur.Count; i++)
        {
            Pm4MsurEntry surface = msur[i];
            uint ck24 = surface.PackedParams >> 8;
            if (ck24 == 0)
                continue;

            if (!verticesByCk24.TryGetValue(ck24, out List<Vector3>? list))
            {
                list = [];
                verticesByCk24[ck24] = list;
            }

            int first = (int)surface.MsviFirstIndex;
            int end = Math.Min(first + surface.IndexCount, msvi.Count);
            for (int k = first; k < end; k++)
            {
                int vIdx = (int)msvi[k];
                if ((uint)vIdx < (uint)msvt.Count)
                    list.Add(Pm4CoordinateService.Pm4LocalToAdtPlacement(msvt[vIdx]));
            }
        }

        // Build candidate pool from the authored WMOs on this validation tile using centered local bounds
        var candidateByPath = new Dictionary<string, Pm4ModelCandidate>(StringComparer.OrdinalIgnoreCase);
        foreach (AdtWorldModelPlacement wmo in catalog.WorldModelPlacements)
        {
            if (candidateByPath.ContainsKey(wmo.ModelPath))
                continue;

            Vector3 span = Vector3.Abs(wmo.BoundsMax - wmo.BoundsMin);
            float halfX = MathF.Max(0.5f, span.X * 0.5f);
            float halfY = MathF.Max(0.5f, span.Y * 0.5f);
            float height = MathF.Max(0.5f, span.Z);
            candidateByPath[wmo.ModelPath] = Pm4ModelCandidate.Create(
                wmo.ModelPath,
                "WMO",
                new Vector3(-halfX, -halfY, 0f),
                new Vector3(halfX, halfY, height));
        }

        IReadOnlyList<Pm4ModelCandidate> candidates = candidateByPath.Values.ToList();
        var calibrationSamples = new List<(Vector3 Solved, Vector3 Authored)>();
        var hullFixtures = new List<(uint Ck24, Pm4FootprintPrism Prism, AdtWorldModelPlacement Authored, Vector3 AuthoredFootprintCenter)>();

        foreach (AdtWorldModelPlacement wmo in catalog.WorldModelPlacements)
        {
            uint matchedCk24 = 0;
            float bestZDelta = float.MaxValue;
            foreach (uint ck24 in verticesByCk24.Keys)
            {
                float z = BitConverter.UInt32BitsToSingle(ck24 << 8);
                float dz = MathF.Abs(z - wmo.Position.Z);
                if (dz < bestZDelta)
                {
                    bestZDelta = dz;
                    matchedCk24 = ck24;
                }
            }

            Assert.True(bestZDelta < 0.05f, $"Expected CK24 match for {wmo.ModelPath} at Z={wmo.Position.Z:F3}");

            List<Vector3> verts = verticesByCk24[matchedCk24];
            float placementZ = BitConverter.UInt32BitsToSingle(matchedCk24 << 8);
            Pm4FootprintPrism prism = Pm4FootprintExtruder.ExtrudeFootprint(verts, null, placementZ, wmo.BoundsMin.Z);

            Vector3 authoredFootprintCenter = new(
                (wmo.BoundsMin.X + wmo.BoundsMax.X) * 0.5f,
                (wmo.BoundsMin.Y + wmo.BoundsMax.Y) * 0.5f,
                placementZ);

            Pm4ShapeMatchProposal uncalibrated = Pm4ShapeMatchAnalysisService.Analyze(
                matchedCk24,
                prism,
                candidates,
                Pm4CentroidCalibration.None,
                maxCandidates: 5);

            Assert.False(uncalibrated.IsUnmatched);
            Assert.NotNull(uncalibrated.TopCandidate);

            calibrationSamples.Add((uncalibrated.TopCandidate!.SolvedPosition, authoredFootprintCenter));
            hullFixtures.Add((matchedCk24, prism, wmo, authoredFootprintCenter));
        }

        Pm4CentroidMeasurement measurement = Pm4ShapeMatchAnalysisService.MeasureCalibration(
            calibrationSamples,
            "development_00_00.pm4 vs development_0_0_obj0.adt");

        Assert.Equal(15, measurement.SampleCount);

        var calibration = new Pm4CentroidCalibration(measurement.MeanOffset, measurement.Source);
        int withinOneYardCount = 0;

        foreach ((uint ck24, Pm4FootprintPrism prism, AdtWorldModelPlacement authored, Vector3 authoredFootprintCenter) in hullFixtures)
        {
            Pm4ShapeMatchProposal calibrated = Pm4ShapeMatchAnalysisService.Analyze(
                ck24,
                prism,
                candidates,
                calibration,
                maxCandidates: 5);

            Assert.False(calibrated.IsUnmatched);
            Assert.NotNull(calibrated.TopCandidate);

            // The candidate pool must contain a matching wall/tower asset and the proposal is credible
            Assert.True(calibrated.TopCandidate!.Score >= Pm4ShapeMatchAnalysisService.MinimumCredibleScore);
            Assert.NotEmpty(calibrated.Candidates);

            float planarError = Vector2.Distance(
                new Vector2(calibrated.TopCandidate.SolvedPosition.X, calibrated.TopCandidate.SolvedPosition.Y),
                new Vector2(authoredFootprintCenter.X, authoredFootprintCenter.Y));
            if (planarError <= 1.0f)
                withinOneYardCount++;
        }

        // SC-001: at least 90% of proposals place the model's footprint center within 1 yard of the hull's footprint center after calibration
        double withinOneYardRate = (double)withinOneYardCount / hullFixtures.Count;
        Assert.True(
            withinOneYardRate >= 0.90,
            $"Expected >= 90% of calibrated proposals within 1.0 yd on validation tile, got {withinOneYardCount}/{hullFixtures.Count} ({withinOneYardRate:P1}).");
    }
}

