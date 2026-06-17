using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4FingerprintMatcherTests
{
    private static readonly IReadOnlyDictionary<byte, int> EmptyTypeFlags = new Dictionary<byte, int>();

    [Fact]
    public void MatchOne_IdenticalPm4AndWmoFingerprint_ProducesTopMatchWithFullOverlap()
    {
        Pm4FingerprintRecord fingerprint = CreateBoxFingerprint(
            assetId: "test-box", assetPath: "test.box", assetKind: "wmo", ck24Type: 0x42);

        Pm4FingerprintDatabase db = new("test", "2026-01-01", 1, [fingerprint]);

        Pm4FingerprintMatchResult result = Pm4FingerprintMatcher.MatchOne(
            fingerprint, db.WmoRecords, Pm4FingerprintMatchOptions.Default);

        Assert.Equal(Pm4FingerprintMatchStatus.Matched, result.Status);
        Assert.Single(result.Candidates);
        Assert.Equal(1, result.Candidates[0].Rank);
        Assert.True(result.Candidates[0].OverallScore >= 0.90);
        Assert.True(result.Candidates[0].FootprintOverlapRatio >= 0.95);
    }

    [Fact]
    public void MatchOne_NoDimCompatibleWmo_ReturnsUnresolved()
    {
        Pm4FingerprintRecord pm4Fp = CreateBoxFingerprint(
            assetId: "pm4-box", assetPath: "pm4.box", assetKind: "wmo", ck24Type: 0x42);
        Pm4FingerprintRecord wmoFp = CreateBoxFingerprint(
            assetId: "wmo-huge", assetPath: "huge.wmo", assetKind: "wmo", ck24Type: 0x42,
            sizeX: 500f, sizeY: 500f, sizeZ: 500f);

        Pm4FingerprintDatabase db = new("test", "2026-01-01", 1, [wmoFp]);

        Pm4FingerprintMatchResult result = Pm4FingerprintMatcher.MatchOne(
            pm4Fp, db.WmoRecords, Pm4FingerprintMatchOptions.Default);

        Assert.Equal(Pm4FingerprintMatchStatus.Unresolved, result.Status);
    }

    [Fact]
    public void MatchTwoSameDimDifferentHullWmos_DisambiguatesByHullOverlap()
    {
        Pm4FingerprintRecord pm4Fp = CreateBoxFingerprint(
            assetId: "pm4-box", assetPath: "pm4.box", assetKind: "wmo", ck24Type: 0x42,
            sizeX: 20f, sizeY: 10f, sizeZ: 5f);

        Pm4FingerprintRecord wmoMatch = CreateBoxFingerprint(
            assetId: "wmo-match", assetPath: "match.wmo", assetKind: "wmo", ck24Type: 0x42,
            sizeX: 20f, sizeY: 10f, sizeZ: 5f);

        Pm4FingerprintRecord wmoSameDim = CreateLShapeFingerprint(
            assetId: "wmo-lshape", assetPath: "lshape.wmo", assetKind: "wmo", ck24Type: 0x42,
            longSide: 20f, shortSide: 10f, height: 5f);

        Pm4FingerprintDatabase db = new("test", "2026-01-01", 2, [wmoMatch, wmoSameDim]);

        Pm4FingerprintMatchResult result = Pm4FingerprintMatcher.MatchOne(
            pm4Fp, db.WmoRecords, Pm4FingerprintMatchOptions.Default);

        Assert.True(result.Candidates.Count >= 1);
        Assert.Equal("wmo-match", result.Candidates[0].Candidate.AssetId);
        Assert.True(result.Candidates[0].FootprintOverlapRatio > result.Candidates[1].FootprintOverlapRatio);
    }

    [Fact]
    public void MatchOne_M2Ck24Type_ReturnsIneligible()
    {
        Pm4FingerprintRecord pm4Fp = CreateBoxFingerprint(
            assetId: "pm4-m2", assetPath: "pm4.m2", assetKind: "m2", ck24Type: 0x40);

        Pm4FingerprintDatabase db = new("test", "2026-01-01", 1, []);

        Pm4FingerprintMatchResult result = Pm4FingerprintMatcher.MatchOne(
            pm4Fp, db.WmoRecords, Pm4FingerprintMatchOptions.Default);

        Assert.Equal(Pm4FingerprintMatchStatus.Ineligible, result.Status);
    }

    [Fact]
    public void MatchOne_UnknownCk24Type_ReturnsIneligible()
    {
        Pm4FingerprintRecord pm4Fp = CreateBoxFingerprint(
            assetId: "pm4-unknown", assetPath: "pm4.unknown", assetKind: "unknown", ck24Type: 0x00);

        Pm4FingerprintDatabase db = new("test", "2026-01-01", 1, []);

        Pm4FingerprintMatchResult result = Pm4FingerprintMatcher.MatchOne(
            pm4Fp, db.WmoRecords, Pm4FingerprintMatchOptions.Default);

        Assert.Equal(Pm4FingerprintMatchStatus.Ineligible, result.Status);
    }

    [Fact]
    public void Match_FullCorpus_ReturnsOneResultPerPm4Fingerprint()
    {
        List<Pm4FingerprintRecord> pm4Fingerprints =
        [
            CreateBoxFingerprint("pm4-1", "p1.wmo", "wmo", 0x42),
            CreateBoxFingerprint("pm4-2", "p2.wmo", "wmo", 0x43),
            CreateBoxFingerprint("pm4-3", "p3.m2", "m2", 0x40),
        ];

        Pm4FingerprintDatabase db = new("test", "2026-01-01", 1,
        [
            CreateBoxFingerprint("wmo-1", "w1.wmo", "wmo", 0x42),
        ]);

        IReadOnlyList<Pm4FingerprintMatchResult> results = Pm4FingerprintMatcher.Match(
            pm4Fingerprints, db, Pm4FingerprintMatchOptions.Default);

        Assert.Equal(3, results.Count);
    }

    private static Pm4FingerprintRecord CreateBoxFingerprint(
        string assetId, string assetPath, string assetKind, byte ck24Type,
        float sizeX = 10f, float sizeY = 20f, float sizeZ = 30f)
    {
        List<Vector3> vertices =
        [
            new(0f, 0f, 0f),
            new(sizeX, 0f, 0f),
            new(sizeX, sizeY, 0f),
            new(0f, sizeY, 0f),
            new(0f, 0f, sizeZ),
            new(sizeX, 0f, sizeZ),
            new(sizeX, sizeY, sizeZ),
            new(0f, sizeY, sizeZ),
        ];
        int[] indices = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];

        Pm4FingerprintRecord? fp = Pm4FingerprintExtractor.ExtractFromGeometry(
            vertices, indices, 4, ck24Type, EmptyTypeFlags,
            assetId, assetPath, assetKind);

        Assert.NotNull(fp);
        return fp;
    }

    private static Pm4FingerprintRecord CreateLShapeFingerprint(
        string assetId, string assetPath, string assetKind, byte ck24Type,
        float longSide, float shortSide, float height)
    {
        List<Vector3> vertices =
        [
            new(0f, 0f, 0f),
            new(longSide, 0f, 0f),
            new(longSide, shortSide, 0f),
            new(shortSide, shortSide, 0f),
            new(shortSide, longSide * 0.5f, 0f),
            new(0f, longSide * 0.5f, 0f),
            new(0f, 0f, height),
            new(longSide, 0f, height),
            new(longSide, shortSide, height),
            new(shortSide, shortSide, height),
            new(shortSide, longSide * 0.5f, height),
            new(0f, longSide * 0.5f, height),
        ];
        int[] indices = [0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5,
                         6, 7, 8, 6, 8, 9, 6, 9, 10, 6, 10, 11];

        Pm4FingerprintRecord? fp = Pm4FingerprintExtractor.ExtractFromGeometry(
            vertices, indices, 8, ck24Type, EmptyTypeFlags,
            assetId, assetPath, assetKind);

        Assert.NotNull(fp);
        return fp;
    }
}
