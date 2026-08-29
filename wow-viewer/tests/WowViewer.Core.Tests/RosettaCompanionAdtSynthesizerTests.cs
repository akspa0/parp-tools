using System.Security.Cryptography;
using System.Text.Json;
using WowViewer.Core.IO.Maps;
using Xunit;

namespace WowViewer.Core.Tests;

public sealed class RosettaCompanionAdtSynthesizerTests
{
    [Fact]
    public void ScanPm4Tiles_IdentifiesOrphansAndExistingCompanions()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"rosetta_scan_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            // 1. Orphan PM4 (no ADT)
            File.WriteAllBytes(Path.Combine(tempDir, "Azeroth_30_48.pm4"), [1, 2, 3]);

            // 2. Paired PM4 (has matching ADT)
            File.WriteAllBytes(Path.Combine(tempDir, "Azeroth_30_49.pm4"), [1, 2, 3]);
            File.WriteAllBytes(Path.Combine(tempDir, "Azeroth_30_49.adt"), [4, 5, 6]);

            // 3. Alternative naming paired PM4 (has matching _obj0.adt)
            File.WriteAllBytes(Path.Combine(tempDir, "Kalimdor_24_24.pm4"), [1, 2, 3]);
            File.WriteAllBytes(Path.Combine(tempDir, "Kalimdor_24_24_obj0.adt"), [7, 8, 9]);

            var results = RosettaCompanionAdtSynthesizer.ScanPm4Tiles(tempDir);

            Assert.Equal(3, results.Count);

            var azeroth3048 = results.Single(static t => t.MapName == "Azeroth" && t.TileX == 48 && t.TileY == 30);
            Assert.False(azeroth3048.HasExistingCompanion);

            var azeroth3049 = results.Single(static t => t.MapName == "Azeroth" && t.TileX == 49 && t.TileY == 30);
            Assert.True(azeroth3049.HasExistingCompanion);

            var kalimdor2424 = results.Single(static t => t.MapName == "Kalimdor" && t.TileX == 24 && t.TileY == 24);
            Assert.True(kalimdor2424.HasExistingCompanion);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void SynthesizeCompanion_ProducesCompliantAdtWithValidSha256()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"rosetta_synth_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string pm4Path = Path.Combine(tempDir, "Development_10_20.pm4");
            File.WriteAllBytes(pm4Path, [0x50, 0x4D, 0x34, 0x00]);

            var orphan = new RosettaOrphanPm4Tile(
                pm4Path,
                "Development",
                TileX: 20,
                TileY: 10,
                ExpectedCompanionFileName: "Development_10_20.adt",
                HasExistingCompanion: false);

            var record = RosettaCompanionAdtSynthesizer.SynthesizeCompanion(orphan, tempDir);

            Assert.Equal("Synthesized", record.Status);
            Assert.True(File.Exists(record.CompanionFilePath));

            byte[] generatedBytes = File.ReadAllBytes(record.CompanionFilePath);
            Assert.True(generatedBytes.Length > 0);

            string computedHash = Convert.ToHexString(SHA256.HashData(generatedBytes)).ToLowerInvariant();
            Assert.Equal(computedHash, record.ContentSha256);

            // Verify the generated file is a valid LK ADT
            var parsed = LkAdtReader.Read(generatedBytes, null, null, 20, 10);
            Assert.Equal(20, parsed.TileX);
            Assert.Equal(10, parsed.TileY);
            Assert.Equal(256, parsed.Chunks.Count);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void SynthesizeCompanion_SafeOverwrite_SkipsExistingWithoutOverwriteFlag()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"rosetta_skip_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string pm4Path = Path.Combine(tempDir, "Azeroth_24_24.pm4");
            string adtPath = Path.Combine(tempDir, "Azeroth_24_24.adt");
            byte[] existingBytes = [0xAA, 0xBB, 0xCC, 0xDD];
            File.WriteAllBytes(adtPath, existingBytes);

            var orphan = new RosettaOrphanPm4Tile(
                pm4Path,
                "Azeroth",
                TileX: 24,
                TileY: 24,
                ExpectedCompanionFileName: "Azeroth_24_24.adt",
                HasExistingCompanion: true);

            var options = new RosettaCompanionSynthesisOptions(Overwrite: false);
            var record = RosettaCompanionAdtSynthesizer.SynthesizeCompanion(orphan, tempDir, options);

            Assert.Equal("SkippedExisting", record.Status);
            byte[] currentBytes = File.ReadAllBytes(adtPath);
            Assert.Equal(existingBytes, currentBytes);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void SynthesizeCompanion_OverwriteFlag_OverwritesExistingFile()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"rosetta_overwrite_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string pm4Path = Path.Combine(tempDir, "Azeroth_24_24.pm4");
            string adtPath = Path.Combine(tempDir, "Azeroth_24_24.adt");
            byte[] dummyBytes = [0x11, 0x22, 0x33];
            File.WriteAllBytes(adtPath, dummyBytes);

            var orphan = new RosettaOrphanPm4Tile(
                pm4Path,
                "Azeroth",
                TileX: 24,
                TileY: 24,
                ExpectedCompanionFileName: "Azeroth_24_24.adt",
                HasExistingCompanion: true);

            var options = new RosettaCompanionSynthesisOptions(Overwrite: true);
            var record = RosettaCompanionAdtSynthesizer.SynthesizeCompanion(orphan, tempDir, options);

            Assert.Equal("Synthesized", record.Status);
            byte[] currentBytes = File.ReadAllBytes(adtPath);
            Assert.NotEqual(dummyBytes, currentBytes);

            var parsed = LkAdtReader.Read(currentBytes, null, null, 24, 24);
            Assert.Equal(24, parsed.TileX);
            Assert.Equal(24, parsed.TileY);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void SynthesizeCompanion_SplitObj0Mode_EmitsObj0File()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"rosetta_obj0_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string pm4Path = Path.Combine(tempDir, "Azeroth_15_25.pm4");

            var orphan = new RosettaOrphanPm4Tile(
                pm4Path,
                "Azeroth",
                TileX: 25,
                TileY: 15,
                ExpectedCompanionFileName: "Azeroth_15_25.adt",
                HasExistingCompanion: false);

            var options = new RosettaCompanionSynthesisOptions(EmitSplitObj0: true);
            var record = RosettaCompanionAdtSynthesizer.SynthesizeCompanion(orphan, tempDir, options);

            Assert.Equal("Synthesized", record.Status);
            Assert.EndsWith("Azeroth_15_25_obj0.adt", record.CompanionFilePath);
            Assert.True(File.Exists(record.CompanionFilePath));
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void SynthesizeAllCompanions_GeneratesValidProvenanceReport()
    {
        string tempPm4Dir = Path.Combine(Path.GetTempPath(), $"rosetta_prov_pm4_{Guid.NewGuid():N}");
        string tempOutDir = Path.Combine(Path.GetTempPath(), $"rosetta_prov_out_{Guid.NewGuid():N}");
        string reportPath = Path.Combine(tempOutDir, "provenance_report.json");

        Directory.CreateDirectory(tempPm4Dir);
        Directory.CreateDirectory(tempOutDir);

        try
        {
            File.WriteAllBytes(Path.Combine(tempPm4Dir, "TestMap_05_10.pm4"), [1, 2, 3]);
            File.WriteAllBytes(Path.Combine(tempPm4Dir, "TestMap_05_11.pm4"), [4, 5, 6]);

            var report = RosettaCompanionAdtSynthesizer.SynthesizeAllCompanions(
                tempPm4Dir,
                tempOutDir,
                adtDirectory: tempOutDir,
                provenanceReportPath: reportPath);

            Assert.Equal(RosettaCompanionAdtSynthesizer.CurrentProvenanceVersion, report.ReportVersion);
            Assert.Equal(2, report.TotalPm4FilesScanned);
            Assert.Equal(2, report.OrphanCount);
            Assert.Equal(2, report.SynthesizedCount);
            Assert.Equal(0, report.SkippedCount);
            Assert.Equal(0, report.FailedCount);
            Assert.Equal(2, report.Records.Count);

            Assert.True(File.Exists(reportPath));
            string json = File.ReadAllText(reportPath);
            using var doc = JsonDocument.Parse(json);
            Assert.Equal(RosettaCompanionAdtSynthesizer.CurrentProvenanceVersion, doc.RootElement.GetProperty("reportVersion").GetString());
            Assert.Equal(2, doc.RootElement.GetProperty("synthesizedCount").GetInt32());
        }
        finally
        {
            if (Directory.Exists(tempPm4Dir))
                Directory.Delete(tempPm4Dir, true);
            if (Directory.Exists(tempOutDir))
                Directory.Delete(tempOutDir, true);
        }
    }
}
