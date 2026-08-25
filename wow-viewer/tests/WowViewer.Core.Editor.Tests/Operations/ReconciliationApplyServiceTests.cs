using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Chunks;
using WowViewer.Core.Editor.Operations;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Reconciliation;

namespace WowViewer.Core.Editor.Tests.Operations;

public class ReconciliationApplyServiceTests
{
    private const string SourcePath = "synthetic_4_9_obj0.adt";
    private const string GuidePath = "synthetic_04_09.pm4";

    [Fact]
    public void Align_apply_moves_the_placement_and_reports_provenance()
    {
        byte[] source = BuildSyntheticAdt();
        var identity = new PlacementIdentity(SourcePath, "synthetic", 4, 9, ExpectedAssetKind.Model, 0, 77, "foo.mdx", "test");
        var proposal = new ReconciliationProposal(
            "prop-align",
            new Pm4GuideIdentity("guide-1", GuidePath, "test", "synthetic", 4, 9),
            identity,
            new PlacementSnapshot(identity, new Vector3(16866.666f, 16966.666f, 300f), Vector3.Zero, 1f),
            ReconciliationAction.Align,
            Candidate: null,
            ProposedPosition: new Vector3(16000f, 15000f, 300f),
            ProposedRotation: Vector3.Zero,
            ProposedScale: 1f,
            Residual: new Dictionary<string, double> { ["position"] = 0d },
            Confidence: 0d,
            Evidence: [new ReconciliationEvidence("pm4-segment-surfaces", 1, "test")],
            Status: ProposalStatus.ReviewRequired);

        string sourceHash = ReconciliationApplyService.ComputeSha256Hex(source);
        (AdtPlacementEditResult result, ReconciliationProvenanceReport report) =
            ReconciliationApplyService.ApplyAccepted(
                [proposal], source, SourcePath, sourceHash, GuidePath, guideSourceHash: null, "test-build");

        AdtPlacementCatalog catalog = Read(result.Bytes);
        AdtModelPlacement moved = Assert.Single(catalog.ModelPlacements);
        Assert.Equal(77, moved.UniqueId);
        Assert.Equal(new Vector3(16000f, 15000f, 300f), moved.Position);

        ReconciliationAppliedDecision decision = Assert.Single(report.Decisions);
        Assert.Equal("prop-align", decision.ProposalId);
        Assert.Equal("Align", decision.Action);
        Assert.Equal(77, decision.UniqueId);

        Assert.Equal(sourceHash, report.PlacementSourceHash);
        Assert.Equal(GuidePath, report.GuideSourcePath);
        Assert.Equal(ReconciliationApplyService.ComputeSha256Hex(result.Bytes), report.OutputHash);
        Assert.Empty(report.AllocatedIds);
        Assert.Equal(32, report.BatchId.Length); // deterministic 16-byte hex digest
    }

    [Fact]
    public void Clone_apply_adds_candidate_asset_with_fresh_reported_id()
    {
        byte[] source = BuildSyntheticAdt();
        var proposal = new ReconciliationProposal(
            "prop-clone",
            new Pm4GuideIdentity("guide-2", GuidePath, "test", "synthetic", 4, 9),
            ExistingPlacement: null,
            Current: null,
            ReconciliationAction.Clone,
            Candidate: new ReconciliationCandidate(
                "wmo:donor", "donor.wmo", ExpectedAssetKind.WorldModel, 0, 0.85d,
                new Dictionary<string, double>(), [], CandidateStatus.Matched),
            ProposedPosition: new Vector3(16000f, 15000f, 300f),
            ProposedRotation: Vector3.Zero,
            ProposedScale: 1f,
            Residual: new Dictionary<string, double>(),
            Confidence: 0.85d,
            Evidence: [],
            Status: ProposalStatus.ReviewRequired);

        string sourceHash = ReconciliationApplyService.ComputeSha256Hex(source);
        (AdtPlacementEditResult result, ReconciliationProvenanceReport report) =
            ReconciliationApplyService.ApplyAccepted(
                [proposal], source, SourcePath, sourceHash, GuidePath, null, "test-build");

        Assert.Equal([78], result.AllocatedIds);
        Assert.Equal(["donor.wmo"], result.AddedWorldModelNames);

        AdtPlacementCatalog catalog = Read(result.Bytes);
        AdtWorldModelPlacement added = Assert.Single(catalog.WorldModelPlacements);
        Assert.Equal(78, added.UniqueId);
        Assert.Equal("donor.wmo", added.ModelPath);
        Assert.Contains(78, report.AllocatedIds);
    }

    [Fact]
    public void Substitute_apply_replaces_asset_and_reports_allocation()
    {
        byte[] source = BuildSyntheticAdt();
        var identity = new PlacementIdentity(SourcePath, "synthetic", 4, 9, ExpectedAssetKind.Model, 0, 77, "foo.mdx", "test");
        var proposal = new ReconciliationProposal(
            "prop-substitute",
            new Pm4GuideIdentity("guide-3", GuidePath, "test", "synthetic", 4, 9),
            identity,
            new PlacementSnapshot(identity, new Vector3(16866.666f, 16966.666f, 300f), new Vector3(1f, 2f, 3f), 1.5f),
            ReconciliationAction.Substitute,
            Candidate: new ReconciliationCandidate(
                "wmo:a", "a.wmo", ExpectedAssetKind.WorldModel, 0, 0.9d,
                new Dictionary<string, double>(), [], CandidateStatus.Matched),
            ProposedPosition: new Vector3(16000f, 15000f, 300f),
            ProposedRotation: new Vector3(1f, 2f, 3f),
            ProposedScale: 1.5f,
            Residual: new Dictionary<string, double>(),
            Confidence: 0.9d,
            Evidence: [],
            Status: ProposalStatus.ReviewRequired);

        string sourceHash = ReconciliationApplyService.ComputeSha256Hex(source);
        (AdtPlacementEditResult result, _) = ReconciliationApplyService.ApplyAccepted(
            [proposal], source, SourcePath, sourceHash, GuidePath, null, "test-build");

        AdtPlacementCatalog catalog = Read(result.Bytes);
        Assert.Empty(catalog.ModelPlacements); // old MDDF row removed
        AdtWorldModelPlacement replacement = Assert.Single(catalog.WorldModelPlacements);
        Assert.Equal("a.wmo", replacement.ModelPath);
        Assert.Equal([78], result.AllocatedIds); // FR-003: the remap is reported
    }

    [Fact]
    public void Stale_source_hash_is_refused()
    {
        byte[] source = BuildSyntheticAdt();

        Assert.Throws<ReconciliationApplyException>(() => ReconciliationApplyService.ApplyAccepted(
            [CreateMinimalAlignProposal()], source, SourcePath,
            expectedSourceHash: "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            GuidePath, null, "test-build"));
    }

    [Fact]
    public void Empty_batch_is_refused()
    {
        byte[] source = BuildSyntheticAdt();
        string hash = ReconciliationApplyService.ComputeSha256Hex(source);

        Assert.Throws<ReconciliationApplyException>(() => ReconciliationApplyService.ApplyAccepted(
            [], source, SourcePath, hash, GuidePath, null, "test-build"));
    }

    private static ReconciliationProposal CreateMinimalAlignProposal()
    {
        var identity = new PlacementIdentity(SourcePath, "synthetic", 4, 9, ExpectedAssetKind.Model, 0, 77, "foo.mdx", "test");
        return new ReconciliationProposal(
            "prop-x",
            new Pm4GuideIdentity("guide-x", GuidePath, "test", "synthetic", 4, 9),
            identity,
            new PlacementSnapshot(identity, Vector3.Zero, Vector3.Zero, 1f),
            ReconciliationAction.Align,
            Candidate: null,
            ProposedPosition: Vector3.One,
            ProposedRotation: Vector3.Zero,
            ProposedScale: 1f,
            Residual: new Dictionary<string, double>(),
            Confidence: 0d,
            Evidence: [],
            Status: ProposalStatus.ReviewRequired);
    }

    private static AdtPlacementCatalog Read(byte[] bytes)
    {
        using MemoryStream stream = new(bytes);
        MapFileSummary summary = MapFileSummaryReader.Read(stream, SourcePath);
        return AdtPlacementReader.Read(stream, summary);
    }

    // Synthetic ADT with one MDDF row (uniqueId 77 at raw 100/200/300) and no WMO rows.
    private static byte[] BuildSyntheticAdt()
    {
        byte[] mmdx = CreateStringBlock("foo.mdx");
        byte[] mwmo = CreateStringBlock("a.wmo");
        byte[] mmid = CreateUInt32Array(0u);
        byte[] mwid = CreateUInt32Array(0u);
        byte[] mddf = CreateMddfEntry(nameId: 1u, uniqueId: 77u, rawX: 100f, rawY: 200f, rawZ: 300f, rotX: 0f, rotY: 0f, rotZ: 0f, scale: 1024);

        var parts = new List<byte[]>
        {
            CreateChunk("MVER", CreateUInt32Payload(18)),
            CreateChunk("MMDX", mmdx),
            CreateChunk("MMID", mmid),
            CreateChunk("MWMO", mwmo),
            CreateChunk("MWID", mwid),
            CreateChunk("MDDF", mddf),
        };

        return [.. parts.SelectMany(p => p)];
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        byte[] bytes = new byte[8 + payload.Length];
        Array.Copy(FourCC.FromString(id).ToFileBytes(), 0, bytes, 0, 4);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), (uint)payload.Length);
        Array.Copy(payload, 0, bytes, 8, payload.Length);
        return bytes;
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        return bytes;
    }

    private static byte[] CreateUInt32Array(params uint[] values)
    {
        byte[] bytes = new byte[values.Length * sizeof(uint)];
        for (int index = 0; index < values.Length; index++)
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(index * sizeof(uint), sizeof(uint)), values[index]);

        return bytes;
    }

    private static byte[] CreateStringBlock(params string[] entries)
    {
        using MemoryStream stream = new();
        foreach (string entry in entries)
        {
            byte[] bytes = System.Text.Encoding.ASCII.GetBytes(entry);
            stream.Write(bytes, 0, bytes.Length);
            stream.WriteByte(0);
        }

        return stream.ToArray();
    }

    private static byte[] CreateMddfEntry(uint nameId, uint uniqueId, float rawX, float rawY, float rawZ, float rotX, float rotY, float rotZ, ushort scale)
    {
        byte[] bytes = new byte[36];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), nameId);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), uniqueId);
        WriteSingle(bytes, 8, rawX);
        WriteSingle(bytes, 12, rawZ);
        WriteSingle(bytes, 16, rawY);
        WriteSingle(bytes, 20, rotX);
        WriteSingle(bytes, 24, rotZ);
        WriteSingle(bytes, 28, rotY);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(32, 2), scale);
        return bytes;
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }
}

public class ReconciliationApplyOperationTests
{
    [Fact]
    public void Forward_operation_writes_new_bytes_and_keeps_report()
    {
        var operation = new ReconciliationApplyOperation(
            "op-1", "pm4.reconciliation", "source.adt", "output.adt", "output.adt.reconciliation.json",
            priorOutputBytes: null,
            writtenOutputBytes: [1, 2, 3],
            reportJson: "{}");

        Assert.Same(operation.WrittenOutputBytes, operation.BytesToWrite);
        Assert.True(operation.ReportExists);
        Assert.True(operation.Undoable);
        Assert.Equal(["source.adt", "output.adt", "output.adt.reconciliation.json"], operation.AffectedPaths);
    }

    [Fact]
    public void Reverse_restores_prior_bytes_and_drops_report()
    {
        byte[] prior = [9, 8, 7];
        byte[] written = [1, 2, 3];
        var operation = new ReconciliationApplyOperation(
            "op-2", "pm4.reconciliation", "source.adt", "output.adt", "report.json",
            prior, written, "{}");

        EditorOperation reverse = operation.CreateReverse();
        var reconciliation = Assert.IsType<ReconciliationApplyOperation>(reverse);

        Assert.Same(prior, reconciliation.BytesToWrite);
        Assert.False(reconciliation.ReportExists);

        EditorOperation reverseOfReverse = reconciliation.CreateReverse();
        var restored = Assert.IsType<ReconciliationApplyOperation>(reverseOfReverse);
        Assert.Same(written, restored.BytesToWrite);
        Assert.True(restored.ReportExists);
    }

    [Fact]
    public void Reverse_of_created_output_removes_it()
    {
        var operation = new ReconciliationApplyOperation(
            "op-3", "pm4.reconciliation", "source.adt", "output.adt", "report.json",
            priorOutputBytes: null, writtenOutputBytes: [1], "{}");

        var reverse = Assert.IsType<ReconciliationApplyOperation>(operation.CreateReverse());

        Assert.Null(reverse.BytesToWrite);
        Assert.False(reverse.ReportExists);
    }
}
