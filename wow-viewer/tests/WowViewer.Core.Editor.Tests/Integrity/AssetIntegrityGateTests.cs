using WowViewer.Core.Editor.Integrity;

namespace WowViewer.Core.Editor.Tests.Integrity;

public class AssetIntegrityGateTests
{
    private sealed class FakeValidator : IAssetValidator
    {
        private readonly Func<AssetReadInput, AssetValidationResult> _validate;

        public FakeValidator(Func<AssetReadInput, AssetValidationResult> validate) => _validate = validate;

        public AssetValidationResult Validate(AssetReadInput input) => _validate(input);
    }

    [Fact]
    public void ValidateOnRead_returns_the_named_verdict()
    {
        var gate = new AssetIntegrityGate(new FakeValidator(_ =>
            AssetValidationResult.Quarantined("bad.adt", [new AssetDiagnostic("MDDF", "entry size not a multiple of 36")])));

        AssetValidationResult result = gate.ValidateOnRead("bad.adt", new byte[] { 1, 2, 3 });

        Assert.Equal(ValidationVerdict.Quarantined, result.Verdict);
        Assert.Equal("MDDF", result.Diagnostics[0].Constraint);
    }

    [Fact]
    public void AuthorizeWrite_refuses_when_any_input_is_unverified_or_quarantined()
    {
        var gate = new AssetIntegrityGate(new FakeValidator(_ => AssetValidationResult.Verified("ok")));

        var inputs = new[]
        {
            AssetValidationResult.Verified("a.adt"),
            AssetValidationResult.Unverified("b.adt"),
            AssetValidationResult.Quarantined("c.adt", [new AssetDiagnostic("MVER", "bad version")]),
        };

        IntegrityWriteDecision decision = gate.AuthorizeWrite(inputs);

        Assert.False(decision.IsApproved);
        Assert.Equal(2, decision.BlockingInputs.Count);
        Assert.Contains(decision.BlockingInputs, b => b.AssetPath == "b.adt");
        Assert.Contains(decision.BlockingInputs, b => b.AssetPath == "c.adt");
    }

    [Fact]
    public void AuthorizeWrite_approves_when_all_verified()
    {
        var gate = new AssetIntegrityGate(new FakeValidator(_ => AssetValidationResult.Verified("ok")));

        IntegrityWriteDecision decision = gate.AuthorizeWrite([AssetValidationResult.Verified("a.adt")]);

        Assert.True(decision.IsApproved);
        Assert.Empty(decision.BlockingInputs);
    }

    [Fact]
    public void VerifyWrittenFile_only_reports_saved_when_revalidated()
    {
        var gate = new AssetIntegrityGate(new FakeValidator(input =>
            input.Bytes.Length == 0
                ? AssetValidationResult.Unverified(input.AssetPath)
                : AssetValidationResult.Verified(input.AssetPath)));

        Assert.False(gate.VerifyWrittenFile(ReadOnlyMemory<byte>.Empty, "out.adt", out _));
        Assert.True(gate.VerifyWrittenFile(new byte[] { 1 }, "out.adt", out AssetValidationResult verified));
        Assert.Equal(ValidationVerdict.Verified, verified.Verdict);
    }

    [Fact]
    public void Provenance_is_required_and_complete()
    {
        var gate = new AssetIntegrityGate(new FakeValidator(_ => AssetValidationResult.Verified("ok")));

        Assert.False(gate.HasCompleteProvenance(null));

        var provenance = new AssetProvenance("3.3.5.12340", losses: ["downsampled specular map texture"]);

        Assert.True(gate.HasCompleteProvenance(provenance));
        Assert.Contains("downsampled specular map texture", provenance.Losses);
    }
}