using WowViewer.Core.Runtime.PromoVideo;

namespace WowViewer.Core.Tests.PromoVideo;

public sealed class PromoVideoOutputPolicyTests
{
    [Fact]
    public void TryCreateAuthoringHandoff_ConvertsManagedAbsolutePathsToRelativeReferences()
    {
        string root = Path.Combine(Path.GetTempPath(), "wow-viewer-promo", Guid.NewGuid().ToString("N"));
        string capture = Path.Combine(root, "Elwynn", "1.12", "flyby.mp4");
        string receipt = capture + ".tour-receipt.json";
        string still = Path.Combine(root, "Elwynn", "1.12", "flyby_001.png");

        AuthoringHandoffRequest request = new(
            "attempt-001",
            capture,
            receipt,
            [still],
            new AuthoringHandoffProvenance("camera-path-overview", "1", "Elwynn", "1.12", PromoCaptureTerminalOutcome.Completed));

        AuthoringHandoffValidationResult result = AuthoringHandoffFactory.TryCreate(root, request);

        Assert.True(result.IsValid, result.Error);
        Assert.NotNull(result.Handoff);
        Assert.False(Path.IsPathRooted(result.Handoff!.CaptureRelativePath));
        Assert.Equal(Path.Combine("Elwynn", "1.12", "flyby.mp4"), result.Handoff.CaptureRelativePath);
        Assert.All(result.Handoff.StillRelativePaths, path => Assert.False(Path.IsPathRooted(path)));
        Assert.DoesNotContain(root, System.Text.Json.JsonSerializer.Serialize(result.Handoff), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TryCreateAuthoringHandoff_RejectsTraversalOutsideManagedOutputRoot()
    {
        string root = Path.Combine(Path.GetTempPath(), "wow-viewer-promo", Guid.NewGuid().ToString("N"));
        string outside = Path.Combine(root, "..", "outside.mp4");
        AuthoringHandoffRequest request = new(
            "attempt-001",
            outside,
            Path.Combine(root, "receipt.json"),
            [],
            new AuthoringHandoffProvenance("camera-path-overview", "1", "Elwynn", "1.12", PromoCaptureTerminalOutcome.Completed));

        AuthoringHandoffValidationResult result = AuthoringHandoffFactory.TryCreate(root, request);

        Assert.False(result.IsValid);
        Assert.Equal("capture-outside-managed-root", result.ErrorCode);
        Assert.Null(result.Handoff);
    }

    [Fact]
    public void TryResolveManagedArtifact_RejectsRelativeTraversalAndAcceptsContainedPath()
    {
        string root = Path.Combine(Path.GetTempPath(), "wow-viewer-promo", Guid.NewGuid().ToString("N"));

        bool traversalAccepted = PromoVideoOutputPolicy.TryResolveManagedArtifact(root, "..\\outside.mp4", out _, out string traversalError);
        bool containedAccepted = PromoVideoOutputPolicy.TryResolveManagedArtifact(root, "Azeroth\\1.12\\tour.mp4", out ManagedPromoArtifact artifact, out string containedError);

        Assert.False(traversalAccepted);
        Assert.Equal("artifact-outside-managed-root", traversalError);
        Assert.True(containedAccepted, containedError);
        Assert.Equal(Path.Combine("Azeroth", "1.12", "tour.mp4"), artifact.RelativePath);
    }
}
