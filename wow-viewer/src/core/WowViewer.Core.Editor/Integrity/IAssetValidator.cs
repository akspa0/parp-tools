namespace WowViewer.Core.Editor.Integrity;

/// <summary>A single asset read input handed to a validator.</summary>
public readonly record struct AssetReadInput(string AssetPath, ReadOnlyMemory<byte> Bytes);

/// <summary>
/// Structural validation boundary. A validator either names the constraint an asset violated or
/// returns <see cref="ValidationVerdict.Unverified"/> when it cannot reach a verdict — it never
/// assumes a silent pass.
/// </summary>
public interface IAssetValidator
{
    AssetValidationResult Validate(AssetReadInput input);
}