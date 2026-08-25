namespace WowViewer.Core.Editor.Integrity;

public enum ValidationVerdict
{
    /// <summary>The input passed structural validation and is safe to contribute to a write.</summary>
    Verified,

    /// <summary>The input failed structural validation; it must be named and must not contribute to a write.</summary>
    Quarantined,

    /// <summary>No verdict could be reached. Treated as failing, never as passing (Spec 173 FR-002).</summary>
    Unverified,
}

/// <summary>A single violated constraint, named specifically rather than as a generic parse error.</summary>
public readonly record struct AssetDiagnostic(string Constraint, string Description)
{
    public override string ToString() => $"{Constraint}: {Description}";
}